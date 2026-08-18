from __future__ import annotations

from rag_app.bm25 import BM25Index
from rag_app.config import AppConfig, load_config
from rag_app.embed import Embedder
from rag_app.filters import MetaFilter
from rag_app.generate import (
    DONT_KNOW,
    Answer,
    cited_sources,
    generate_answer,
    is_refusal,
    sources_from_contexts,
)
from rag_app.hybrid import hybrid_retrieve
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.rerank import CrossEncoderReranker, rerank
from rag_app.retrieve import retrieve
from rag_app.store import SearchBackend, VectorStore, store_path_for_preset


def open_store(cfg: AppConfig, preset: str) -> SearchBackend:
    """Open the configured backend, or explain exactly how to build it."""
    if cfg.backend == "qdrant":
        path = qdrant_path_for_preset(cfg.store_dir, preset)
        if not cfg.qdrant.url and not path.exists():
            raise FileNotFoundError(
                f"No Qdrant store for preset {preset!r} at {path}. Run: "
                f"python -m rag_app ingest --preset {preset}"
            )
        store = QdrantStore(
            path=None if cfg.qdrant.url else path,
            url=cfg.qdrant.url,
            m=cfg.qdrant.m,
            ef_construct=cfg.qdrant.ef_construct,
            hnsw_ef=cfg.qdrant.hnsw_ef,
        )
        if not store.exists():
            store.close()
            raise FileNotFoundError(
                f"Qdrant collection for preset {preset!r} is empty. Run: "
                f"python -m rag_app ingest --preset {preset}"
            )
        return store

    path = store_path_for_preset(cfg.store_dir, preset)
    if not path.exists():
        raise FileNotFoundError(
            f"No store for preset {preset!r} at {path}. Run: "
            f"python -m rag_app ingest --preset {preset}"
        )
    store = VectorStore.load(path)
    if store.meta is not None:
        # Catches "changed the embedding model, forgot to re-ingest" before it
        # becomes a silently wrong ranking.
        store.meta.assert_compatible_with(cfg.bi_encoder_model)
    return store


def ask(
    question: str,
    preset: str | None = None,
    config: AppConfig | None = None,
    *,
    embedder: Embedder | None = None,
    reranker: CrossEncoderReranker | None = None,
    generate_fn=None,
    store: SearchBackend | None = None,
    flt: MetaFilter | None = None,
    bm25: BM25Index | None = None,
) -> Answer:
    cfg = config or load_config()
    name = preset or cfg.default_preset
    backend = store or open_store(cfg, name)

    emb = embedder or Embedder(cfg.bi_encoder_model)
    # Asymmetric models need the *query* prefix here, not the passage one.
    query_vec = emb.encode_queries([question])[0]

    if cfg.retrieval.mode == "hybrid":
        index = bm25 or BM25Index.from_store(backend)
        retrieved = hybrid_retrieve(
            backend,
            index,
            query_vec,
            question,
            k=cfg.retrieve_k,
            flt=flt,
            dense_pool=cfg.retrieve_k,
            bm25_pool=cfg.retrieval.bm25_pool,
            k_rrf=cfg.retrieval.rrf_k,
        )
    else:
        retrieved = retrieve(backend, query_vec, k=cfg.retrieve_k, flt=flt)

    rr = reranker or CrossEncoderReranker(cfg.cross_encoder_model)
    reranked = rerank(
        question, retrieved, n=cfg.rerank_n, scorer=rr, scale=cfg.rerank_score_scale
    )

    base_meta = {
        "preset": name,
        "backend": cfg.backend,
        "retrieval_mode": cfg.retrieval.mode,
        "filter": flt.describe() if flt else "(none)",
        "scale": cfg.rerank_score_scale,
        "threshold": cfg.score_threshold,
    }

    # ---- Gate 1: nothing survived retrieval/filtering ----------------------
    if not reranked:
        return Answer(
            text=DONT_KNOW,
            sources=[],
            best_score=float("-inf"),
            used_llm=False,
            retrieved=retrieved,
            reranked=reranked,
            refused=True,
            gate="no-candidates",
            meta=base_meta,
        )

    best_score = reranked[0].score

    # ---- Gate 2: best cross-encoder score below threshold ------------------
    # The LLM is never called. `used_llm=False` is the observable proof.
    if best_score < cfg.score_threshold:
        return Answer(
            text=DONT_KNOW,
            sources=[],
            best_score=best_score,
            used_llm=False,
            retrieved=retrieved,
            reranked=reranked,
            refused=True,
            gate="below-threshold",
            meta=base_meta,
        )

    gen = generate_fn or generate_answer
    text = gen(question, reranked, cfg)

    # ---- Gate 3: the model refused despite passing the score gate ----------
    # Attaching sources to a refusal would credit documents for a non-answer,
    # which is the exact ungrounded-citation failure the gate exists to stop.
    if is_refusal(text):
        return Answer(
            text=text,
            sources=[],
            best_score=best_score,
            used_llm=True,
            retrieved=retrieved,
            reranked=reranked,
            refused=True,
            gate="model-refused",
            meta=base_meta,
        )

    grounded, invented = cited_sources(text, reranked)
    # Fall back to the supplied excerpts only when the model cited nothing at
    # all, and record that it happened rather than papering over it.
    sources = grounded or sources_from_contexts(reranked)
    meta = dict(base_meta)
    meta["cited"] = bool(grounded)

    return Answer(
        text=text,
        sources=sources,
        best_score=best_score,
        used_llm=True,
        retrieved=retrieved,
        reranked=reranked,
        refused=False,
        hallucinated_citations=invented,
        gate="answered",
        meta=meta,
    )


def format_answer(answer: Answer, *, verbose: bool = True) -> str:
    score = "n/a" if answer.best_score == float("-inf") else f"{answer.best_score:.4f}"
    lines = [
        f"Answer: {answer.text}",
        f"Sources: {', '.join(answer.sources) if answer.sources else '(none)'}",
        (
            f"(K={len(answer.retrieved)} → N={len(answer.reranked)}; "
            f"best_score={score}; threshold={answer.meta.get('threshold')}; "
            f"used_llm={answer.used_llm}; gate={answer.gate})"
        ),
    ]
    if answer.hallucinated_citations:
        lines.append(
            f"!! HALLUCINATED CITATIONS (not in context): "
            f"{', '.join(answer.hallucinated_citations)}"
        )
    if answer.used_llm and not answer.refused and not answer.meta.get("cited"):
        lines.append("!! Model produced no citations; sources fell back to retrieved chunks.")

    if verbose:
        flt = answer.meta.get("filter", "(none)")
        if flt != "(none)":
            lines.append(f"\nFilter: {flt}")
        mode = answer.meta.get("retrieval_mode", "dense")
        label = "hybrid (RRF-fused)" if mode == "hybrid" else "bi-encoder, cosine"
        score_tag = "rrf" if mode == "hybrid" else "cos"
        lines.append(f"\nRetrieved ({label}):")
        for i, item in enumerate(answer.retrieved, start=1):
            preview = item.chunk.text.replace("\n", " ")[:110]
            bleed = " [BLEED]" if item.chunk.metadata.get("bleed") else ""
            lines.append(
                f"  {i}. [{item.chunk.source}]{bleed} {score_tag}={item.score:.4f} :: {preview}..."
            )
        lines.append("\nReranked (cross-encoder):")
        for i, item in enumerate(answer.reranked, start=1):
            preview = item.chunk.text.replace("\n", " ")[:110]
            bleed = " [BLEED]" if item.chunk.metadata.get("bleed") else ""
            lines.append(
                f"  {i}. [{item.chunk.source}]{bleed} ce={item.score:.4f} :: {preview}..."
            )
    return "\n".join(lines)
