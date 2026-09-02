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
from rag_app.mmr import apply_mmr
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.rerank import CrossEncoderReranker, build_reranker, rerank_all
from rag_app.retrieve import retrieve
from rag_app.rewrite import transform_query
from rag_app.store import SearchBackend


def open_store(cfg: AppConfig, preset: str) -> SearchBackend:
    """Open the Qdrant store for a preset, or explain exactly how to build it."""
    meta_dir = qdrant_path_for_preset(cfg.store_dir, preset)
    if not cfg.qdrant.url and not meta_dir.exists():
        raise FileNotFoundError(
            f"No index for preset {preset!r} at {meta_dir}. Add documents and build "
            f"it from the app's 'Add documents' tab (python -m rag_app ui)."
        )
    try:
        store = QdrantStore(
            meta_dir=meta_dir,
            path=None if cfg.qdrant.url else meta_dir,
            url=cfg.qdrant.url,
            m=cfg.qdrant.m,
            ef_construct=cfg.qdrant.ef_construct,
            hnsw_ef=cfg.qdrant.hnsw_ef,
        )
    except RuntimeError as exc:
        # Embedded Qdrant allows exactly one process to hold the index. Its own
        # message names a "Storage folder", which is accurate and unhelpful when
        # the actual cause is almost always the UI running in another window.
        if "already accessed" not in str(exc):
            raise
        raise RuntimeError(
            f"The index at {meta_dir} is open in another process — usually the "
            f"Streamlit app. Embedded Qdrant allows one reader at a time. Stop the "
            f"other process, or run a Qdrant server (docker run -p 6333:6333 "
            f"qdrant/qdrant) and set qdrant.url to allow concurrent access."
        ) from exc
    if not store.exists():
        store.close()
        raise FileNotFoundError(
            f"The index for preset {preset!r} is empty. Add documents and build it "
            f"from the app's 'Add documents' tab (python -m rag_app ui)."
        )
    meta = store.load_meta()
    if meta is not None:
        # Catches "changed the embedding model, forgot to re-ingest" before it
        # becomes a silently wrong ranking.
        meta.assert_compatible_with(cfg.bi_encoder_model)
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
    transform_fn=None,
) -> Answer:
    cfg = config or load_config()
    name = preset or cfg.default_preset
    backend = store or open_store(cfg, name)

    emb = embedder or Embedder(cfg.bi_encoder_model)

    # Query transform (rewrite / HyDE) happens BEFORE embedding: it changes
    # what we search for, never what we ask the model to answer. `question` is
    # still what reaches generation — answering the rewrite instead would let a
    # bad transform change the user's question without anyone noticing.
    transform = transform_query(question, cfg, transform_fn=transform_fn)
    search_text = transform.transformed

    # Asymmetric models need the *query* prefix here, not the passage one.
    query_vec = emb.encode_queries([search_text])[0]

    if cfg.retrieval.mode == "hybrid":
        index = bm25 or BM25Index.from_store(backend)
        retrieved = hybrid_retrieve(
            backend,
            index,
            query_vec,
            search_text,
            k=cfg.retrieve_k,
            flt=flt,
            dense_pool=cfg.retrieve_k,
            bm25_pool=cfg.retrieval.bm25_pool,
            k_rrf=cfg.retrieval.rrf_k,
        )
    else:
        retrieved = retrieve(backend, query_vec, k=cfg.retrieve_k, flt=flt)

    rr = reranker or build_reranker(cfg.cross_encoder_model)
    # One scoring pass over all K. `reranked` is what the LLM sees;
    # `reranked_all` keeps the demoted remainder for inspection at no extra
    # cost, since the cross-encoder had to score them to rank them anyway.
    #
    # Rerank against the ORIGINAL question, not the transform: the
    # cross-encoder's whole value is judging relevance to what was actually
    # asked, and a HyDE passage is a fabricated answer, not a question.
    reranked_all = rerank_all(question, retrieved, scorer=rr, scale=cfg.rerank_score_scale)

    if cfg.retrieval.mmr and reranked_all:
        # MMR re-selects among the reranked candidates for coverage. Its first
        # pick is always the most relevant, so `reranked[0]` — and therefore
        # the score gate below — is unchanged. See mmr.py.
        chunk_vectors = emb.encode_documents([c.chunk.text for c in reranked_all])
        reranked = apply_mmr(
            reranked_all, chunk_vectors, cfg.rerank_n, cfg.retrieval.mmr_lambda
        )
    else:
        reranked = reranked_all[: cfg.rerank_n]

    base_meta = {
        "preset": name,
        "backend": "qdrant",
        "qdrant_mode": "server" if cfg.qdrant.url else "embedded",
        "retrieval_mode": cfg.retrieval.mode,
        "filter": flt.describe() if flt else "(none)",
        "scale": cfg.rerank_score_scale,
        "threshold": cfg.score_threshold,
        "query_mode": transform.describe(),
        "search_text": search_text,
        "mmr": cfg.retrieval.mmr,
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
            reranked_all=reranked_all,
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
            reranked_all=reranked_all,
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
            reranked_all=reranked_all,
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
        reranked_all=reranked_all,
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
            lines.append(
                f"  {i}. [{item.chunk.source}] {score_tag}={item.score:.4f} :: {preview}..."
            )
        lines.append("\nReranked (cross-encoder):")
        for i, item in enumerate(answer.reranked, start=1):
            preview = item.chunk.text.replace("\n", " ")[:110]
            lines.append(
                f"  {i}. [{item.chunk.source}] ce={item.score:.4f} :: {preview}..."
            )
    return "\n".join(lines)
