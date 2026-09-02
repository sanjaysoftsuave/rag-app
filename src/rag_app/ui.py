"""Streamlit front end — the pipeline with its lid off.

This is a second surface over the same `ask()`, not a second implementation.
Every number shown comes from the Answer that `ask()` returned, for the same
reason nothing here re-derives pipeline behaviour: a debugging view that
reimplements the thing it debugs will eventually disagree with it, and the
disagreement is invisible.

Timing is measured through the dependency-injection seam rather than by
instrumenting the pipeline — `ask()` already accepts `embedder`, `reranker`,
`generate_fn` and `store`, so wrapping each in a stopwatch needs no change to
pipeline.py and cannot alter what the pipeline computes.

Run it with `python -m rag_app ui`, or `streamlit run app.py`.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

from rag_app.bm25 import BM25Index
from rag_app.chunking import chunk_docs
from rag_app.config import AppConfig, ChunkPreset, load_config
from rag_app.docs import load_docs
from rag_app.embed import Embedder, spec_for
from rag_app.evaluate import (
    FAILURE_BUCKETS,
    evaluate,
    gold_path,
    label_failures,
    load_gold,
    report_to_json,
)
from rag_app.explain import (
    audit_citations,
    build_candidate_table,
    config_rows,
    corpus_mix,
    explain_gate,
    prompt_messages,
    query_encoding,
    retrieval_score_label,
)
from rag_app.filters import MetaFilter
from rag_app.generate import generate_answer
from rag_app.ingest import run_ingest
from rag_app.pdfs import load_pdfs
from rag_app.pipeline import ask, open_store
from rag_app.qdrant_store import qdrant_path_for_preset, read_provenance
from rag_app.rerank import build_reranker
from rag_app.store import StoreMeta

UPLOAD_TYPES = ["pdf", "md", "markdown", "txt"]


def skip_generation(question, contexts, cfg) -> str:
    """Stand-in `generate_fn` for retrieval-only runs.

    Never touches the network, but still lets the real score gate run, so
    `gate` stays meaningful with generation switched off.
    """
    return "(generation skipped — retrieval only; tick 'Call the LLM' to answer)"


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _models(bi_encoder: str, cross_encoder: str):
    """Load both torch models once per process, keyed by checkpoint name.

    Constructing these per rerun would reload hundreds of MB of torch weights
    on every widget interaction.
    """
    return Embedder(bi_encoder), build_reranker(cross_encoder)


@st.cache_resource(show_spinner=False)
def _open_handles() -> dict:
    """A registry of open stores/indexes we control explicitly.

    Embedded Qdrant holds a real file lock while open. Ingest writes to that
    same directory, so the UI must be able to *close* a handle on demand, not
    merely evict it from a cache whose finalisation timing is not ours to
    choose. Hence a plain dict we own rather than one @cache_resource per store.

    Keys are `(kind, preset, model)` tuples rather than formatted strings so
    `close_handles(preset)` can match on a field instead of substring-guessing.
    """
    return {}


def get_store(cfg: AppConfig, preset: str):
    handles = _open_handles()
    key = ("store", preset, cfg.bi_encoder_model)
    if key not in handles:
        handles[key] = open_store(cfg, preset)
    return handles[key]


def get_bm25(cfg: AppConfig, preset: str) -> BM25Index:
    handles = _open_handles()
    key = ("bm25", preset, cfg.bi_encoder_model)
    if key not in handles:
        handles[key] = BM25Index.from_store(get_store(cfg, preset))
    return handles[key]


def close_handles(preset: str | None = None) -> None:
    """Release file locks before anything writes to the store directory."""
    handles = _open_handles()
    doomed = [k for k in list(handles) if preset is None or k[1] == preset]
    # BM25 indexes are derived from a store, so drop them before the store
    # they were built from.
    for key in sorted(doomed, key=lambda k: k[0] != "bm25"):
        handle = handles.pop(key, None)
        close = getattr(handle, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # an already-closed handle must not break the page
                pass


# ---------------------------------------------------------------------------
# Stopwatch proxies — they delegate, they never change behaviour
# ---------------------------------------------------------------------------


class _Timed:
    """Base proxy: forwards everything it does not explicitly time."""

    def __init__(self, inner, sink: dict, label: str):
        self._inner = inner
        self._sink = sink
        self._label = label

    def _record(self, started: float) -> None:
        self._sink[self._label] = self._sink.get(self._label, 0.0) + (
            time.perf_counter() - started
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedEmbedder(_Timed):
    def encode_queries(self, texts):
        started = time.perf_counter()
        try:
            return self._inner.encode_queries(texts)
        finally:
            self._record(started)

    def encode_documents(self, texts):
        return self._inner.encode_documents(texts)


class TimedReranker(_Timed):
    def predict(self, pairs):
        started = time.perf_counter()
        try:
            return self._inner.predict(pairs)
        finally:
            self._record(started)


class TimedStore(_Timed):
    def search(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.search(*args, **kwargs)
        finally:
            self._record(started)

    def __len__(self):
        return len(self._inner)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

KIND_ICON = {"doc": "📄", "pdf": "📕"}
GATE_STYLE = {
    "answered": ("✅", "success"),
    "below-threshold": ("🛑", "warning"),
    "no-candidates": ("🛑", "warning"),
    "model-refused": ("🤷", "info"),
}


def _preview(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


def render_gate(gate, timings: dict, generated: bool) -> None:
    """`generated` is False when the LLM was stubbed out.

    Without it this panel would report `LLM called: yes` for a run that never
    touched the network — the stub still passes through `ask()`'s generation
    step, so `used_llm` is True in the strict sense and misleading in every
    sense that matters here.
    """
    icon, style = GATE_STYLE.get(gate.gate, ("•", "info"))
    headline = gate.headline
    if not generated and gate.gate == "answered":
        icon, style = "⏭️", "info"
        headline = "Gate passed — the LLM would have been called"
    getattr(st, style)(f"{icon}  **{headline}**  ·  `gate={gate.gate}`")

    cols = st.columns(4)
    if gate.best_score is None:
        cols[0].metric("best rerank score", "n/a")
    else:
        cols[0].metric(
            "best rerank score",
            f"{gate.best_score:.4f}",
            delta=f"{gate.margin:+.4f} vs threshold",
            delta_color="normal",
        )
    cols[1].metric("threshold", f"{gate.threshold}")
    if not generated:
        cols[2].metric("LLM called", "skipped")
    else:
        cols[2].metric("LLM called", "yes" if gate.llm_called else "no")
    cols[3].metric("total time", f"{timings.get('total', 0):.2f}s")
    st.caption(gate.detail)
    if not generated and gate.gate == "answered":
        st.caption(
            "Generation is off, so nothing was actually written — but retrieval, reranking "
            "and the score gate all ran for real, and this question cleared them."
        )


def render_timings(timings: dict) -> None:
    total = timings.get("total", 0.0)
    measured = {k: v for k, v in timings.items() if k != "total"}
    rows = [
        {"stage": k, "seconds": round(v, 3), "% of total": f"{(100 * v / total):.0f}%" if total else "—"}
        for k, v in measured.items()
    ]
    other = total - sum(measured.values())
    if other > 0:
        rows.append(
            {
                "stage": "everything else",
                "seconds": round(other, 3),
                "% of total": f"{(100 * other / total):.0f}%" if total else "—",
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_retrieval(answer, rows) -> None:
    label = retrieval_score_label(answer)
    if label == "rrf":
        st.caption(
            "Hybrid mode: this score is a **Reciprocal Rank Fusion** value built from rank "
            "positions in the dense and BM25 rankings — not a cosine similarity, and not "
            "comparable to one. See hybrid.py."
        )
    else:
        st.caption(
            "Dense retrieval: cosine similarity between the query vector and each chunk "
            "vector. Vectors are pre-normalized, so this is a plain dot product."
        )

    table = [
        {
            "#": r.dense_rank,
            "source": f"{KIND_ICON.get(r.kind, '•')} {r.source}",
            "where": r.detail,
            label: None if r.dense_score is None else round(r.dense_score, 4),
            "preview": _preview(r.text, 120),
        }
        for r in sorted(
            [r for r in rows if r.dense_rank is not None], key=lambda r: r.dense_rank
        )
    ]
    st.dataframe(table, width="stretch", hide_index=True)


def render_rerank(rows, cfg: AppConfig) -> None:
    st.caption(
        f"The cross-encoder re-scores every retrieved candidate by reading it *together* "
        f"with the question — something the bi-encoder structurally cannot do — and the top "
        f"{cfg.rerank_n} go to the LLM. Rows marked **✘ cut** were scored and discarded; they "
        f"never reached the model's context. Scores are on the `{cfg.rerank_score_scale}` "
        f"scale, and the logit column is the model's raw unbounded output behind it."
    )
    scored = [r for r in rows if r.rerank_rank is not None]
    table = [
        {
            "#": r.rerank_rank,
            "in context": "✔" if r.in_context else "✘ cut",
            "source": f"{KIND_ICON.get(r.kind, '•')} {r.source}",
            "score": None if r.rerank_score is None else round(r.rerank_score, 4),
            "logit": "saturated" if r.rerank_logit is None else round(r.rerank_logit, 2),
            "was #": r.dense_rank,
            "moved": r.movement,
            "preview": _preview(r.text, 110),
        }
        for r in sorted(scored, key=lambda r: r.rerank_rank)
    ]
    st.dataframe(table, width="stretch", hide_index=True)

    promoted = [r for r in scored if (r.rank_delta or 0) > 0 and r.in_context]
    demoted = [r for r in scored if (r.rank_delta or 0) < 0 and not r.in_context]
    if promoted or demoted:
        bits = []
        if promoted:
            bits.append(
                "promoted into context: "
                + ", ".join(f"`{r.source}` ({r.movement})" for r in promoted)
            )
        if demoted:
            bits.append(
                "demoted out of context: "
                + ", ".join(f"`{r.source}` ({r.movement})" for r in demoted)
            )
        st.markdown("**Rerank changed the outcome** — " + "; ".join(bits) + ".")
    else:
        st.markdown(
            "_The reranker did not change which chunks reached the LLM — "
            "the bi-encoder's ordering already agreed._"
        )


def render_citations(answer) -> None:
    audit = audit_citations(answer)
    if audit.invented:
        st.error(
            "**Hallucinated citations** — the model cited "
            + ", ".join(f"`{c}`" for c in audit.invented)
            + ", which were never in its context. This is the failure "
            "`cited_sources()` exists to catch."
        )
    if audit.fell_back:
        st.warning(
            "The model produced **no citations at all**, so the sources listed are the "
            "excerpts it was given, not sources it claimed. The answer looks sourced "
            "without the model ever having attributed anything."
        )
    if audit.grounded:
        st.success("Grounded citations: " + ", ".join(f"`{c}`" for c in audit.grounded))
    elif not audit.invented and not audit.fell_back:
        st.caption("No citations — expected, since this answer was a refusal.")


def render_chunk_texts(answer) -> None:
    if not answer.reranked:
        st.caption("Nothing reached the context window.")
        return
    for i, item in enumerate(answer.reranked, start=1):
        meta = item.chunk.metadata or {}
        kind = str(meta.get("source_type") or "ticket")
        with st.expander(
            f"{i}. {KIND_ICON.get(kind, '•')} [{item.chunk.source}] · score {item.score:.4f}"
        ):
            st.code(item.chunk.text, language=None)
            st.json({k: v for k, v in meta.items() if k not in ("text",)}, expanded=False)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def chunking_controls(
    current: ChunkPreset, indexed: StoreMeta | None, bi_encoder: str = ""
) -> ChunkPreset:
    """Size and overlap — the only two knobs chunking has.

    They default to the store's own provenance rather than to config.yaml, so
    the sidebar describes what you are really querying and a divergence is a
    genuine "you have not re-ingested yet" rather than an artefact of the page
    having been reloaded.

    Chunking is an ingest-time decision: `ask()` never reads these values, so
    nothing changes until the index is rebuilt.
    """
    st.subheader("Chunking")
    st.caption("Applies at **ingest** time — rebuild the index for changes to take effect.")

    base = (
        ChunkPreset(indexed.chunk_size, indexed.overlap) if indexed is not None else current
    )
    spec = spec_for(bi_encoder) if bi_encoder else None
    ceiling = spec.max_chars if spec and spec.max_tokens else 0
    help_text = (
        "Characters, not tokens. The embedding model truncates at its own token "
        "ceiling — past that the tail of every chunk is silently dropped before it "
        "is ever embedded."
    )
    if ceiling:
        help_text += (
            f" `{spec.name}` stops at {spec.max_tokens} tokens, roughly "
            f"{ceiling} characters."
        )
    chunk_size = st.slider(
        "chunk_size (characters)", 200, 4000, int(base.chunk_size), step=50, help=help_text,
    )
    if ceiling and chunk_size > ceiling:
        st.warning(
            f"⚠️ Above the embedder's ~{ceiling}-character ceiling — text past it is "
            f"dropped before embedding, so it can never be retrieved."
        )
    # chunk_text() raises if overlap >= chunk_size, so the widget cannot offer it.
    overlap = st.slider(
        "overlap (characters)", 0, max(0, chunk_size - 50),
        min(int(base.overlap), chunk_size - 50), step=10,
        help="Stops a fact being split across two chunks so neither holds it whole. "
             "Costs storage and can put two near-identical chunks in your top-K.",
    )
    return ChunkPreset(chunk_size=chunk_size, overlap=overlap)


def render_index_drift(chunking: ChunkPreset, indexed: StoreMeta | None) -> None:
    """Say plainly when the sidebar no longer describes the live index.

    Without this the chunking sliders look broken: you move them, nothing about
    the answers changes, and there is no clue that a rebuild is the missing step.
    """
    if indexed is None:
        return
    live = ChunkPreset(indexed.chunk_size, indexed.overlap)
    if live == chunking:
        st.caption(
            f"Index: **{indexed.n_chunks} chunks**, `{live.describe()}`, "
            f"`{indexed.embedding_model}` ({indexed.dim}d) — matches the settings above."
        )
        return

    changed = []
    if live.chunk_size != chunking.chunk_size:
        changed.append(f"chunk_size `{live.chunk_size}` → `{chunking.chunk_size}`")
    if live.overlap != chunking.overlap:
        changed.append(f"overlap `{live.overlap}` → `{chunking.overlap}`")
    st.warning(
        "⚠️ **The index does not match these settings.**\n\n"
        + "\n".join(f"- {c}" for c in changed)
        + f"\n\nIndexed: **{indexed.n_chunks} chunks**. Queries keep using the indexed "
        f"chunking until you run **Add documents → Run ingest**."
    )


def tab_ask(cfg: AppConfig, preset: str, flt: MetaFilter, do_generate: bool) -> None:
    question = st.text_input(
        "Question",
        placeholder="What is the rate limit on the Free plan?",
        key="question",
    )
    run = st.button("Ask", type="primary", width="content")

    if not question.strip():
        st.caption(
            "Ask something to see the full trace: what was retrieved, how the cross-encoder "
            "reordered it, which candidates were cut, why the gate fired the way it did, and "
            "the exact prompt the model received."
        )
        return

    try:
        store = get_store(cfg, preset)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Build it from the **Add documents** tab, or run the command it names in a terminal.")
        return

    timings: dict[str, float] = {}
    try:
        embedder, reranker = _models(cfg.bi_encoder_model, cfg.cross_encoder_model)
    except Exception as exc:  # model download / torch problems are worth naming
        st.error(f"Could not load the embedding or reranking model: {exc}")
        return

    # Streamlit reruns the whole script on every widget change, so the sliders
    # are only useful if the pipeline re-runs with them. Retrieval, reranking
    # and the gate are cheap and always re-run — that is what makes dragging
    # `score_threshold` and watching the gate flip work at all.
    #
    # Generation is neither cheap nor idempotent, so it is called ONLY on an
    # explicit click, and its result is memoised on (question, exact context
    # set, model). Nudging the threshold afterwards therefore keeps the answer
    # on screen without paying for it twice; changing K or N changes the
    # context set, the memo misses, and the answer honestly disappears rather
    # than being shown next to excerpts that did not produce it.
    cache: dict = st.session_state.setdefault("_generation_cache", {})
    produced_real_text = {"value": False}
    allow_call = do_generate and run

    def generate_fn(q, contexts, config):
        key = (q, tuple(c.chunk.chunk_id for c in contexts), config.llm.model)
        if key in cache:
            produced_real_text["value"] = True
            return cache[key]
        if not allow_call:
            # Retrieval-only: no network, but the real gate still runs, so
            # `gate` stays meaningful and the trace is honest about what ran.
            return skip_generation(q, contexts, config)
        started = time.perf_counter()
        try:
            text = generate_answer(q, contexts, config)
        finally:
            timings["generate (LLM)"] = time.perf_counter() - started
        cache[key] = text
        produced_real_text["value"] = True
        return text

    bm25 = get_bm25(cfg, preset) if cfg.retrieval.mode == "hybrid" else None

    started = time.perf_counter()
    try:
        with st.spinner("Running the pipeline…"):
            answer = ask(
                question,
                preset=preset,
                config=cfg,
                store=TimedStore(store, timings, "retrieve (vector search)"),
                embedder=TimedEmbedder(embedder, timings, "embed query"),
                reranker=TimedReranker(reranker, timings, "rerank (cross-encoder)"),
                generate_fn=generate_fn,
                flt=flt or None,
                bm25=bm25,
            )
    except Exception as exc:
        st.error(f"The pipeline raised: {exc}")
        return
    timings["total"] = time.perf_counter() - started

    gate = explain_gate(answer, cfg)
    rows = build_candidate_table(answer)
    generated = produced_real_text["value"]

    st.markdown("### Answer")
    if answer.refused:
        st.info(answer.text)
    elif not generated:
        st.info(
            "**No answer was written and no API call was made.** Everything below — "
            "retrieval, reranking, the score gate — ran for real.\n\n"
            + (
                "Press **Ask** to generate one for this context set."
                if do_generate
                else "Tick *Call the LLM* in the sidebar, then press **Ask**."
            )
        )
    else:
        st.markdown(answer.text)

    # Citations are only meaningful when a real model produced the text. The
    # stub emits none, and reporting that as "the model cited nothing" would
    # invent a finding that says something about this app rather than the model.
    if generated:
        render_citations(answer)

    st.divider()
    render_gate(gate, timings, generated)

    st.divider()
    st.markdown("### How that answer was produced")

    with st.expander("① Configuration in effect", expanded=False):
        st.dataframe(
            [
                {"setting": k, "value": v}
                for k, v in config_rows(
                    cfg, preset, read_provenance(qdrant_path_for_preset(cfg.store_dir, preset))
                )
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Metadata filter: `{flt.describe() if flt else '(none)'}`")

    with st.expander("② Query encoding — the string the bi-encoder actually saw", expanded=False):
        searched = answer.meta.get("search_text", question)
        if searched != question:
            st.info(
                f"**The question was transformed before retrieval "
                f"(`{answer.meta.get('query_mode')}`).** Retrieval searched for the text "
                f"below; generation still answered the question you typed."
            )
            st.code(searched, language=None)
        elif answer.meta.get("query_mode", "off") != "off":
            st.caption(f"Query transform: `{answer.meta.get('query_mode')}`")
        enc = query_encoding(cfg, searched)
        st.dataframe(
            [{"field": k, "value": v} for k, v in enc.items()],
            width="stretch",
            hide_index=True,
        )
        if enc["query_prefix"] != "(none)":
            st.info(
                "This model is **asymmetric**: the query needs a prefix the passages do not "
                "get. Omitting it does not error — it just quietly retrieves worse. That is "
                "the whole reason `encode_queries` and `encode_documents` are separate."
            )

    with st.expander(
        f"③ Retrieval — {len(answer.retrieved)} candidates (K={cfg.retrieve_k})", expanded=True
    ):
        render_retrieval(answer, rows)
        mix = corpus_mix([r for r in rows if r.dense_rank is not None])
        if mix:
            st.caption(
                "Candidate mix: "
                + ", ".join(f"{KIND_ICON.get(k, '•')} {v} {k}" for k, v in sorted(mix.items()))
            )

    with st.expander(
        f"④ Reranking — {len(answer.reranked)} of {len(answer.reranked_all or answer.reranked)} survive to context",
        expanded=True,
    ):
        if answer.meta.get("mmr"):
            st.caption(
                f"**MMR is on** (lambda {cfg.retrieval.mmr_lambda}). The final selection "
                f"below is not simply the top {cfg.rerank_n} by score — a candidate is "
                f"penalised for resembling one already chosen. The top pick is still the "
                f"most relevant, so the gate is unaffected."
            )
        render_rerank(rows, cfg)

    with st.expander("⑤ The score gate", expanded=False):
        st.markdown(f"**{gate.headline}** — `{gate.gate}`")
        st.caption(gate.detail)
        st.markdown(
            "The four paths through `ask()`:\n\n"
            "- `no-candidates` — retrieval returned nothing. LLM never called.\n"
            "- `below-threshold` — best score under `score_threshold`. **LLM never called.**\n"
            "- `model-refused` — LLM was called, read the excerpts, and declined. Sources stripped.\n"
            "- `answered` — LLM was called and answered."
        )

    if answer.used_llm:
        sent = "sent to" if generated else "that WOULD be sent to"
        with st.expander(f"⑥ The exact prompt {sent} the model", expanded=False):
            if not generated:
                st.caption("Built here for display only — this exact prompt was never sent.")
            for message in prompt_messages(question, answer.reranked):
                st.markdown(f"**{message['role']}**")
                st.code(message["content"], language=None)
            st.caption(
                "Each excerpt is labelled with the same token the model is asked to cite. "
                "Labelling them [1]/[2] while demanding [handbook.pdf] would teach the wrong "
                "format by example."
            )
    else:
        with st.expander("⑥ Prompt — never built", expanded=False):
            st.caption(
                "The gate refused before generation, so no prompt was ever constructed. "
                "That is the observable signal `used_llm=False` exists to give you: this "
                "refusal cost nothing and involved no model judgment."
            )

    with st.expander("⑦ Full text of what reached the context window", expanded=False):
        render_chunk_texts(answer)

    with st.expander("⑧ Timing breakdown", expanded=False):
        render_timings(timings)
        st.caption(
            "Measured by wrapping the injected embedder, store and reranker in stopwatches — "
            "the pipeline itself is not instrumented and behaves identically."
        )


def tab_corpus(cfg: AppConfig, preset: str) -> None:
    st.markdown("### Chunking — what the corpus becomes before anything is embedded")
    st.caption(
        "No models load here, so this is instant. Every source is windowed as one file, "
        "`.md`/`.txt` and `.pdf` alike, so a chunk can never span two sources and a "
        "citation always names the one file its text came from."
    )

    docs = load_docs(cfg.tickets_dir)
    pdf_docs, pdf_reports = ([], [])
    try:
        pdf_docs, pdf_reports = load_pdfs(cfg.tickets_dir)
    except RuntimeError as exc:
        st.warning(str(exc))

    cols = st.columns(3)
    cols[0].metric("documents (.md/.txt)", len(docs))
    cols[1].metric("PDFs", len(pdf_reports))
    cols[2].metric("PDF pages of text", sum(r.n_text_pages for r in pdf_reports))

    unreadable = [r for r in pdf_reports if r.looks_scanned or r.encrypted]
    if unreadable:
        st.error(
            "These PDFs contribute **nothing** to the corpus:\n\n"
            + "\n".join(f"- {r.describe()}" for r in unreadable)
            + "\n\nA scanned PDF has no text layer to extract. OCR it first; this app does not."
        )

    def all_chunks(pc: ChunkPreset):
        return (
            chunk_docs(docs, pc.chunk_size, pc.overlap)
            + chunk_docs(pdf_docs, pc.chunk_size, pc.overlap, source_type="pdf")
        )

    single = len(cfg.chunk_presets) == 1
    st.markdown("#### Chunking" if single else "#### Every preset, side by side")
    table = []
    for name in sorted(cfg.chunk_presets):
        pc = cfg.chunk_presets[name]
        chunks = all_chunks(pc)
        table.append({
            "preset": name + (" ←" if name == preset else ""),
            "size": pc.chunk_size,
            "overlap": pc.overlap,
            "chunks": len(chunks),
            "doc": sum(1 for c in chunks if c.metadata.get("source_type") == "doc"),
            "pdf": sum(1 for c in chunks if c.metadata.get("source_type") == "pdf"),
            "avg chars": round(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0,
        })
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Smaller chunks retrieve more precisely and give the LLM less surrounding context; "
        "larger ones do the reverse, up to the embedder's token ceiling, past which the "
        "excess is not stored at all. Preview here before spending the embedding time."
    )

    st.markdown("#### Browse chunks" if single else f"#### Browse chunks for preset {preset}")
    allc = all_chunks(cfg.chunk_presets[preset])
    if not allc:
        st.info("No chunks — add documents in the **Add documents** tab.")
        return
    limit = st.slider("How many to show", 1, min(100, len(allc)), min(10, len(allc)))
    for c in allc[:limit]:
        kind = str(c.metadata.get("source_type") or "doc")
        with st.expander(f"{KIND_ICON.get(kind, '•')} [{c.source}] · {len(c.text)} chars"):
            st.code(c.text, language=None)


def tab_ingest(cfg: AppConfig, preset: str) -> None:
    st.markdown("### Add documents and rebuild the index")
    st.caption(
        f"Files are written to `{cfg.tickets_dir}` — the one corpus folder. "
        "`.md`, `.txt` and `.pdf` all load the same way: one file, windowed as a unit, "
        "cited by its filename. A PDF's pages are joined before windowing, so a fact "
        "spanning a page break stays whole."
    )

    uploaded = st.file_uploader(
        "Drop files here", type=UPLOAD_TYPES, accept_multiple_files=True
    )
    if uploaded and st.button("Save to corpus", type="secondary"):
        cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for item in uploaded:
            # Only ever the basename: an uploaded name is untrusted input and
            # "../../config.yaml" must not be able to escape the corpus folder.
            safe = Path(item.name).name
            if not safe or safe.startswith("."):
                st.warning(f"Skipped {item.name!r} — unusable filename.")
                continue
            (cfg.tickets_dir / safe).write_bytes(item.getbuffer())
            saved.append(safe)
        if saved:
            st.success(
                f"Saved {len(saved)} file(s): {', '.join(saved)}. "
                "Now re-ingest below so they enter the index."
            )

    existing = sorted(p.name for p in cfg.tickets_dir.glob("*") if p.is_file()) \
        if cfg.tickets_dir.exists() else []
    if existing:
        with st.expander(f"Corpus folder contents ({len(existing)} files)", expanded=False):
            st.write(existing)

    st.divider()
    st.markdown("#### Re-ingest")
    st.caption(
        "Chunk → embed → persist. Re-run this after adding files, changing the chunking "
        "settings, or switching the embedding model — the store records which model built "
        "it and refuses to serve queries from a different one."
    )
    will_use = cfg.chunk_presets[preset]
    st.info(
        f"Will rebuild at **chunk_size `{will_use.chunk_size}` / overlap "
        f"`{will_use.overlap}`** — change those in the sidebar. "
        f"See the **Corpus & chunking** tab to preview the resulting chunk counts "
        f"*before* spending the embedding time."
    )
    names = sorted(cfg.chunk_presets)
    targets = (
        names
        if len(names) == 1
        else st.multiselect("Presets to rebuild", names, default=[preset])
    )
    if st.button("Run ingest", type="primary", disabled=not targets):
        try:
            embedder, _ = _models(cfg.bi_encoder_model, cfg.cross_encoder_model)
        except Exception as exc:
            st.error(f"Could not load the embedding model: {exc}")
            return
        progress = st.progress(0.0)
        for i, name in enumerate(targets, start=1):
            # Embedded Qdrant holds a file lock; the writer cannot open the
            # directory while this process still has a reader on it.
            close_handles(name)
            try:
                with st.spinner(f"Ingesting preset {name}…"):
                    report = run_ingest(preset=name, config=cfg, embedder=embedder)
            except Exception as exc:
                st.error(f"Preset {name} failed: {exc}")
                continue
            # Memoised answers are keyed on chunk_id, which is stable across a
            # re-ingest while the text behind it is not. Keeping them would show
            # an old answer beside the new excerpts it was not generated from.
            st.session_state.pop("_generation_cache", None)
            st.success(f"Preset {name} rebuilt.")
            st.code(report.describe(), language=None)
            for pdf in report.pdf_reports:
                (st.error if (pdf.looks_scanned or pdf.encrypted) else st.caption)(
                    pdf.describe()
                )
            progress.progress(i / len(targets))
        progress.progress(1.0)


def tab_evaluate(cfg: AppConfig, preset: str) -> None:
    st.markdown("### Measure it")
    st.caption(
        "Every other panel shows what happened for ONE question. This scores a whole "
        "set of questions whose answers you already know — the only way to tell whether "
        "a change to chunking, the embedder or the threshold actually helped."
    )

    path = gold_path(cfg)
    if not path.exists():
        st.info(
            f"**No gold set yet.** Metrics need questions whose correct answers you "
            f"already know; there is no useful default, because every entry has to be "
            f"checked against your own documents.\n\n"
            f"Create `{path}` — see `gold.example.yaml` at the repo root for the format."
        )
        return

    try:
        gold = load_gold(cfg, path)
    except ValueError as exc:
        st.error(f"The gold set could not be read: {exc}")
        return

    answerable = [g for g in gold if g.answerable]
    cols = st.columns(3)
    cols[0].metric("gold questions", len(gold))
    cols[1].metric("answerable", len(answerable))
    cols[2].metric("must be refused", len(gold) - len(answerable))
    st.caption(f"Loaded from `{path}`")

    generate = st.checkbox(
        "Also call the LLM",
        value=False,
        help="Scores answer accuracy and resolves 'unconfirmed' into pass/generation. "
             "Costs one API call per answerable question.",
    )
    if not st.button("Run evaluation", type="primary"):
        return

    try:
        store = get_store(cfg, preset)
        embedder, reranker = _models(cfg.bi_encoder_model, cfg.cross_encoder_model)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not load models: {exc}")
        return

    with st.spinner(f"Running {len(gold)} questions through the real pipeline…"):
        report = evaluate(
            gold, preset=preset, config=cfg, store=store,
            embedder=embedder, reranker=reranker,
            use_llm=generate, generate_fn=generate_answer if generate else None,
        )
        failures = label_failures(
            gold, preset=preset, config=cfg, store=store,
            embedder=embedder, reranker=reranker,
            use_llm=generate, generate_fn=generate_answer if generate else None,
        )

    st.markdown("#### Retrieval")
    cols = st.columns(4)
    cols[0].metric(f"hit-rate@{report.k}", f"{report.hit_rate:.0%}",
                   help="Some expected text reached the funnel at all.")
    cols[1].metric(f"recall@{report.k}", f"{report.recall_at_k:.0%}",
                   help="Of ALL expected snippets — stricter than hit-rate when a "
                        "question needs several facts.")
    cols[2].metric("MRR", f"{report.mrr:.3f}",
                   help="Mean of 1/rank of the first hit. Rewards ranking it first.")
    cols[3].metric("rerank lift", f"{report.rerank_lift:+.0%}",
                   help=f"hit-rate@{report.n} after reranking, minus the retriever's "
                        f"own top-{report.n}. What the cross-encoder bought.")

    st.markdown("#### The gate")
    cols = st.columns(3)
    cols[0].metric("refusal accuracy", f"{report.refusal_accuracy:.0%}",
                   help="Of the questions the corpus cannot answer, how many were refused.")
    cols[1].metric("false refusals", len(report.false_refusals),
                   help="Answerable questions that were refused. Read WITH refusal "
                        "accuracy — a gate that refuses everything scores 100% and is useless.")
    cols[2].metric("answer accuracy",
                   f"{report.answer_accuracy:.0%}" if report.generated else "n/a")

    if report.false_refusals:
        st.warning(
            "**Answerable questions the gate refused** — lower `score_threshold` or "
            "improve retrieval:\n\n"
            + "\n".join(f"- {r.gold.question}  (best score {r.best_score:.4f})"
                        for r in report.false_refusals)
        )

    missed = [r for r in report.unanswerable if not r.refused]
    if missed:
        st.error(
            "**Questions the corpus cannot answer, that were NOT refused** — the gate "
            "let these through to the model:\n\n"
            + "\n".join(f"- {r.gold.question}  (best score {r.best_score:.4f})"
                        for r in missed)
        )

    st.markdown("#### Where failures happen")
    st.caption(
        "`retrieval` = the text never reached the model, so no LLM could have answered. "
        "`generation` = it was in context and the answer still missed. Different fixes."
    )
    counts = {b: len(failures.of(b)) for b in FAILURE_BUCKETS}
    cols = st.columns(4)
    for col, bucket in zip(cols, FAILURE_BUCKETS):
        col.metric(bucket, counts[bucket])
    for bucket in ("retrieval", "generation", "unconfirmed"):
        rows = failures.of(bucket)
        if not rows:
            continue
        with st.expander(f"{bucket} ({len(rows)})", expanded=bucket != "unconfirmed"):
            for row in rows:
                st.markdown(f"**{row.gold.question}**")
                st.caption(row.evidence)

    with st.expander("Machine-readable summary (diff two runs)", expanded=False):
        st.code(report_to_json(report), language="json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Ask my documents", page_icon="📄", layout="wide")
    st.title("Ask my documents")
    st.caption(
        "Grounded answers over your own documents and PDFs — with every stage of the "
        "pipeline visible, not just the answer."
    )

    try:
        base = load_config()
    except Exception as exc:
        st.error(f"Could not load config.yaml: {exc}")
        return

    names = sorted(base.chunk_presets)
    with st.sidebar:
        st.header("Settings")
        if len(names) == 1:
            # A one-item dropdown is a decision the user does not have.
            preset = names[0]
        else:
            preset = st.selectbox(
                "Chunk preset",
                names,
                index=names.index(base.default_preset),
                format_func=lambda n: f"{n} — {base.chunk_presets[n].describe()}",
            )

        indexed = read_provenance(qdrant_path_for_preset(base.store_dir, preset))
        chunking = chunking_controls(base.chunk_presets[preset], indexed, base.bi_encoder_model)

        st.subheader("Retrieval")
        mode = st.radio(
            "Mode",
            ["dense", "hybrid"],
            index=0 if base.retrieval.mode == "dense" else 1,
            help=(
                "hybrid adds BM25 keyword search fused by RRF. Measured on this corpus it "
                "did not help and slightly hurt the flat-chunked presets — kept switchable "
                "so you can re-measure on your own corpus."
            ),
        )
        retrieve_k = st.slider("retrieve_k (K)", 1, 50, base.retrieve_k)
        rerank_n = st.slider("rerank_n (N)", 1, max(1, retrieve_k), min(base.rerank_n, retrieve_k))
        gate_help = (
            "Below this cross-encoder score the LLM is never called. Tuned by "
            "`eval --sweep`, not by feel — but move it here to watch the gate flip."
        )
        if base.rerank_score_scale == "sigmoid":
            threshold = st.slider(
                "score_threshold", 0.0, 1.0, float(base.score_threshold),
                step=0.01, help=gate_help,
            )
        else:
            # On the `raw` scale the gate compares unbounded logits (~-11..+11),
            # so a 0-1 slider could not express a valid threshold at all.
            threshold = st.number_input(
                "score_threshold (raw logits)",
                value=float(base.score_threshold), step=0.5, help=gate_help,
            )
        st.caption(f"Scale: `{base.rerank_score_scale}`")

        st.subheader("Query transform")
        query_mode = st.radio(
            "Before retrieval",
            ["off", "rewrite", "hyde"],
            index=["off", "rewrite", "hyde"].index(base.retrieval.query_mode),
            help=(
                "off — embed the question as typed. "
                "rewrite — restate it in document vocabulary first. "
                "hyde — invent a plausible answer and search with that. "
                "Both non-off modes cost an LLM call per question, BEFORE retrieval, "
                "and a bad transform retrieves confidently wrong chunks."
            ),
        )
        if query_mode != "off" and not base.llm_api_key:
            st.warning("No API key — the transform will fail and fall back to the question.")

        st.subheader("Diversity (MMR)")
        use_mmr = st.checkbox(
            "Re-select for coverage",
            value=base.retrieval.mmr,
            help=(
                "Maximal Marginal Relevance. Penalises a candidate for resembling one "
                "already picked, so three near-identical chunks cannot fill every slot. "
                "The top pick is always the most relevant, so the score gate is unaffected."
            ),
        )
        mmr_lambda = base.retrieval.mmr_lambda
        if use_mmr:
            mmr_lambda = st.slider(
                "lambda (1.0 = relevance only, 0.0 = novelty only)",
                0.0, 1.0, float(base.retrieval.mmr_lambda), step=0.05,
            )

        st.subheader("Generation")
        do_generate = st.checkbox(
            "Call the LLM",
            value=bool(base.llm_api_key),
            help="Off = retrieval and gating only, no API cost. The gate still runs.",
        )
        if do_generate and not base.llm_api_key:
            st.warning("No OPENROUTER_API_KEY in .env — generation will fail.")

        st.subheader("Metadata filter")
        filter_text = st.text_input(
            "field=value, comma separated",
            placeholder="customer_tier=free, product=API",
            help=(
                "Filtering supplies information the question itself lacks. "
                "'What is my rate limit?' is genuinely ambiguous across plans and no "
                "reranker can fix that — a filter can. Try source_type=pdf."
            ),
        )

    pairs = [p.strip() for p in filter_text.split(",") if p.strip()]
    try:
        flt = MetaFilter.parse(pairs)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        flt = MetaFilter()

    if rerank_n > retrieve_k:  # config.load_config enforces this; keep the UI honest too
        st.sidebar.error("rerank_n cannot exceed retrieve_k.")
        rerank_n = retrieve_k

    cfg = replace(
        base,
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        score_threshold=threshold,
        retrieval=replace(
            base.retrieval, mode=mode, query_mode=query_mode,
            mmr=use_mmr, mmr_lambda=mmr_lambda,
        ),
        # Only the selected preset is overridden; any others stay as configured.
        chunk_presets={**base.chunk_presets, preset: chunking},
    )

    with st.sidebar:
        render_index_drift(chunking, indexed)

    if (retrieve_k, rerank_n, threshold, mode, chunking, query_mode, use_mmr) != (
        base.retrieve_k,
        base.rerank_n,
        float(base.score_threshold),
        base.retrieval.mode,
        base.chunk_presets[preset],
        base.retrieval.query_mode,
        base.retrieval.mmr,
    ):
        st.sidebar.info(
            "Settings differ from config.yaml — this session only, nothing is written. "
            "Edit config.yaml to make them the default."
        )

    ask_tab, corpus_tab, eval_tab, ingest_tab = st.tabs(
        ["Ask", "Corpus & chunking", "Evaluate", "Add documents"]
    )
    with ask_tab:
        tab_ask(cfg, preset, flt, do_generate)
    with corpus_tab:
        tab_corpus(cfg, preset)
    with eval_tab:
        tab_evaluate(cfg, preset)
    with ingest_tab:
        tab_ingest(cfg, preset)


if __name__ == "__main__":
    main()
