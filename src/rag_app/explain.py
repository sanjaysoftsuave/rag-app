"""Turn an `Answer` into the story of how it was produced.

`format_answer()` in pipeline.py prints enough to debug a query in a terminal.
This module answers the harder questions a reader actually has — which
candidate the reranker *demoted* out of context, how far each one moved, what
the raw logit was behind a sigmoid score, which of the four gate paths fired
and what would have to change for it to fire differently.

Deliberately free of any UI dependency. `ui.py` imports streamlit; this
module imports nothing heavier than the app's own dataclasses, so the
interesting logic stays testable in the offline suite instead of only being
exercisable by clicking. Nothing here re-derives pipeline behaviour: every
number comes from what `ask()` actually did — a debugging view that
reimplements the thing it debugs will eventually disagree with it, and the
disagreement is invisible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag_app.config import AppConfig
from rag_app.embed import spec_for
from rag_app.generate import Answer, build_prompt
from rag_app.store import ScoredChunk


def logit(p: float) -> float | None:
    """Undo `rerank.sigmoid`, recovering the cross-encoder's raw output.

    The gate compares a calibrated 0-1 probability, which is the right scale
    to threshold on but hides how confident the model actually was: 0.9999 and
    0.99999 look nearly identical and are 2.3 logits apart. Showing both means
    never having to choose between an interpretable number and an informative
    one. Returns None at the saturated ends, where the inverse is undefined
    and any finite answer would be invented precision.
    """
    if not 0.0 < p < 1.0:
        return None
    return math.log(p / (1.0 - p))


def source_kind(chunk) -> str:
    """`doc` or `pdf` — how this chunk entered the corpus."""
    return str((chunk.metadata or {}).get("source_type") or "doc")


def source_detail(chunk) -> str:
    """A human pointer back to the original."""
    return "PDF" if source_kind(chunk) == "pdf" else "document"


def retrieval_score_label(answer: Answer) -> str:
    """What the first-stage score actually IS, which changes with the mode.

    In hybrid mode `retrieved` carries an RRF score — a sum of reciprocal rank
    positions, roughly 0.015-0.033, on no relation to cosine's scale. Labelling
    it "cosine" would invite exactly the cross-scale comparison hybrid.py
    exists to avoid.
    """
    return "rrf" if answer.meta.get("retrieval_mode") == "hybrid" else "cosine"


@dataclass(frozen=True)
class CandidateRow:
    """One chunk's journey through both ranking stages."""

    chunk_id: str
    source: str
    kind: str
    detail: str
    text: str
    dense_rank: int | None
    dense_score: float | None
    rerank_rank: int | None
    rerank_score: float | None
    rerank_logit: float | None
    in_context: bool

    @property
    def rank_delta(self) -> int | None:
        """Positions gained at rerank. Positive = promoted, negative = demoted."""
        if self.dense_rank is None or self.rerank_rank is None:
            return None
        return self.dense_rank - self.rerank_rank

    @property
    def movement(self) -> str:
        delta = self.rank_delta
        if delta is None:
            return "—"
        if delta > 0:
            return f"▲ {delta}"
        if delta < 0:
            return f"▼ {abs(delta)}"
        return "="


def build_candidate_table(answer: Answer) -> list[CandidateRow]:
    """Join the retrieval ranking and the full rerank ranking, one row per chunk.

    Ordered by the reranker's opinion, because that is the ordering that
    decided what reached the LLM. Candidates the reranker scored but cut are
    included with `in_context=False` — they are the whole point of the table.
    """
    dense_rank = {
        item.chunk.chunk_id: (i, item.score)
        for i, item in enumerate(answer.retrieved, start=1)
    }
    # Fall back to `reranked` when `reranked_all` is absent, so an Answer
    # rebuilt from an older trace still renders rather than showing nothing.
    full = answer.reranked_all or answer.reranked
    context_ids = {item.chunk.chunk_id for item in answer.reranked}

    rows: list[CandidateRow] = []
    for rank, item in enumerate(full, start=1):
        cid = item.chunk.chunk_id
        d_rank, d_score = dense_rank.get(cid, (None, None))
        rows.append(
            CandidateRow(
                chunk_id=cid,
                source=item.chunk.source,
                kind=source_kind(item.chunk),
                detail=source_detail(item.chunk),
                text=item.chunk.text,
                dense_rank=d_rank,
                dense_score=d_score,
                rerank_rank=rank,
                rerank_score=item.score,
                rerank_logit=logit(item.score),
                in_context=cid in context_ids,
            )
        )

    # A chunk retrieved but never scored by the reranker should still be
    # visible rather than silently vanishing from the accounting.
    scored = {item.chunk.chunk_id for item in full}
    for cid, (d_rank, d_score) in dense_rank.items():
        if cid in scored:
            continue
        match = next(i for i in answer.retrieved if i.chunk.chunk_id == cid)
        rows.append(
            CandidateRow(
                chunk_id=cid,
                source=match.chunk.source,
                kind=source_kind(match.chunk),
                detail=source_detail(match.chunk),
                text=match.chunk.text,
                dense_rank=d_rank,
                dense_score=d_score,
                rerank_rank=None,
                rerank_score=None,
                rerank_logit=None,
                in_context=False,
            )
        )
    return rows


@dataclass(frozen=True)
class GateExplanation:
    gate: str
    headline: str
    detail: str
    llm_called: bool
    passed: bool
    best_score: float | None
    threshold: float
    margin: float | None  # best_score - threshold; negative means refused here


GATE_HEADLINES = {
    "no-candidates": "Refused — nothing to read",
    "below-threshold": "Refused — score gate",
    "model-refused": "Refused — the model declined",
    "answered": "Answered",
}


def explain_gate(answer: Answer, cfg: AppConfig) -> GateExplanation:
    """Which of the four paths through `ask()` fired, and what it means.

    The distinction that matters and is easy to lose: `below-threshold` never
    called the LLM, so no prompt, no tokens, no model judgment was involved —
    the app refused on its own. `model-refused` DID call the LLM, which read
    the excerpts and said no. They look identical to a user and have opposite
    fixes (lower the threshold vs improve retrieval or the prompt).
    """
    threshold = cfg.score_threshold
    best = None if answer.best_score == float("-inf") else answer.best_score
    margin = None if best is None else best - threshold

    if answer.gate == "no-candidates":
        detail = (
            "Retrieval returned nothing at all, so there was no candidate to score. "
            "With a metadata filter applied this usually means the filter excluded "
            "every chunk; without one it means the store is empty."
        )
    elif answer.gate == "below-threshold":
        detail = (
            f"The best cross-encoder score was {best:.4f}, below score_threshold "
            f"{threshold}. The LLM was never called — this refusal cost nothing and "
            f"involved no model judgment. If the corpus really does answer this "
            f"question, that is a false refusal: check the top candidate below, then "
            f"re-tune with `eval --sweep` rather than nudging the threshold by feel."
        )
    elif answer.gate == "model-refused":
        detail = (
            f"The score gate passed ({best:.4f} >= {threshold}) and the LLM was called, "
            f"but it judged the excerpts insufficient and returned the refusal string. "
            f"Sources were stripped deliberately — attaching them would credit those "
            f"documents for a non-answer. Retrieval may still have been correct here; "
            f"read the excerpts below and decide whether the model was right."
        )
    else:
        detail = (
            f"The score gate passed ({best:.4f} >= {threshold}) and the LLM answered "
            f"from the {len(answer.reranked)} excerpts below."
        )

    return GateExplanation(
        gate=answer.gate,
        headline=GATE_HEADLINES.get(answer.gate, answer.gate),
        detail=detail,
        llm_called=answer.used_llm,
        passed=answer.gate == "answered",
        best_score=best,
        threshold=threshold,
        margin=margin,
    )


@dataclass(frozen=True)
class CitationAudit:
    grounded: list[str]
    invented: list[str]
    fell_back: bool  # model cited nothing; sources came from the excerpts

    @property
    def clean(self) -> bool:
        return not self.invented and not self.fell_back


def audit_citations(answer: Answer) -> CitationAudit:
    """Grounding guarantee #4, made visible.

    `fell_back` is the quiet one: the model produced no citation at all, so
    `ask()` listed the retrieved excerpts as sources. The answer then *looks*
    sourced without the model ever having claimed those sources.
    """
    grounded = [] if answer.refused else list(answer.sources)
    if answer.hallucinated_citations:
        grounded = [s for s in grounded if s not in answer.hallucinated_citations]
    fell_back = bool(
        answer.used_llm and not answer.refused and not answer.meta.get("cited")
    )
    return CitationAudit(
        grounded=grounded,
        invented=list(answer.hallucinated_citations),
        fell_back=fell_back,
    )


def query_encoding(cfg: AppConfig, question: str) -> dict[str, str]:
    """The exact string handed to the bi-encoder, prefix included.

    Asymmetric models (E5, BGE) require a query-side prefix that never appears
    in any log, and omitting it degrades retrieval with no error at all — the
    failure embed.py's registry exists to prevent. Printing the prepared string
    makes the invisible half of that contract visible.
    """
    spec = spec_for(cfg.bi_encoder_model)
    prefix = spec.query_prefix
    return {
        "model": cfg.bi_encoder_model,
        "family": "asymmetric" if spec.asymmetric else "symmetric",
        "query_prefix": prefix or "(none)",
        "passage_prefix": spec.passage_prefix or "(none)",
        "encoded": f"{prefix}{question}",
        "note": spec.note,
    }


def config_rows(cfg: AppConfig, preset: str, indexed=None) -> list[tuple[str, str]]:
    """Every setting that shaped this run, as label/value pairs.

    `indexed` is the store's `StoreMeta` when available. It matters because
    chunking is an INGEST-time setting: the chunks that were just retrieved
    were produced by whatever built the index, not by whatever the config
    currently says. Reporting the config value here would name a setting that
    had no part in this answer — so the store's own provenance wins, and any
    divergence is called out rather than papered over.
    """
    p = cfg.chunk_presets[preset]
    if indexed is not None:
        live = f"{indexed.chunk_size} chars, {indexed.overlap} overlap"
        if (indexed.chunk_size, indexed.overlap) != (p.chunk_size, p.overlap):
            live += f"  ← as indexed; settings now say {p.describe()}, needs re-ingest"
        chunking_row = ("chunking (as indexed)", live)
    else:
        chunking_row = ("chunking", f"{p.chunk_size} chars, {p.overlap} overlap")

    return [
        ("preset", f"{preset} ({p.describe()})"),
        chunking_row,
        ("bi-encoder", cfg.bi_encoder_model),
        ("cross-encoder", cfg.cross_encoder_model),
        ("retrieval mode", cfg.retrieval.mode),
        ("K → N", f"{cfg.retrieve_k} → {cfg.rerank_n}"),
        ("score scale", cfg.rerank_score_scale),
        ("score_threshold", str(cfg.score_threshold)),
        ("vector store", "qdrant server" if cfg.qdrant.url else "qdrant embedded (brute force)"),
        ("LLM", f"{cfg.llm.model} @ temp {cfg.llm.temperature}"),
    ]


def prompt_messages(question: str, contexts: list[ScoredChunk]) -> list[dict[str, str]]:
    """The literal messages `generate_answer` would send — no re-derivation.

    Calls `build_prompt` itself rather than reconstructing it, so what the UI
    displays cannot drift from what the app sends.
    """
    return build_prompt(question, contexts)


def corpus_mix(rows: list[CandidateRow]) -> dict[str, int]:
    """How many candidates came from tickets vs documents vs PDF pages."""
    mix: dict[str, int] = {}
    for row in rows:
        mix[row.kind] = mix.get(row.kind, 0) + 1
    return mix
