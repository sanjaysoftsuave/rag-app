"""Retrieval + gate evaluation.

Runs without the LLM by default, so it costs nothing and can be run on every
change. It answers four questions:

  1. hit@k   — did the correct ticket make it into the top-K at all?
               If not, no amount of reranking or prompting can save the answer.
  2. MRR     — how high did it rank? Separates "barely retrieved" from "top hit".
  3. rerank lift — how much did the cross-encoder improve position over the
               bi-encoder alone? This is the number that justifies its cost.
  4. gate accuracy — does it refuse the things it should refuse, and answer the
               things it should answer? A gate that refuses everything scores
               100% on refusals and is useless.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from rag_app.bm25 import BM25Index
from rag_app.config import AppConfig, load_config
from rag_app.embed import Embedder
from rag_app.filters import MetaFilter
from rag_app.hybrid import hybrid_retrieve
from rag_app.pipeline import ask
from rag_app.rerank import CrossEncoderReranker, rerank
from rag_app.retrieve import retrieve
from rag_app.store import SearchBackend


@dataclass(frozen=True)
class GoldQuestion:
    question: str
    ticket_id: str | None  # None = deliberately unanswerable
    must_contain: str = ""
    note: str = ""

    @property
    def answerable(self) -> bool:
        return self.ticket_id is not None


# The three rate-limit questions are the point of this set: near-identical in
# embedding space, different correct answers. A bi-encoder alone confuses them.
GOLD: list[GoldQuestion] = [
    GoldQuestion("What is the API rate limit on the Free plan?", "TIC-1001", "60"),
    GoldQuestion("How many requests per minute does the Pro plan allow?", "TIC-1002", "600"),
    GoldQuestion("What rate limit do Enterprise accounts get by default?", "TIC-1003", "6000"),
    GoldQuestion("How long does a refund to a credit card take?", "TIC-1004", "5"),
    GoldQuestion("How long does a bank transfer refund take?", "TIC-1005", "15"),
    GoldQuestion("Can a VAT number be added to an invoice after it was issued?", "TIC-1006", "credit note"),
    GoldQuestion("How long is a password reset link valid for?", "TIC-1014", "60 minutes"),
    GoldQuestion("How many MFA backup codes are issued?", "TIC-1015", "10"),
    GoldQuestion("How many times does a failed webhook get retried?", "TIC-1010", "5"),
    GoldQuestion("Is offset pagination supported?", "TIC-1012", "not supported"),
    GoldQuestion("How many days of data does mobile offline mode cache?", "TIC-1023", "7 days"),
    GoldQuestion("When is an account suspended after a failed payment?", "TIC-1032", "14"),
    GoldQuestion("How long are audit logs retained on Enterprise?", "TIC-1031", "2 years"),
    GoldQuestion("Is there an on-premise or self-hosted version?", "TIC-1036", "cloud-only"),
    GoldQuestion("What is the gateway timeout for synchronous requests?", "TIC-1027", "30 second"),
    GoldQuestion("What permission is needed to install the Slack app?", "TIC-1019", "admin"),
    # Terse, keyword-style queries — what people actually type into a help-centre
    # search box. Unambiguous, but the cross-encoder scores them far lower than
    # a well-formed question, so these are where the gate threshold actually
    # bites. Without them the sweep is flat and tells you nothing.
    GoldQuestion("password reset expired", "TIC-1014", "60 minutes", note="terse"),
    GoldQuestion("webhook retries", "TIC-1010", "5", note="terse"),
    GoldQuestion("offline cache days", "TIC-1023", "7 days", note="terse"),
    GoldQuestion("slack install failed", "TIC-1019", "admin", note="terse"),
    GoldQuestion("429 free plan", "TIC-1001", "60", note="terse"),
    GoldQuestion("duplicate charge", "TIC-1007", "retr", note="terse"),
    # Deliberately absent from the corpus — the gate must refuse these.
    GoldQuestion("Are you HIPAA compliant and will you sign a BAA?", None,
                 note="compliance topic never discussed in any ticket"),
    GoldQuestion("What are your phone support opening hours?", None,
                 note="support hours never stated"),
    GoldQuestion("How much does the Pro plan cost per month?", None,
                 note="no ticket quotes a price"),
    GoldQuestion("What is your uptime SLA percentage?", None,
                 note="SLA never mentioned"),
]


@dataclass
class QuestionResult:
    gold: GoldQuestion
    retrieved_rank: int | None  # 1-based position after bi-encoder, None = miss
    reranked_rank: int | None  # 1-based position after cross-encoder
    best_score: float
    gated_out: bool
    top_source: str = ""

    @property
    def hit_retrieval(self) -> bool:
        return self.retrieved_rank is not None

    @property
    def hit_rerank(self) -> bool:
        return self.reranked_rank is not None

    @property
    def gate_correct(self) -> bool:
        # Answerable questions must pass the gate AND surface the right ticket
        # at rank 1. Passing the gate with the wrong ticket is a worse failure
        # than refusing, because it produces a confident wrong answer.
        if self.gold.answerable:
            return not self.gated_out and self.reranked_rank == 1
        return self.gated_out


@dataclass
class EvalReport:
    preset: str
    strategy: str
    backend: str
    embedding_model: str
    results: list[QuestionResult] = field(default_factory=list)

    def _answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.gold.answerable]

    def _unanswerable(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.gold.answerable]

    @property
    def hit_at_k(self) -> float:
        rows = self._answerable()
        return sum(r.hit_retrieval for r in rows) / len(rows) if rows else 0.0

    @property
    def mrr_retrieval(self) -> float:
        rows = self._answerable()
        if not rows:
            return 0.0
        return sum(1.0 / r.retrieved_rank if r.retrieved_rank else 0.0 for r in rows) / len(rows)

    @property
    def mrr_rerank(self) -> float:
        rows = self._answerable()
        if not rows:
            return 0.0
        return sum(1.0 / r.reranked_rank if r.reranked_rank else 0.0 for r in rows) / len(rows)

    @property
    def top1_retrieval(self) -> float:
        rows = self._answerable()
        return sum(r.retrieved_rank == 1 for r in rows) / len(rows) if rows else 0.0

    @property
    def top1_rerank(self) -> float:
        rows = self._answerable()
        return sum(r.reranked_rank == 1 for r in rows) / len(rows) if rows else 0.0

    @property
    def refusal_accuracy(self) -> float:
        rows = self._unanswerable()
        return sum(r.gated_out for r in rows) / len(rows) if rows else 0.0

    @property
    def false_refusals(self) -> int:
        """Right ticket at rank 1, and the gate threw it away anyway.

        Deliberately NOT "every answerable question that got gated out". When
        the top hit is the *wrong* ticket, refusing is correct behaviour — that
        is the gate doing its job, and counting it as a failure would push you
        to lower the threshold in exactly the wrong direction. Counted
        separately as `saved_by_gate`.
        """
        return sum(1 for r in self._answerable() if r.gated_out and r.reranked_rank == 1)

    @property
    def saved_by_gate(self) -> int:
        """Wrong top-1, correctly refused instead of answering confidently wrong."""
        return sum(1 for r in self._answerable() if r.gated_out and r.reranked_rank != 1)

    def describe(self) -> str:
        lines = [
            f"Preset {self.preset} (strategy={self.strategy}, backend={self.backend})",
            f"  embedding model : {self.embedding_model}",
            f"  hit@K           : {self.hit_at_k:.0%}  (gold ticket anywhere in top-K)",
            f"  top-1 bi-encoder: {self.top1_retrieval:.0%}",
            f"  top-1 reranked  : {self.top1_rerank:.0%}   <- rerank lift: "
            f"{self.top1_rerank - self.top1_retrieval:+.0%}",
            f"  MRR bi-encoder  : {self.mrr_retrieval:.3f}",
            f"  MRR reranked    : {self.mrr_rerank:.3f}",
            f"  refusal accuracy: {self.refusal_accuracy:.0%}  "
            f"({len(self._unanswerable())} unanswerable questions)",
            f"  false refusals  : {self.false_refusals} "
            f"(right ticket at rank 1, gated out anyway)",
            f"  saved by gate   : {self.saved_by_gate} "
            f"(wrong top-1, correctly refused instead of answering wrongly)",
        ]
        misses = [r for r in self._answerable() if r.reranked_rank != 1]
        if misses:
            lines.append("  misses:")
            for r in misses:
                where = f"rank {r.reranked_rank}" if r.reranked_rank else "not retrieved"
                lines.append(
                    f"    - {r.gold.ticket_id} {where} (got [{r.top_source}]) "
                    f"| {r.gold.question}"
                )
        return "\n".join(lines)


def _rank_of(items, ticket_id: str) -> int | None:
    for i, item in enumerate(items, start=1):
        if item.chunk.source == ticket_id:
            return i
    return None


def evaluate(
    store: SearchBackend,
    cfg: AppConfig,
    preset: str,
    *,
    embedder: Embedder,
    reranker: CrossEncoderReranker,
    gold: list[GoldQuestion] | None = None,
    flt: MetaFilter | None = None,
) -> EvalReport:
    questions = gold or GOLD
    strategy = cfg.chunk_presets[preset].strategy
    report = EvalReport(
        preset=preset,
        strategy=strategy,
        backend=cfg.backend,
        embedding_model=cfg.bi_encoder_model,
    )

    for item in questions:
        qvec = embedder.encode_queries([item.question])[0]
        retrieved = retrieve(store, qvec, k=cfg.retrieve_k, flt=flt)
        reranked = rerank(
            item.question,
            retrieved,
            n=cfg.rerank_n,
            scorer=reranker,
            scale=cfg.rerank_score_scale,
        )
        best = reranked[0].score if reranked else float("-inf")
        report.results.append(
            QuestionResult(
                gold=item,
                retrieved_rank=_rank_of(retrieved, item.ticket_id) if item.ticket_id else None,
                reranked_rank=_rank_of(reranked, item.ticket_id) if item.ticket_id else None,
                best_score=best,
                gated_out=(not reranked) or best < cfg.score_threshold,
                top_source=reranked[0].chunk.source if reranked else "",
            )
        )
    return report


def sweep_threshold(
    report: EvalReport, thresholds: list[float] | None = None
) -> list[tuple[float, int, int, float]]:
    """Replay the gate at different thresholds without re-running retrieval.

    The gate has exactly one tuning knob and two failure modes that pull in
    opposite directions:

      threshold too HIGH -> false refusals: the answer was retrieved correctly
                            and thrown away anyway.
      threshold too LOW  -> the "I don't know" guarantee weakens and the model
                            gets handed weak context it will happily write from.

    Scores are already recorded per question, so this is pure arithmetic — which
    means a threshold can be chosen from evidence instead of taste.

    Returns (threshold, false_refusals, correct_refusals, f1)
    """
    grid = thresholds or [i / 20 for i in range(1, 20)]
    answerable = [r for r in report.results if r.gold.answerable]
    unanswerable = [r for r in report.results if not r.gold.answerable]

    rows = []
    for t in grid:
        # An answerable question is served only if it passes the gate AND the
        # right ticket is at rank 1; passing with the wrong ticket is worse
        # than refusing, because it yields a confident wrong answer.
        served = sum(1 for r in answerable if r.best_score >= t and r.reranked_rank == 1)
        false_refusals = sum(
            1 for r in answerable if r.best_score < t and r.reranked_rank == 1
        )
        correct_refusals = sum(1 for r in unanswerable if r.best_score < t)
        leaked = len(unanswerable) - correct_refusals

        precision = served / (served + leaked) if (served + leaked) else 0.0
        recall = served / len(answerable) if answerable else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append((t, false_refusals, correct_refusals, f1))
    return rows


def format_sweep(report: EvalReport) -> str:
    rows = sweep_threshold(report)
    n_unanswerable = len([r for r in report.results if not r.gold.answerable])
    lines = [
        f"Gate threshold sweep — preset {report.preset} ({report.strategy})",
        f"{'thresh':<9}{'false refusals':<17}{'correct refusals':<19}{'F1'}",
        "-" * 52,
    ]
    best = max(rows, key=lambda r: r[3])
    for t, false_ref, correct_ref, f1 in rows:
        marker = "  <-- best F1" if (t, false_ref, correct_ref, f1) == best else ""
        lines.append(
            f"{t:<9.2f}{false_ref:<17}{correct_ref}/{n_unanswerable:<16}{f1:.3f}{marker}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failure separation — "wrong document fetched" vs "right document, wrong
# answer". These are different bugs with different fixes: no amount of LLM
# quality fixes a retrieval failure, and no amount of retrieval tuning fixes a
# model that had the right excerpt and still answered badly.
# ---------------------------------------------------------------------------

FAILURE_BUCKETS = ("retrieval", "generation", "pass", "unconfirmed")


def _skip_generation(question, contexts, cfg) -> str:
    """Stand-in generate_fn for retrieval-only labeling — never touches the
    network, but still lets the real score gate run, so `used_llm`/`gate`
    stay meaningful even without calling the LLM."""
    return "(generation skipped — labeling retrieval only; rerun with --generate to check the answer)"


@dataclass
class FailureLabel:
    gold: GoldQuestion
    retrieved_rank: int | None
    reranked_rank: int | None
    bucket: str  # one of FAILURE_BUCKETS
    evidence: str
    answer_text: str = ""
    gate: str = ""

    def describe(self) -> str:
        lines = [f"[{self.bucket.upper()}] {self.gold.question!r}  (expects {self.gold.ticket_id})"]
        lines.append(f"    {self.evidence}")
        if self.answer_text:
            preview = self.answer_text.replace("\n", " ")[:160]
            lines.append(f"    answer: {preview}")
        return "\n".join(lines)


@dataclass
class FailureReport:
    preset: str
    generated: bool  # whether --generate actually called the LLM
    labels: list[FailureLabel] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(label.bucket for label in self.labels))

    def describe(self, *, show_pass: bool = False) -> str:
        counts = self.counts()
        mode = "retrieval + generation checked" if self.generated else "retrieval only — pass --generate to confirm generation failures"
        lines = [
            f"Failure labels — preset {self.preset} ({mode})",
            f"  retrieval failures : {counts.get('retrieval', 0):>3}  (wrong document fetched — LLM never had a chance)",
        ]
        if self.generated:
            lines.append(f"  generation failures : {counts.get('generation', 0):>3}  (right document, wrong answer)")
            lines.append(f"  pass                : {counts.get('pass', 0):>3}")
        else:
            lines.append(
                f"  unconfirmed         : {counts.get('unconfirmed', 0):>3}  "
                f"(correct doc reached the context; whether the answer used it is unchecked)"
            )
        lines.append("")
        for label in self.labels:
            if show_pass or label.bucket != "pass":
                lines.append(label.describe())
                lines.append("")
        return "\n".join(lines).rstrip()


def label_failures(
    store: SearchBackend,
    cfg: AppConfig,
    preset: str,
    *,
    embedder: Embedder,
    reranker: CrossEncoderReranker,
    gold: list[GoldQuestion] | None = None,
    generate_fn=None,
    use_llm: bool = False,
    bm25: BM25Index | None = None,
    flt: MetaFilter | None = None,
) -> FailureReport:
    """Run every answerable gold question through the real pipeline and sort
    the result into a bucket, with the evidence that justifies it.

    Reuses `ask()` end to end rather than re-deriving gate logic, so a
    labeled failure reflects exactly what the live app would have done —
    including the score gate and refusal detection, both of which can turn a
    "document was there" case into a wrong answer just as easily as the LLM
    misreading the excerpt can.

    `use_llm=False` (default) costs nothing: it can only prove bucket
    "retrieval" (the document never reached the context) or mark a question
    "unconfirmed" (it reached the context, but nobody checked what happened
    next). `use_llm=True` spends one real LLM call per unconfirmed question to
    resolve it into "pass" or "generation" using `GoldQuestion.must_contain`.
    """
    questions = [g for g in (gold or GOLD) if g.answerable]
    report = FailureReport(preset=preset, generated=use_llm)
    gen = generate_fn if use_llm else _skip_generation

    for item in questions:
        answer = ask(
            item.question,
            preset=preset,
            config=cfg,
            embedder=embedder,
            reranker=reranker,
            store=store,
            generate_fn=gen,
            flt=flt,
            bm25=bm25,
        )
        retrieved_rank = _rank_of(answer.retrieved, item.ticket_id)
        reranked_rank = _rank_of(answer.reranked, item.ticket_id)

        if reranked_rank is None:
            if retrieved_rank is None:
                evidence = "never entered the retrieved candidate set at all — the retriever's fault"
            else:
                evidence = (
                    f"retrieved at rank {retrieved_rank}, but reranked OUT of the top "
                    f"{cfg.rerank_n} shown to the LLM — the cross-encoder's fault, not the retriever's"
                )
            report.labels.append(FailureLabel(
                gold=item, retrieved_rank=retrieved_rank, reranked_rank=None,
                bucket="retrieval", evidence=evidence, gate=answer.gate,
            ))
            continue

        if not use_llm:
            report.labels.append(FailureLabel(
                gold=item, retrieved_rank=retrieved_rank, reranked_rank=reranked_rank,
                bucket="unconfirmed",
                evidence=(
                    f"correct ticket WAS in context at rank {reranked_rank}/{cfg.rerank_n} — "
                    f"rerun with --generate to check whether the answer actually used it"
                ),
                gate=answer.gate,
            ))
            continue

        if not answer.used_llm:
            report.labels.append(FailureLabel(
                gold=item, retrieved_rank=retrieved_rank, reranked_rank=reranked_rank,
                bucket="generation",
                evidence=(
                    f"correct ticket WAS in context at rank {reranked_rank}, but the score gate "
                    f"refused anyway (best_score={answer.best_score:.3f} < threshold {cfg.score_threshold}) "
                    f"— retrieval did its job, the gate is what threw the answer away"
                ),
                answer_text=answer.text, gate=answer.gate,
            ))
            continue

        if answer.refused:
            report.labels.append(FailureLabel(
                gold=item, retrieved_rank=retrieved_rank, reranked_rank=reranked_rank,
                bucket="generation",
                evidence=(
                    f"correct ticket was in context at rank {reranked_rank}, but the model "
                    f"refused to answer despite it being right there"
                ),
                answer_text=answer.text, gate=answer.gate,
            ))
            continue

        if item.must_contain and item.must_contain.lower() not in answer.text.lower():
            report.labels.append(FailureLabel(
                gold=item, retrieved_rank=retrieved_rank, reranked_rank=reranked_rank,
                bucket="generation",
                evidence=(
                    f"correct ticket was in context at rank {reranked_rank}, but the answer does "
                    f"not mention the expected content {item.must_contain!r} — the model had it "
                    f"and still got it wrong"
                ),
                answer_text=answer.text, gate=answer.gate,
            ))
            continue

        report.labels.append(FailureLabel(
            gold=item, retrieved_rank=retrieved_rank, reranked_rank=reranked_rank,
            bucket="pass",
            evidence=f"correct ticket at rank {reranked_rank}, answer used it correctly",
            answer_text=answer.text, gate=answer.gate,
        ))

    return report


# ---------------------------------------------------------------------------
# Dense vs hybrid — the ONE-change, before/after retrieval comparison.
# ---------------------------------------------------------------------------


def _hit_at_k(results: list, ticket_id: str, k: int) -> tuple[bool, int | None]:
    for i, item in enumerate(results[:k], start=1):
        if item.chunk.source == ticket_id:
            return True, i
    return False, None


@dataclass
class RetrievalCompareRow:
    gold: GoldQuestion
    dense_hit: bool
    hybrid_hit: bool
    dense_rank: int | None
    hybrid_rank: int | None

    @property
    def outcome(self) -> str:
        if self.dense_hit and self.hybrid_hit:
            return "always-hit"
        if not self.dense_hit and self.hybrid_hit:
            return "fixed"
        if self.dense_hit and not self.hybrid_hit:
            return "regressed"
        return "still-broken"


@dataclass
class RetrievalCompareReport:
    preset: str
    k: int
    rows: list[RetrievalCompareRow] = field(default_factory=list)

    def rows_by_outcome(self, outcome: str) -> list[RetrievalCompareRow]:
        return [r for r in self.rows if r.outcome == outcome]

    @property
    def dense_hit_rate(self) -> float:
        return sum(r.dense_hit for r in self.rows) / len(self.rows) if self.rows else 0.0

    @property
    def hybrid_hit_rate(self) -> float:
        return sum(r.hybrid_hit for r in self.rows) / len(self.rows) if self.rows else 0.0

    def describe(self) -> str:
        fixed = self.rows_by_outcome("fixed")
        regressed = self.rows_by_outcome("regressed")
        still_broken = self.rows_by_outcome("still-broken")
        n = len(self.rows)
        lines = [
            f"Retrieval mode comparison — preset {self.preset}, hit-rate@{self.k} "
            f"(dense-only vs dense+BM25 fused by RRF; same embedder, same store, same k — "
            f"the ONE thing that changed is the retrieval strategy)",
            f"  dense  hit-rate@{self.k} : {self.dense_hit_rate:.0%}  "
            f"({sum(r.dense_hit for r in self.rows)}/{n})",
            f"  hybrid hit-rate@{self.k} : {self.hybrid_hit_rate:.0%}  "
            f"({sum(r.hybrid_hit for r in self.rows)}/{n})   "
            f"<- {self.hybrid_hit_rate - self.dense_hit_rate:+.0%}",
            f"  fixed by hybrid     : {len(fixed)}",
            f"  regressed by hybrid : {len(regressed)}  (hybrid made these WORSE — report this, don't hide it)",
            f"  still broken        : {len(still_broken)}  (this change did NOT fix these)",
        ]
        if fixed:
            lines.append("\n  fixed:")
            for r in fixed:
                lines.append(
                    f"    - {r.gold.ticket_id}  dense=miss -> hybrid=rank {r.hybrid_rank}  | {r.gold.question}"
                )
        if regressed:
            lines.append("\n  regressed:")
            for r in regressed:
                lines.append(
                    f"    - {r.gold.ticket_id}  dense=rank {r.dense_rank} -> hybrid=miss  | {r.gold.question}"
                )
        if still_broken:
            lines.append("\n  NOT fixed by this change:")
            for r in still_broken:
                lines.append(f"    - {r.gold.ticket_id}  both miss  | {r.gold.question}")
        return "\n".join(lines)


def compare_retrieval_modes(
    store: SearchBackend,
    cfg: AppConfig,
    preset: str,
    *,
    embedder: Embedder,
    bm25: BM25Index | None = None,
    gold: list[GoldQuestion] | None = None,
    k: int = 3,
    flt: MetaFilter | None = None,
) -> RetrievalCompareReport:
    """Measure hit-rate@k for dense-only vs hybrid retrieval on the SAME
    questions, SAME embedder, SAME store, SAME pool size.

    Deliberately does not involve the cross-encoder: the reranker is
    unchanged in both conditions, so running results through it would blend
    its effect into what should be a clean measurement of the retrieval
    strategy alone — "did the right document even reach the candidate pool",
    not "where did it end up after reranking".
    """
    questions = [g for g in (gold or GOLD) if g.answerable]
    index = bm25 or BM25Index.from_store(store)
    report = RetrievalCompareReport(preset=preset, k=k)

    for item in questions:
        qvec = embedder.encode_queries([item.question])[0]
        dense = retrieve(store, qvec, k=cfg.retrieve_k, flt=flt)
        # dense_pool is pinned to the SAME cfg.retrieve_k used by the dense-only
        # arm above. hybrid_retrieve's own default would silently widen it to
        # max(k*2, 10), which would let hybrid see more dense candidates than
        # dense-only does — a second uncontrolled variable hiding inside what
        # should be a one-variable comparison.
        hybrid = hybrid_retrieve(
            store, index, qvec, item.question, k=cfg.retrieve_k, flt=flt,
            dense_pool=cfg.retrieve_k,
            bm25_pool=cfg.retrieval.bm25_pool, k_rrf=cfg.retrieval.rrf_k,
        )
        dense_hit, dense_rank = _hit_at_k(dense, item.ticket_id, k)
        hybrid_hit, hybrid_rank = _hit_at_k(hybrid, item.ticket_id, k)
        report.rows.append(RetrievalCompareRow(
            gold=item, dense_hit=dense_hit, hybrid_hit=hybrid_hit,
            dense_rank=dense_rank, hybrid_rank=hybrid_rank,
        ))
    return report


def boundary_bleed(chunks) -> tuple[int, int]:
    """(chunks spanning >1 ticket, total chunks) — the flat-strategy defect."""
    total = len(chunks)
    bleeding = sum(1 for c in chunks if c.metadata.get("bleed"))
    return bleeding, total


def load_default_config() -> AppConfig:
    return load_config()
