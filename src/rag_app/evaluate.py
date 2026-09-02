"""Measurement: does retrieval find the right text, and does the answer use it?

WHY THIS LOOKS DIFFERENT FROM THE USUAL TEXTBOOK VERSION
--------------------------------------------------------
Retrieval metrics are normally defined over *documents*: hit-rate@k asks
whether the relevant document id appeared in the top k. That works when a
corpus is thousands of separate records. It is useless here, because a corpus
of one PDF has exactly one document id — hit-rate would be 1.0 for every
question including the nonsense ones, and would measure nothing at all.

So relevance is defined by **content**, not by source id. A gold question names
a snippet that must appear in a retrieved chunk. That works whether the corpus
is one file or ten thousand, and it is checkable by a human reading the
document rather than requiring pre-labelled record ids.

THE METRICS
-----------
  hit-rate@k   fraction of questions where SOME expected snippet appeared in
               the top-k retrieved chunks. "Did we find anything useful?"
  recall@k     fraction of ALL expected snippets found across the top-k. With
               one snippet per question this equals hit-rate; with several it
               is stricter, and it is the honest number when a question needs
               two facts to be answered fully.
  MRR          mean of 1/rank of the first chunk containing an expected
               snippet. Rewards ranking the right chunk first, not merely
               somewhere in the funnel.
  rerank lift  hit-rate after reranking (top-N) minus hit-rate before it
               (top-K restricted to N). The number that justifies the
               cross-encoder's cost, or fails to.
  refusal accuracy / false refusals
               Read these together. A gate that refuses everything scores
               100% on refusals and is worthless.

RETRIEVAL vs GENERATION FAILURES
--------------------------------
`label_failures()` sorts every answerable question into exactly one bucket,
using `ask()` itself rather than re-deriving gate logic, so a label reflects
what the live app actually did:

  retrieval    the expected text never reached the final context. No LLM,
               however good, could have answered. Evidence distinguishes
               "never retrieved at all" (bi-encoder's fault) from "retrieved
               then reranked out" (cross-encoder's fault).
  generation   the text WAS in context and the answer still missed. Includes
               the gate refusing despite a passing chunk — a false refusal is
               a generation-side failure, because retrieval did its job.
  pass         answered, and the answer contained what it should.
  unconfirmed  text reached context but --generate was not passed, so nothing
               checked what the LLM did with it.

The boundary is membership in the final `rerank_n` context, not "rank 1". A
chunk at rank 3 of 3 still reached the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_app.config import AppConfig, load_config
from rag_app.pipeline import ask
from rag_app.store import ScoredChunk

GOLD_FILENAME = "gold.yaml"


# ---------------------------------------------------------------------------
# The gold set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldQuestion:
    """One question with a known-correct outcome.

    `expect_in_chunk` is what must appear in a RETRIEVED CHUNK — the retrieval
    ground truth. `must_contain` is what must appear in the ANSWER — the
    generation ground truth. They are usually the same string, so
    `expect_in_chunk` defaults to `must_contain`, but they come apart when the
    document phrases a fact differently from how an answer would state it.

    `unanswerable=True` marks a question the corpus genuinely cannot answer.
    These are not padding: without them you cannot tell a well-calibrated gate
    from one that never refuses anything.
    """

    question: str
    must_contain: list[str] = field(default_factory=list)
    expect_in_chunk: list[str] = field(default_factory=list)
    unanswerable: bool = False
    note: str = ""

    @property
    def answerable(self) -> bool:
        return not self.unanswerable

    @property
    def snippets(self) -> list[str]:
        return self.expect_in_chunk or self.must_contain

    def found_in(self, text: str) -> list[str]:
        """Which expected snippets appear in `text`, case-insensitively."""
        low = text.lower()
        return [s for s in self.snippets if s.lower() in low]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_gold(raw: Any) -> list[GoldQuestion]:
    """Build gold questions from parsed YAML/JSON, with useful errors.

    A malformed gold set should say which entry is wrong. Silently skipping a
    bad entry would quietly shrink the denominator of every metric.
    """
    if not isinstance(raw, list):
        raise ValueError("The gold set must be a list of question entries.")
    questions: list[GoldQuestion] = []
    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Gold entry {i} is not a mapping: {entry!r}")
        text = entry.get("question")
        if not text:
            raise ValueError(f"Gold entry {i} has no 'question'.")
        unanswerable = bool(entry.get("unanswerable", False))
        must = _as_list(entry.get("must_contain"))
        expect = _as_list(entry.get("expect_in_chunk"))
        if not unanswerable and not (must or expect):
            raise ValueError(
                f"Gold entry {i} ({text!r}) is answerable but names nothing to look "
                f"for. Add 'must_contain', or mark it 'unanswerable: true'."
            )
        questions.append(
            GoldQuestion(
                question=str(text),
                must_contain=must,
                expect_in_chunk=expect,
                unanswerable=unanswerable,
                note=str(entry.get("note", "")),
            )
        )
    return questions


def gold_path(cfg: AppConfig) -> Path:
    return cfg.tickets_dir.parent / GOLD_FILENAME


def load_gold(cfg: AppConfig | None = None, path: Path | None = None) -> list[GoldQuestion]:
    """Read the gold set, or explain how to create one."""
    cfg = cfg or load_config()
    target = path or gold_path(cfg)
    if not target.exists():
        raise FileNotFoundError(
            f"No gold set at {target}. Metrics need questions whose correct answers "
            f"you already know — write them for YOUR documents; there is no default. "
            f"See gold.example.yaml at the repo root for the format."
        )
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    return parse_gold(raw)


def write_gold(questions: list[GoldQuestion], path: Path) -> None:
    payload = []
    for q in questions:
        entry: dict[str, Any] = {"question": q.question}
        if q.unanswerable:
            entry["unanswerable"] = True
        if q.must_contain:
            entry["must_contain"] = q.must_contain
        if q.expect_in_chunk:
            entry["expect_in_chunk"] = q.expect_in_chunk
        if q.note:
            entry["note"] = q.note
        payload.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------


def _first_hit_rank(gold: GoldQuestion, items: list[ScoredChunk]) -> int | None:
    for rank, item in enumerate(items, start=1):
        if gold.found_in(item.chunk.text):
            return rank
    return None


def _snippets_found(gold: GoldQuestion, items: list[ScoredChunk]) -> set[str]:
    found: set[str] = set()
    for item in items:
        found.update(gold.found_in(item.chunk.text))
    return found


@dataclass
class QuestionResult:
    gold: GoldQuestion
    retrieved_rank: int | None    # 1-based rank of the first hit after retrieval
    reranked_rank: int | None     # ... after reranking, within the final context
    retrieved_found: set[str]     # which expected snippets appeared in top-k
    reranked_found: set[str]      # ... in the final top-n
    best_score: float
    used_llm: bool
    refused: bool
    gate: str
    answer_text: str = ""

    @property
    def hit(self) -> bool:
        return self.retrieved_rank is not None

    @property
    def hit_after_rerank(self) -> bool:
        return self.reranked_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.retrieved_rank if self.retrieved_rank else 0.0

    @property
    def answer_ok(self) -> bool:
        """Did the answer contain every string it was supposed to?"""
        if not self.gold.must_contain:
            return not self.refused
        low = self.answer_text.lower()
        return all(s.lower() in low for s in self.gold.must_contain)


@dataclass
class EvalReport:
    preset: str
    k: int
    n: int
    results: list[QuestionResult]
    generated: bool = False

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.gold.answerable]

    @property
    def unanswerable(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.gold.answerable]

    def _frac(self, values: list[bool]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    @property
    def hit_rate(self) -> float:
        """Did SOME expected snippet reach the top-k?"""
        return self._frac([r.hit for r in self.answerable])

    @property
    def recall_at_k(self) -> float:
        """Of ALL expected snippets, how many reached the top-k?

        Distinct from hit-rate only when a question expects several snippets —
        which is exactly the case where hit-rate flatters a partial retrieval.
        """
        total = sum(len(r.gold.snippets) for r in self.answerable)
        found = sum(len(r.retrieved_found) for r in self.answerable)
        return (found / total) if total else 0.0

    @property
    def recall_at_n(self) -> float:
        total = sum(len(r.gold.snippets) for r in self.answerable)
        found = sum(len(r.reranked_found) for r in self.answerable)
        return (found / total) if total else 0.0

    @property
    def mrr(self) -> float:
        rows = self.answerable
        return (sum(r.reciprocal_rank for r in rows) / len(rows)) if rows else 0.0

    @property
    def hit_rate_after_rerank(self) -> float:
        return self._frac([r.hit_after_rerank for r in self.answerable])

    @property
    def rerank_lift(self) -> float:
        """Hit-rate in the final N vs the retriever's own top-N.

        Compares like with like: both are 'did the right text make it into N
        slots', one chosen by the bi-encoder and one by the cross-encoder. The
        difference is what the reranker bought.
        """
        baseline = self._frac(
            [r.retrieved_rank is not None and r.retrieved_rank <= self.n
             for r in self.answerable]
        )
        return self.hit_rate_after_rerank - baseline

    @property
    def refusal_accuracy(self) -> float:
        return self._frac([r.refused for r in self.unanswerable])

    @property
    def false_refusals(self) -> list[QuestionResult]:
        """Answerable questions the app refused. Read WITH refusal accuracy."""
        return [r for r in self.answerable if r.refused]

    @property
    def answer_accuracy(self) -> float:
        return self._frac([r.answer_ok for r in self.answerable])

    def describe(self) -> str:
        lines = [
            f"Preset {self.preset}  (K={self.k} -> N={self.n}, "
            f"{len(self.answerable)} answerable + {len(self.unanswerable)} unanswerable)",
            "",
            f"  hit-rate@{self.k}      {self.hit_rate:6.1%}   some expected text reached the funnel",
            f"  recall@{self.k}        {self.recall_at_k:6.1%}   of all expected snippets",
            f"  MRR              {self.mrr:6.3f}   1/rank of the first hit",
            f"  hit-rate@{self.n}       {self.hit_rate_after_rerank:6.1%}   survived reranking into context",
            f"  recall@{self.n}         {self.recall_at_n:6.1%}",
            f"  rerank lift      {self.rerank_lift:+6.1%}   vs the retriever's own top-{self.n}",
            f"  refusal accuracy {self.refusal_accuracy:6.1%}   of {len(self.unanswerable)} unanswerable",
            f"  false refusals   {len(self.false_refusals):6d}   answerable questions refused",
        ]
        if self.generated:
            lines.append(
                f"  answer accuracy  {self.answer_accuracy:6.1%}   answers containing what they should"
            )
        else:
            lines.append("  answer accuracy     n/a   (retrieval only; pass --generate)")
        if self.false_refusals:
            lines.append("\n  Refused but answerable:")
            for r in self.false_refusals:
                lines.append(f"    - {r.gold.question}   (best score {r.best_score:.4f})")
        return "\n".join(lines)


def _skip_generation(question, contexts, cfg) -> str:
    """Stand-in generate_fn: no network, but the real gate still runs, so
    `gate` and `used_llm` stay meaningful without spending anything."""
    return "(generation skipped)"


def evaluate(
    gold: list[GoldQuestion],
    preset: str | None = None,
    config: AppConfig | None = None,
    *,
    embedder=None,
    reranker=None,
    store=None,
    use_llm: bool = False,
    generate_fn=None,
) -> EvalReport:
    """Run every gold question through the real `ask()` and score the results."""
    cfg = config or load_config()
    name = preset or cfg.default_preset
    results: list[QuestionResult] = []

    for question in gold:
        answer = ask(
            question.question,
            preset=name,
            config=cfg,
            embedder=embedder,
            reranker=reranker,
            store=store,
            generate_fn=generate_fn if use_llm else _skip_generation,
        )
        results.append(
            QuestionResult(
                gold=question,
                retrieved_rank=_first_hit_rank(question, answer.retrieved),
                reranked_rank=_first_hit_rank(question, answer.reranked),
                retrieved_found=_snippets_found(question, answer.retrieved),
                reranked_found=_snippets_found(question, answer.reranked),
                best_score=answer.best_score,
                used_llm=answer.used_llm,
                refused=answer.refused,
                gate=answer.gate,
                answer_text=answer.text if use_llm else "",
            )
        )

    return EvalReport(
        preset=name, k=cfg.retrieve_k, n=cfg.rerank_n, results=results, generated=use_llm
    )


# ---------------------------------------------------------------------------
# Failure labelling
# ---------------------------------------------------------------------------

FAILURE_BUCKETS = ("retrieval", "generation", "pass", "unconfirmed")


@dataclass
class FailureLabel:
    gold: GoldQuestion
    bucket: str
    evidence: str

    @property
    def failed(self) -> bool:
        return self.bucket in ("retrieval", "generation")


@dataclass
class FailureReport:
    preset: str
    labels: list[FailureLabel]
    generated: bool = False

    def of(self, bucket: str) -> list[FailureLabel]:
        return [x for x in self.labels if x.bucket == bucket]

    def describe(self, show_pass: bool = False) -> str:
        counts = {b: len(self.of(b)) for b in FAILURE_BUCKETS}
        lines = [
            f"Preset {self.preset}: "
            + "  ".join(f"{b}={counts[b]}" for b in FAILURE_BUCKETS),
            "",
            "  retrieval   = the text never reached the LLM's context. No model could have answered.",
            "  generation  = the text WAS in context and the answer still missed.",
        ]
        if not self.generated:
            lines.append(
                "  unconfirmed = reached context, but --generate was not passed, so "
                "nothing checked the answer."
            )
        lines.append("")
        for bucket in ("retrieval", "generation", "unconfirmed", "pass"):
            rows = self.of(bucket)
            if not rows or (bucket == "pass" and not show_pass):
                continue
            lines.append(f"  [{bucket}]")
            for row in rows:
                lines.append(f"    - {row.gold.question}")
                lines.append(f"        {row.evidence}")
            lines.append("")
        return "\n".join(lines).rstrip()


def label_failures(
    gold: list[GoldQuestion],
    preset: str | None = None,
    config: AppConfig | None = None,
    *,
    embedder=None,
    reranker=None,
    store=None,
    use_llm: bool = False,
    generate_fn=None,
) -> FailureReport:
    """Sort each answerable question into exactly one bucket.

    Calls `ask()` rather than reimplementing the gate, so the label always
    describes what the live app did — the two cannot drift apart.
    """
    cfg = config or load_config()
    name = preset or cfg.default_preset
    labels: list[FailureLabel] = []

    for question in [g for g in gold if g.answerable]:
        answer = ask(
            question.question,
            preset=name,
            config=cfg,
            embedder=embedder,
            reranker=reranker,
            store=store,
            generate_fn=generate_fn if use_llm else _skip_generation,
        )
        retrieved_rank = _first_hit_rank(question, answer.retrieved)
        context_rank = _first_hit_rank(question, answer.reranked)

        if context_rank is None:
            # Which stage lost it? Different fixes: widen retrieve_k / change
            # the embedder, versus change the reranker or rerank_n.
            if retrieved_rank is None:
                evidence = (
                    f"never retrieved — no chunk in the top {cfg.retrieve_k} contained "
                    f"the expected text (bi-encoder did not find it)"
                )
            else:
                evidence = (
                    f"retrieved at rank {retrieved_rank} but reranked out of the top "
                    f"{cfg.rerank_n} (cross-encoder demoted it)"
                )
            labels.append(FailureLabel(question, "retrieval", evidence))
            continue

        if answer.refused:
            # Retrieval did its job; the refusal is a generation-side failure.
            if not answer.used_llm:
                evidence = (
                    f"the right text was in context at rank {context_rank}, but the "
                    f"score gate refused: best score {answer.best_score:.4f} < "
                    f"threshold {cfg.score_threshold}. A false refusal."
                )
            else:
                evidence = (
                    f"the right text was in context at rank {context_rank}, the LLM was "
                    f"called, and it declined to answer."
                )
            labels.append(FailureLabel(question, "generation", evidence))
            continue

        if not use_llm:
            labels.append(
                FailureLabel(
                    question,
                    "unconfirmed",
                    f"reached context at rank {context_rank}; pass --generate to check "
                    f"what the answer did with it",
                )
            )
            continue

        low = answer.text.lower()
        missing = [s for s in question.must_contain if s.lower() not in low]
        if missing:
            labels.append(
                FailureLabel(
                    question,
                    "generation",
                    f"the right text was in context at rank {context_rank}, but the "
                    f"answer omitted: {', '.join(repr(m) for m in missing)}",
                )
            )
        else:
            labels.append(
                FailureLabel(question, "pass", f"answered from context rank {context_rank}")
            )

    return FailureReport(preset=name, labels=labels, generated=use_llm)


def report_to_json(report: EvalReport) -> str:
    """Machine-readable summary, for diffing two runs against each other."""
    return json.dumps(
        {
            "preset": report.preset,
            "k": report.k,
            "n": report.n,
            "hit_rate_at_k": round(report.hit_rate, 4),
            "recall_at_k": round(report.recall_at_k, 4),
            "mrr": round(report.mrr, 4),
            "hit_rate_at_n": round(report.hit_rate_after_rerank, 4),
            "recall_at_n": round(report.recall_at_n, 4),
            "rerank_lift": round(report.rerank_lift, 4),
            "refusal_accuracy": round(report.refusal_accuracy, 4),
            "false_refusals": len(report.false_refusals),
            "answer_accuracy": round(report.answer_accuracy, 4) if report.generated else None,
        },
        indent=2,
    )
