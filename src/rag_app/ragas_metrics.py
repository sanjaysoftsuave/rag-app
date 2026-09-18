"""RAGAS's four metrics, implemented here rather than imported.

WHY FROM SCRATCH
----------------
Same argument as bm25.py: each of these is roughly forty lines, and reading
those forty lines is the only way to know what the number means. Imported as an
opaque dependency they are four floats with authoritative-sounding names, and
the interesting part — what counts as a "statement", whether an empty answer
scores 1.0, whether precision is rank-aware — is exactly the part the import
hides. They also each cost real money per question, and a metric whose cost you
cannot see in the code is a metric you will run by accident.

WHAT EACH ONE ASKS
------------------
  faithfulness       is every claim in the ANSWER supported by the CONTEXTS?
                     (generation-side: did the model make things up?)
  answer relevancy   does the ANSWER address the QUESTION that was asked?
                     (generation-side: did it answer something else?)
  context precision  are the USEFUL contexts ranked FIRST?
                     (retrieval-side: is the ranking any good?)
  context recall     do the CONTEXTS cover everything the REFERENCE says?
                     (retrieval-side: did we fetch enough?)

The two retrieval metrics and the two generation metrics split the same way
`label_failures()` already splits failures, and for the same reason: they lead
to different fixes.

EVERY METRIC CARRIES ITS EVIDENCE
---------------------------------
Each returns a frozen dataclass holding the statements, the per-context
verdicts, the generated questions — not just a float. A bare number cannot be
audited, and these numbers come from an LLM's judgement about text. When
faithfulness reads 0.67 the only useful next question is "which statement was
unsupported", and that has to be answerable without a second run.

ABSOLUTE VALUES DO NOT TRANSFER
-------------------------------
Answer relevancy in particular has no meaningful floor: two unrelated questions
in bge space score around 0.3-0.6, not 0. None of these numbers is comparable
across corpora or embedding models. What is comparable is the same metric on
the same questions before and after a change — which is what before_after.py is
for, and why these are reported next to a snapshot rather than as a grade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag_app.config import AppConfig, judge_llm
from rag_app.generate import is_refusal
from rag_app.llm import CallBudget, chat_once, extract_json
from rag_app.store import ScoredChunk

# How many questions to reverse-generate for answer relevancy. RAGAS uses 3.
RELEVANCY_QUESTIONS = 3

# Statements past this are truncated: a twenty-claim answer to a support
# question is a different problem from an unfaithful one, and scoring it
# precisely is not worth six more calls.
MAX_STATEMENTS = 20


# ---------------------------------------------------------------------------
# Prompts (module constants so tests assert identity, not a retyped copy)
# ---------------------------------------------------------------------------

FAITHFULNESS_STATEMENTS_SYSTEM = (
    "Break the answer into atomic factual statements.\n\n"
    "Each statement must stand alone: resolve every pronoun and reference, so that "
    '"it expired after 3 days" becomes "the password reset link expired after 3 days". '
    "A statement that cannot be understood without reading its neighbours cannot be "
    "checked against a document either.\n\n"
    "Ignore hedging, pleasantries and citations. Extract only claims about the world.\n\n"
    'Reply with one JSON object and nothing else: {"statements": ["...", "..."]}'
)

FAITHFULNESS_VERDICTS_SYSTEM = (
    "Decide, for each numbered statement, whether the CONTEXT supports it.\n\n"
    "  1 - the context directly states or clearly implies the statement\n"
    "  0 - the context does not support it, or contradicts it\n\n"
    "A statement that is TRUE IN THE WORLD but absent from the context scores 0. "
    "The question is whether this text supports it, not whether it is correct.\n\n"
    'Reply with one JSON array and nothing else, one entry per statement, in order: '
    '[{"index": 1, "verdict": 1, "reason": "..."}, ...]'
)

RELEVANCY_QUESTIONS_SYSTEM = (
    f"Read the answer and write {RELEVANCY_QUESTIONS} different questions that this "
    "answer would be a complete and direct reply to.\n\n"
    "Also decide whether the answer is NONCOMMITTAL - evasive, or a refusal, or a "
    'statement that it does not know. "I don\'t know" and "the documents do not say" '
    "are noncommittal; a specific factual answer is not.\n\n"
    'Reply with one JSON object and nothing else: '
    '{"questions": ["...", "..."], "noncommittal": true|false}'
)

CONTEXT_PRECISION_SYSTEM = (
    "Decide, for each numbered context, whether it was USEFUL for answering the "
    "question - whether an answer could draw on it.\n\n"
    "  1 - useful: contains information that helps answer the question\n"
    "  0 - not useful: on another topic, or too generic to help\n\n"
    "Judge each context on its own merits. You must return exactly one verdict per "
    "context, in the order given.\n\n"
    'Reply with one JSON array and nothing else: '
    '[{"position": 1, "useful": 1}, {"position": 2, "useful": 0}, ...]'
)

CONTEXT_RECALL_SYSTEM = (
    "Decide, for each numbered sentence of the reference answer, whether it can be "
    "ATTRIBUTED to the context - whether the context contains the information that "
    "sentence conveys.\n\n"
    "  1 - the context supports this sentence\n"
    "  0 - the context does not contain this information\n\n"
    "This measures whether retrieval fetched enough, not whether the answer was good.\n\n"
    'Reply with one JSON array and nothing else, one entry per sentence, in order: '
    '[{"index": 1, "attributed": 1, "reason": "..."}, ...]'
)


def _context_block(contexts: list[ScoredChunk]) -> str:
    """Contexts numbered by POSITION, because position is what the judge returns.

    Same discipline as `generate.build_prompt`: the label in the header is the
    token the model has to emit back. Here the ranking is the whole point of
    context precision, so the label is the rank.
    """
    return "\n\n".join(
        f"CONTEXT {i}\n{c.chunk.text}" for i, c in enumerate(contexts, start=1)
    )


def _numbered(items: list[str], word: str) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, start=1))


def _call(system: str, user: str, cfg: AppConfig) -> str:
    return chat_once(system, user, judge_llm(cfg), cfg)


def _spend(budget: CallBudget | None, n: int = 1) -> bool:
    return True if budget is None else budget.spend(n)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Faithfulness:
    score: float | None = None
    statements: list[str] = field(default_factory=list)
    supported: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    failed: bool = False
    detail: str = ""

    def describe(self) -> str:
        if self.failed or self.score is None:
            return f"faithfulness: unscored ({self.detail})"
        return (
            f"faithfulness: {self.score:.2f} "
            f"({sum(self.supported)}/{len(self.statements)} statements supported)"
        )


def faithfulness(
    answer_text: str,
    contexts: list[ScoredChunk],
    cfg: AppConfig,
    *,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> Faithfulness:
    """Fraction of the answer's atomic claims that the contexts support.

    Two LLM calls: one to decompose, one to verdict every statement at once.
    Verdicting statements individually would cost N calls and buy nothing.

    THE EMPTY-ANSWER TRAP
    ---------------------
    An answer with no extractable claims scores `unscored`, never 1.0. Scoring
    it 1.0 would make a refusal MAXIMALLY faithful, and the metric would then
    reward the gate for refusing everything — the same degenerate optimum that
    `refusal_accuracy` has to be read against `false_refusals` to avoid.
    Refusals are excluded before any call is made.
    """
    if is_refusal(answer_text) or not answer_text.strip():
        return Faithfulness(
            failed=True,
            detail="the answer was a refusal, so there are no claims to check",
        )
    if not contexts:
        return Faithfulness(failed=True, detail="no contexts were retrieved")
    if not _spend(budget, 2):
        return Faithfulness(failed=True, detail="the call budget could not afford 2 calls")

    caller = judge_fn or _call
    try:
        raw = caller(FAITHFULNESS_STATEMENTS_SYSTEM, f"ANSWER\n{answer_text}", cfg)
    except Exception as exc:
        return Faithfulness(failed=True, detail=f"statement extraction failed: {exc}")

    payload = extract_json(raw or "")
    statements = []
    if isinstance(payload, dict):
        statements = [str(x).strip() for x in (payload.get("statements") or []) if str(x).strip()]
    if not statements:
        return Faithfulness(
            failed=True, detail="the answer yielded no atomic statements to check"
        )

    truncated = len(statements) > MAX_STATEMENTS
    statements = statements[:MAX_STATEMENTS]

    user = (
        f"CONTEXT\n{_context_block(contexts)}\n\n"
        f"STATEMENTS\n{_numbered(statements, 'statement')}"
    )
    try:
        raw = caller(FAITHFULNESS_VERDICTS_SYSTEM, user, cfg)
    except Exception as exc:
        return Faithfulness(
            statements=statements, failed=True, detail=f"verdicts failed: {exc}"
        )

    rows = extract_json(raw or "")
    if not isinstance(rows, list) or len(rows) != len(statements):
        got = len(rows) if isinstance(rows, list) else 0
        return Faithfulness(
            statements=statements,
            failed=True,
            detail=(
                f"the judge returned {got} verdicts for {len(statements)} statements; "
                f"scoring the ones it did return would silently change the denominator"
            ),
        )

    supported, reasons = [], []
    for row in rows:
        value = row.get("verdict") if isinstance(row, dict) else None
        supported.append(1 if str(value).strip() in ("1", "True", "true") else 0)
        reasons.append(str(row.get("reason", "") or "") if isinstance(row, dict) else "")

    return Faithfulness(
        score=sum(supported) / len(supported),
        statements=statements,
        supported=supported,
        reasons=reasons,
        detail="truncated at 20 statements" if truncated else "",
    )


# ---------------------------------------------------------------------------
# 2. Answer relevancy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerRelevancy:
    score: float | None = None
    generated_questions: list[str] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    noncommittal: bool = False
    failed: bool = False
    detail: str = ""

    def describe(self) -> str:
        if self.failed or self.score is None:
            return f"answer relevancy: unscored ({self.detail})"
        tag = " (noncommittal)" if self.noncommittal else ""
        return f"answer relevancy: {self.score:.2f}{tag}"


def answer_relevancy(
    question: str,
    answer_text: str,
    contexts: list[ScoredChunk],
    cfg: AppConfig,
    *,
    embedder=None,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> AnswerRelevancy:
    """How closely the answer's implied questions match the one actually asked.

    One LLM call (reverse-generate questions from the answer), then a free local
    embedding comparison.

    BOTH SIDES ARE QUESTIONS, SO BOTH USE encode_queries
    ------------------------------------------------------
    The active model `BAAI/bge-small-en-v1.5` is ASYMMETRIC: it prepends an
    instruction to queries and leaves passages bare. Encoding the generated
    questions with `encode_documents` would apply that prefix to one side of a
    comparison and not the other, shifting every similarity by a constant with
    no error anywhere. That is the same class of silent degradation
    `embed.MODEL_REGISTRY` exists to prevent, applied to a metric instead of to
    retrieval, and a test enforces it with an embedder that raises if
    `encode_documents` is touched.

    Vectors come back pre-normalized (`normalize_embeddings=True`), so cosine is
    a plain dot product. Renormalizing here would be a no-op that implies the
    caller cannot rely on that invariant.
    """
    if not answer_text.strip():
        return AnswerRelevancy(failed=True, detail="the answer was empty")
    if not _spend(budget):
        return AnswerRelevancy(failed=True, detail="the call budget was exhausted")

    caller = judge_fn or _call
    user = f"QUESTION (for vocabulary only)\n{question}\n\nANSWER\n{answer_text}"
    if contexts:
        user += f"\n\nCONTEXT (for vocabulary only)\n{_context_block(contexts)}"
    try:
        raw = caller(RELEVANCY_QUESTIONS_SYSTEM, user, cfg)
    except Exception as exc:
        return AnswerRelevancy(failed=True, detail=f"question generation failed: {exc}")

    payload = extract_json(raw or "")
    if not isinstance(payload, dict):
        return AnswerRelevancy(failed=True, detail="the judge returned no JSON object")

    noncommittal = str(payload.get("noncommittal", False)).strip().lower() in ("1", "true")
    generated = [str(q).strip() for q in (payload.get("questions") or []) if str(q).strip()]

    if noncommittal:
        # RAGAS's own guard. "I don't know" is topically close to the question
        # and would otherwise score highly for saying nothing.
        return AnswerRelevancy(
            score=0.0,
            generated_questions=generated,
            noncommittal=True,
            detail="an evasive answer scores 0 regardless of similarity",
        )
    if not generated:
        return AnswerRelevancy(failed=True, detail="the judge generated no questions")

    from rag_app.embed import Embedder

    emb = embedder or Embedder(cfg.bi_encoder_model)
    # encode_queries on BOTH sides -- see the docstring.
    vectors = emb.encode_queries([question] + generated)
    original, others = vectors[0], vectors[1:]
    sims = [float(original @ v) for v in others]

    return AnswerRelevancy(
        score=sum(sims) / len(sims),
        generated_questions=generated,
        similarities=sims,
    )


# ---------------------------------------------------------------------------
# 3. Context precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextPrecision:
    score: float | None = None
    useful: list[int] = field(default_factory=list)
    failed: bool = False
    detail: str = ""

    def describe(self) -> str:
        if self.failed or self.score is None:
            return f"context precision: unscored ({self.detail})"
        return (
            f"context precision: {self.score:.2f} "
            f"({sum(self.useful)}/{len(self.useful)} useful, rank-weighted)"
        )


def average_precision(useful: list[int]) -> float:
    """Average Precision over a ranked list of 0/1 relevance flags.

        precision@k    = (# useful in positions 1..k) / k
        AP             = sum_k (precision@k * useful_k) / (total useful)

    Rank-aware is the entire point, and the reason this is not just "fraction
    useful": [1,0,0] scores 1.0 and [0,0,1] scores 0.33, because a reranker that
    puts the answer third is worse than one that puts it first even though both
    retrieved it. Same family as `EvalReport.mrr`, different input.
    """
    total = sum(useful)
    if not total:
        return 0.0
    running = 0
    acc = 0.0
    for k, flag in enumerate(useful, start=1):
        running += flag
        if flag:
            acc += running / k
    return acc / total


def context_precision(
    question: str,
    contexts: list[ScoredChunk],
    cfg: AppConfig,
    *,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> ContextPrecision:
    """Are the useful contexts ranked first? One LLM call for all of them.

    RAGAS makes one call per context. Judging them together is cheaper (3 calls
    become 1 at rerank_n=3) and lets the judge compare them against each other.
    That is a genuine DIFFERENCE, not a strict improvement: a context judged
    beside its rivals is judged on a slightly different question than one judged
    alone. Named here rather than hidden.
    """
    if not contexts:
        return ContextPrecision(failed=True, detail="no contexts were retrieved")
    if not _spend(budget):
        return ContextPrecision(failed=True, detail="the call budget was exhausted")

    caller = judge_fn or _call
    user = f"QUESTION\n{question}\n\n{_context_block(contexts)}"
    try:
        raw = caller(CONTEXT_PRECISION_SYSTEM, user, cfg)
    except Exception as exc:
        return ContextPrecision(failed=True, detail=f"the judge call failed: {exc}")

    rows = extract_json(raw or "")
    if not isinstance(rows, list) or len(rows) != len(contexts):
        got = len(rows) if isinstance(rows, list) else 0
        # Zipping a short list would score the missing contexts as if they had
        # never been retrieved -- a silently smaller denominator.
        return ContextPrecision(
            failed=True,
            detail=(
                f"the judge returned {got} verdicts for {len(contexts)} contexts; "
                f"scoring only the ones returned would change what was measured"
            ),
        )

    useful = [
        1 if str(r.get("useful")).strip() in ("1", "True", "true") else 0
        for r in rows
        if isinstance(r, dict)
    ]
    if len(useful) != len(contexts):
        return ContextPrecision(failed=True, detail="a verdict entry was not an object")

    return ContextPrecision(score=average_precision(useful), useful=useful)


# ---------------------------------------------------------------------------
# 4. Context recall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextRecall:
    score: float | None = None
    sentences: list[str] = field(default_factory=list)
    attributed: list[int] = field(default_factory=list)
    failed: bool = False
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.score is not None

    def describe(self) -> str:
        if self.score is None:
            return f"context recall: not measured ({self.detail})"
        return (
            f"context recall: {self.score:.2f} "
            f"({sum(self.attributed)}/{len(self.sentences)} reference sentences covered)"
        )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split a reference answer into sentences, deterministically.

    RAGAS spends an LLM call decomposing the reference. It does not need one: a
    human wrote the reference in sentences already, so a regex recovers exactly
    what they wrote and costs nothing. One call per question saved.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def context_recall(
    contexts: list[ScoredChunk],
    reference_answer: str,
    cfg: AppConfig,
    *,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> ContextRecall:
    """Do the contexts cover everything the reference answer says?

    WHY THIS NEEDS A NEW GOLD FIELD, AND WHY THE SHORTCUTS ARE WORSE
    -----------------------------------------------------------------
    This is the only metric here that needs a ground-truth answer. Two
    tempting ways to avoid adding one, and both are rejected:

      * derive it from `must_contain` - those are fragments ("cached",
        "3 business days"). Asking whether "cached" is attributable to a context
        is degenerate, and the result just re-measures the substring check.

      * derive it from `expect_in_chunk` - far worse. That field is DEFINED as
        text that appears in a retrieved chunk, so recall would be 1.0 by
        construction whenever retrieval succeeded. A metric that is 1.0 by
        construction is strictly worse than a missing one, because it looks
        like evidence.

    So a question with no `reference_answer` returns score=None and is reported
    as NOT MEASURED. Never 0.0: averaging a zero for missing data reports a
    retrieval regression that never happened.
    """
    sentences = split_sentences(reference_answer)
    if not sentences:
        return ContextRecall(
            detail="this gold entry has no reference_answer - add one to measure it"
        )
    if not contexts:
        return ContextRecall(failed=True, detail="no contexts were retrieved")
    if not _spend(budget):
        return ContextRecall(failed=True, detail="the call budget was exhausted")

    caller = judge_fn or _call
    user = (
        f"CONTEXT\n{_context_block(contexts)}\n\n"
        f"REFERENCE SENTENCES\n{_numbered(sentences, 'sentence')}"
    )
    try:
        raw = caller(CONTEXT_RECALL_SYSTEM, user, cfg)
    except Exception as exc:
        return ContextRecall(sentences=sentences, failed=True, detail=f"failed: {exc}")

    rows = extract_json(raw or "")
    if not isinstance(rows, list) or len(rows) != len(sentences):
        got = len(rows) if isinstance(rows, list) else 0
        return ContextRecall(
            sentences=sentences,
            failed=True,
            detail=f"the judge returned {got} verdicts for {len(sentences)} sentences",
        )

    attributed = [
        1 if str(r.get("attributed")).strip() in ("1", "True", "true") else 0
        for r in rows
        if isinstance(r, dict)
    ]
    if len(attributed) != len(sentences):
        return ContextRecall(sentences=sentences, failed=True, detail="a verdict was not an object")

    return ContextRecall(
        score=sum(attributed) / len(attributed),
        sentences=sentences,
        attributed=attributed,
    )


# ---------------------------------------------------------------------------
# Per-question bundle and aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagasScores:
    faithfulness: Faithfulness | None = None
    answer_relevancy: AnswerRelevancy | None = None
    context_precision: ContextPrecision | None = None
    context_recall: ContextRecall | None = None

    def describe(self) -> str:
        parts = [
            m.describe()
            for m in (
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_recall,
            )
            if m is not None
        ]
        return "\n".join(f"    {p}" for p in parts)


def score_question(
    question: str,
    answer_text: str,
    contexts: list[ScoredChunk],
    cfg: AppConfig,
    *,
    gold: Any = None,
    embedder=None,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> RagasScores:
    """All four metrics for one question. 4 calls, plus 1 with a reference.

    `contexts` must be `answer.reranked` — what the model actually read. Scoring
    `reranked_all` would measure a system that does not exist.
    """
    reference = (getattr(gold, "reference_answer", "") or "") if gold is not None else ""
    return RagasScores(
        faithfulness=faithfulness(answer_text, contexts, cfg, judge_fn=judge_fn, budget=budget),
        answer_relevancy=answer_relevancy(
            question, answer_text, contexts, cfg,
            embedder=embedder, judge_fn=judge_fn, budget=budget,
        ),
        context_precision=context_precision(
            question, contexts, cfg, judge_fn=judge_fn, budget=budget
        ),
        context_recall=context_recall(
            contexts, reference, cfg, judge_fn=judge_fn, budget=budget
        ),
    )


@dataclass
class RagasReport:
    rows: list[tuple[str, RagasScores]] = field(default_factory=list)

    def _values(self, name: str) -> list[float]:
        out = []
        for _, scores in self.rows:
            metric = getattr(scores, name, None)
            if metric is not None and getattr(metric, "score", None) is not None:
                if not getattr(metric, "failed", False):
                    out.append(metric.score)
        return out

    def mean(self, name: str) -> float | None:
        vals = self._values(name)
        return (sum(vals) / len(vals)) if vals else None

    def describe(self) -> str:
        lines = [f"RAGAS over {len(self.rows)} questions", ""]
        for name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            vals = self._values(name)
            label = name.replace("_", " ")
            if not vals:
                lines.append(f"  {label:<20} not measured (0 of {len(self.rows)} scored)")
                continue
            # Every mean carries its own denominator: a shrinking n must never
            # be able to look like a stable score.
            lines.append(
                f"  {label:<20} {sum(vals) / len(vals):.2f}   "
                f"(n={len(vals)} of {len(self.rows)})"
            )
        unmeasured = sum(
            1 for _, s in self.rows
            if s.context_recall is not None and s.context_recall.score is None
        )
        if unmeasured:
            lines.append("")
            lines.append(
                f"  {unmeasured} questions have no reference_answer, so context recall "
                f"could not be measured for them. They are excluded, not scored 0."
            )
        return "\n".join(lines)


def ragas_report_to_json(report: RagasReport) -> str:
    import json

    return json.dumps(
        {
            "n_questions": len(report.rows),
            "faithfulness": report.mean("faithfulness"),
            "answer_relevancy": report.mean("answer_relevancy"),
            "context_precision": report.mean("context_precision"),
            "context_recall": report.mean("context_recall"),
        },
        indent=2,
    )
