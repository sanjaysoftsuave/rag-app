"""Measuring the measurer: does the LLM judge agree with the human?

WHY THIS EXISTS
---------------
Every other module in Week 6 produces a number by asking a model. That model is
an instrument, and an instrument nobody has calibrated is a source of confident
noise. The only ground truth available is the labels a human wrote while reading
the traces — which is exactly what `error_analysis.py`'s coding sheet produces,
and why that file carries a closed `label` set rather than free text.

WHY RAW AGREEMENT IS NOT ENOUGH
-------------------------------
If the human marked 17 of 20 traces `correct` and the judge marks everything
`correct`, raw agreement is 85% and the judge is a constant function that has
learned nothing. Cohen's kappa corrects for exactly that:

    kappa = (observed - expected) / (1 - expected)

where `expected` is the agreement two raters would reach by chance given how
often each one uses each label. A constant rater scores kappa ~0 however high
its raw agreement.

WHY THE TRACE PROMPT AND NOT THE GOLD PROMPT
---------------------------------------------
`judge_trace` is used, not `judge_answer`. The human in the coding sheet saw the
question, the answer and the retrieved context — no gold reference. Scoring a
gold-referenced judge against them would measure the asymmetry between two
different tasks and report it as disagreement.

WHAT n=20 CAN AND CANNOT SUPPORT
---------------------------------
`describe()` says this unconditionally, because a number without its error bar
gets quoted without it. Twenty labels can support "this judge is not obviously
broken" and can point at specific disagreements worth reading. They cannot
support "the judge is 85% accurate", cannot rank two judge prompts a few points
apart, and cannot detect a bias affecting fewer than about four traces. At this
size the deliverable is the disagreement list, not the coefficient.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from rag_app.chunking import Chunk
from rag_app.config import AppConfig
from rag_app.error_analysis import HumanLabel, load_labels
from rag_app.judge import SCORABLE, Verdict, judge_trace
from rag_app.llm import CallBudget
from rag_app.store import ScoredChunk


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """One complete recorded request, rebuilt from a .jsonl line."""

    id: str
    question: str
    answer: str
    gate: str = ""
    used_llm: bool = False
    best_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    hallucinated_citations: list[str] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)


def load_traces(path: Path) -> list[Trace]:
    """Read a trace .jsonl, rebuilding real ScoredChunks from `reranked`.

    Rebuilding real objects rather than dicts is what lets `ragas_metrics`
    score a trace with no adapter at all — so RAGAS can be run over a Week-5
    sample that has no gold set, for every metric except context recall.
    """
    if not path.exists():
        raise FileNotFoundError(f"No trace file at {path}.")
    traces: list[Trace] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {i} is not valid JSON: {exc}") from exc

        contexts = [
            ScoredChunk(
                Chunk(
                    chunk_id=f"{row.get('id', i)}::{j}",
                    source=str(c.get("source", "")),
                    text=str(c.get("text", "")),
                    metadata={},
                ),
                float(c.get("score", 0.0)),
            )
            for j, c in enumerate(row.get("reranked") or [])
        ]
        traces.append(
            Trace(
                id=str(row.get("id", f"line-{i}")),
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                gate=str(row.get("gate", "")),
                used_llm=bool(row.get("used_llm", False)),
                best_score=float(row.get("best_score", 0.0) or 0.0),
                sources=list(row.get("sources") or []),
                hallucinated_citations=list(row.get("hallucinated_citations") or []),
                contexts=contexts,
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    labels: tuple[str, ...]
    matrix: list[list[int]]  # human rows x judge columns

    @property
    def n(self) -> int:
        return sum(sum(row) for row in self.matrix)

    @property
    def raw(self) -> float:
        if not self.n:
            return 0.0
        return sum(self.matrix[i][i] for i in range(len(self.labels))) / self.n

    @property
    def expected(self) -> float:
        """Agreement two raters would reach by chance, given their label habits."""
        if not self.n:
            return 0.0
        total = self.n
        acc = 0.0
        for i in range(len(self.labels)):
            row = sum(self.matrix[i])
            col = sum(self.matrix[r][i] for r in range(len(self.labels)))
            acc += (row / total) * (col / total)
        return acc

    @property
    def kappa(self) -> float:
        """Chance-corrected agreement. NaN when chance agreement is total."""
        expected = self.expected
        if math.isclose(expected, 1.0):
            # Both raters used exactly one label. Kappa is 0/0 — undefined, not
            # perfect, and reporting 1.0 here would be the single most
            # misleading number this module could produce.
            return float("nan")
        return (self.raw - expected) / (1.0 - expected)

    @property
    def ci95(self) -> tuple[float, float]:
        """Normal approximation on raw agreement. Wide at this n, deliberately."""
        if self.n < 2:
            return (0.0, 1.0)
        p = self.raw
        half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / self.n)
        return (max(0.0, p - half), min(1.0, p + half))

    def describe(self) -> str:
        lines = [
            f"  n = {self.n}",
            f"  raw agreement    {self.raw:6.1%}   "
            f"95% CI {self.ci95[0]:.0%}-{self.ci95[1]:.0%}",
            f"  chance agreement {self.expected:6.1%}   given how each rater uses the labels",
        ]
        if math.isnan(self.kappa):
            lines.append("  Cohen's kappa      n/a   undefined (see below)")
            lines.append("")
            lines.append(
                "  Both raters used a single label, so chance agreement is 100% and kappa"
            )
            lines.append(
                "  is 0/0. Raw agreement of 1.00 here is NOT evidence the judge works:"
            )
            lines.append(
                "  a judge that always says 'correct' scores exactly the same."
            )
        else:
            lines.append(f"  Cohen's kappa    {self.kappa:6.3f}   raw, corrected for chance")
        lines.append("")
        lines.append("  Confusion (human down, judge across):")
        head = "        " + "".join(f"{lab[:9]:>10}" for lab in self.labels)
        lines.append(head)
        for i, lab in enumerate(self.labels):
            lines.append(f"  {lab[:6]:<6}" + "".join(f"{v:>10}" for v in self.matrix[i]))
        return "\n".join(lines)


def agreement_of(pairs: list[tuple[str, str]]) -> Agreement:
    """Build a confusion matrix from (human, judge) label pairs."""
    labels = tuple(SCORABLE)
    index = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for human, judge in pairs:
        if human in index and judge in index:
            matrix[index[human]][index[judge]] += 1
    return Agreement(labels=labels, matrix=matrix)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class ValidationRow:
    trace: Trace
    human: HumanLabel
    verdict: Verdict

    @property
    def agrees(self) -> bool:
        return self.verdict.scored and self.verdict.verdict == self.human.label


@dataclass
class ValidationReport:
    rows: list[ValidationRow] = field(default_factory=list)
    unscored: list[ValidationRow] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    unlabelled_ids: list[str] = field(default_factory=list)
    judge_model: str = ""

    @property
    def agreement(self) -> Agreement:
        return agreement_of([(r.human.label, r.verdict.verdict) for r in self.rows])

    @property
    def disagreements(self) -> list[ValidationRow]:
        return [r for r in self.rows if not r.agrees]

    def describe(self) -> str:
        lines = [
            f"Judge validation - {self.judge_model or 'the judge'} vs your labels",
            "",
            self.agreement.describe(),
            "",
        ]
        if self.unscored:
            lines.append(
                f"  {len(self.unscored)} traces went unscored (judge failures). They are "
                f"excluded from the matrix, not counted as disagreement."
            )
            lines.append("")
        if self.missing_ids:
            lines.append(f"  Labelled but not in the trace file: {', '.join(self.missing_ids)}")
        if self.unlabelled_ids:
            lines.append(f"  In the trace file but unlabelled: {', '.join(self.unlabelled_ids)}")
        if self.missing_ids or self.unlabelled_ids:
            lines.append("")

        lines.append("-" * 72)
        lines.append(
            f"  WHAT {self.agreement.n} LABELS CAN SUPPORT"
        )
        lines.append("")
        lines.append(
            "  One flipped label moves raw agreement by "
            f"{(1 / self.agreement.n if self.agreement.n else 0):.0%}. These numbers can"
        )
        lines.append("  support \"the judge is not obviously broken\", and can point at the")
        lines.append("  disagreements below as worth reading. They CANNOT support \"the judge is")
        lines.append("  X% accurate\", cannot rank two judge prompts a few points apart, and")
        lines.append("  cannot detect a bias affecting fewer than about four traces.")
        lines.append("")
        lines.append("  At this sample size the deliverable is the list below, not the number.")

        if self.disagreements:
            lines.append("")
            lines.append(f"  DISAGREEMENTS ({len(self.disagreements)}):")
            for r in self.disagreements:
                lines.append("")
                lines.append(f"    {r.trace.id}  human={r.human.label}  judge={r.verdict.verdict}")
                lines.append(f"      Q: {r.trace.question}")
                if r.human.note:
                    lines.append(f"      you:   {r.human.note}")
                if r.verdict.reasoning:
                    lines.append(f"      judge: {r.verdict.reasoning}")
        return "\n".join(lines)


def validate_judge(
    traces: list[Trace],
    labels: list[HumanLabel],
    cfg: AppConfig,
    *,
    judge_fn=None,
    budget: CallBudget | None = None,
) -> ValidationReport:
    """Run the trace judge over every labelled trace and compare."""
    by_id = {t.id: t for t in traces}
    labelled = {lab.id: lab for lab in labels}

    rows: list[ValidationRow] = []
    unscored: list[ValidationRow] = []
    for lab in labels:
        trace = by_id.get(lab.id)
        if trace is None:
            continue
        verdict = judge_trace(
            trace.question, trace.answer, trace.contexts, cfg,
            judge_fn=judge_fn, budget=budget,
        )
        row = ValidationRow(trace=trace, human=lab, verdict=verdict)
        (rows if verdict.scored else unscored).append(row)

    return ValidationReport(
        rows=rows,
        unscored=unscored,
        missing_ids=[lab.id for lab in labels if lab.id not in by_id],
        unlabelled_ids=[t.id for t in traces if t.id not in labelled],
        judge_model=cfg.evaluation.judge_model,
    )


def validation_to_json(report: ValidationReport) -> str:
    a = report.agreement
    kappa = a.kappa
    return json.dumps(
        {
            "judge_model": report.judge_model,
            "n": a.n,
            "raw_agreement": round(a.raw, 4),
            "expected_agreement": round(a.expected, 4),
            "cohens_kappa": None if math.isnan(kappa) else round(kappa, 4),
            "kappa_undefined": math.isnan(kappa),
            "unscored": len(report.unscored),
            "disagreements": [
                {
                    "id": r.trace.id,
                    "human": r.human.label,
                    "judge": r.verdict.verdict,
                    "question": r.trace.question,
                }
                for r in report.disagreements
            ],
        },
        indent=2,
    )


def write_labels_template_for(trace_path: Path, labels_path: Path) -> int:
    """Convenience for the CLI: blank sheet from a trace file. Returns the count."""
    from rag_app.error_analysis import write_labels_template

    traces = load_traces(trace_path)
    write_labels_template(traces, labels_path, source=trace_path.name)
    return len(traces)


def load_labels_for(path: Path) -> list[HumanLabel]:
    return load_labels(path)
