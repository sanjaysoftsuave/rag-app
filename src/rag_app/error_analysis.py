"""Open coding, and turning the notes into a ranked list of real problems.

WHY THIS IS SCAFFOLDING AND NOT AN ANALYSIS
-------------------------------------------
Nothing in this module reads a trace and decides what went wrong. That judgement
is the one part of error analysis that cannot be automated, and automating it
would defeat the exercise: the value is in a human reading twenty real answers
and writing an honest sentence about each one BEFORE any category exists. A tool
that proposed the categories would hand back exactly the preconceptions the
method exists to get around.

So this module does the two things around that hour of reading which a human
should not do by hand:

  1. `write_labels_template()` turns a trace file into a blank sheet, one row per
     trace, carrying the severity rubric in its header so twenty judgements are
     made on one scale instead of drifting.
  2. `build_taxonomy()` groups the finished labels and ranks the groups, so the
     ranking is reproducible from the data rather than being a table someone
     typed once and cannot re-derive after changing their mind about one trace.

THE RANKING
-----------
`weight = count x mean_severity`. Both inputs are printed next to the weight,
deliberately: frequency and severity pull in different directions (one
catastrophic failure versus nine cosmetic ones) and collapsing them into a
single number hides the trade the reader is supposed to be making. The weight
orders the list; the two columns are what you argue with.

Groups are built only from labels that NAMED A CATEGORY. A trace the human
marked `correct` and left uncategorised is counted in the denominator and
belongs to no group -- this is a taxonomy of problems, not of traces. A category
whose members all have severity 0 sorts to the bottom with weight 0, which is
the right place for "clean lookup" if someone categorises their successes too.

THE FILE THIS PRODUCES IS ALSO WEEK 6'S GROUND TRUTH
-----------------------------------------------------
`label` is a closed set -- the same three values an LLM judge emits -- because
the same file is later read by `judge_validation.py` to measure whether the judge
agrees with the human. Keeping one file rather than two means the judge is
validated against the labels that were actually reasoned about, not a second set
transcribed later with less care.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The closed label set, shared with the LLM judge in Week 6. "unscored" is
# deliberately absent: it is a thing a judge returns when it fails, not a
# judgement a human who read the trace can make.
LABELS = ("correct", "partial", "incorrect")

# 0 means "not a problem" and is what an uncategorised correct answer carries.
SEVERITIES = (0, 1, 2, 3)

SEVERITY_RUBRIC = {
    3: "harmful  - a confident wrong answer someone would act on, or a real "
       "source cited for a claim it does not support",
    2: "useless  - a refusal on an answerable question, or an answer that omits "
       "the fact that was actually asked for",
    1: "annoying - correct, but padded, badly ordered, or citing more than it needed",
    0: "not a problem",
}


# ---------------------------------------------------------------------------
# Traces (only the two fields a blank sheet needs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceStub:
    """Just enough of a trace to write a row for it.

    Week 6's `judge_validation.load_traces()` returns a much richer object that
    also carries the contexts and rebuilds real `ScoredChunk`s. Both satisfy the
    `.id` / `.question` shape `write_labels_template()` asks for, so neither has
    to know about the other.
    """

    id: str
    question: str


def load_trace_stubs(path: Path) -> list[TraceStub]:
    """Read `id` and `question` from a trace .jsonl, one object per line."""
    if not path.exists():
        raise FileNotFoundError(
            f"No trace file at {path}. Error analysis reads a batch of real answers; "
            f"generate one before labelling it."
        )
    stubs: list[TraceStub] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {i} is not valid JSON: {exc}") from exc
        if "id" not in row:
            raise ValueError(f"{path.name} line {i} has no 'id'.")
        stubs.append(TraceStub(id=str(row["id"]), question=str(row.get("question", ""))))
    return stubs


# ---------------------------------------------------------------------------
# The labels a human writes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanLabel:
    id: str
    label: str
    category: str = ""
    severity: int = 0
    note: str = ""

    @property
    def failed(self) -> bool:
        return self.label in ("partial", "incorrect")


def _template_header(source: str) -> str:
    rubric = "\n".join(
        f"#     {k}  {v}" for k, v in sorted(SEVERITY_RUBRIC.items(), reverse=True)
    )
    legal = " | ".join(LABELS)
    return (
        f"# Open-coding labels for {source}\n"
        f"#\n"
        f"# Read the trace, write the `note` FIRST -- one honest sentence about what, if\n"
        f"# anything, went wrong -- and only invent a `category` once you have read enough\n"
        f"# of them to see a pattern. Categories that exist before the notes are\n"
        f"# preconceptions, not findings.\n"
        f"#\n"
        f"#   label     one of: {legal}\n"
        f"#             the same closed set the LLM judge emits, so Week 6 can measure\n"
        f"#             whether the judge agrees with you\n"
        f"#   category  your own short name for the problem. Invent it; do not pick from\n"
        f"#             a list. Leave blank for a trace with no problem.\n"
        f"#   severity  how much it hurts, not how often it happens:\n"
        f"{rubric}\n"
        f"#   note      the sentence. This is the part that cannot be automated.\n"
        f"#\n"
        f"# `python -m rag_app codes` ranks the categories once every row has a label.\n"
    )


def write_labels_template(traces, path: Path, *, source: str = "the trace file") -> None:
    """Write one blank row per trace.

    Refuses to overwrite. The finished file is an hour of reading that exists
    nowhere else -- a silent clobber would be the single most expensive thing
    this module could do.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists and holds your labels. Delete it deliberately if you "
            f"really mean to start the reading over."
        )
    legal = " | ".join(LABELS)
    rows = []
    for t in traces:
        rows.append(
            f"- id: {t.id}\n"
            f"  label:            # {legal}\n"
            f"  category:         # invent one, or leave blank if nothing went wrong\n"
            f"  severity:         # 0-3, see the rubric above\n"
            f"  note:             # one honest sentence\n"
            f"  # Q: {t.question}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_template_header(source) + "\n" + "\n".join(rows), encoding="utf-8")


def parse_labels(raw: Any) -> list[HumanLabel]:
    """Build labels from parsed YAML, naming exactly what is wrong with which row.

    Blank rows are collected and reported together rather than failing on the
    first one: a half-filled sheet is the normal state halfway through the
    reading, and "8 of 20 rows are blank" is a useful message where "entry 3 has
    no label" is a puzzle.
    """
    if raw is None:
        raise ValueError("The labels file is empty. Run `codes --init` to write a template.")
    if not isinstance(raw, list):
        raise ValueError("The labels file must be a list of entries.")

    labels: list[HumanLabel] = []
    blank: list[str] = []
    seen: set[str] = set()

    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Labels entry {i} is not a mapping: {entry!r}")
        trace_id = entry.get("id")
        if not trace_id:
            raise ValueError(f"Labels entry {i} has no 'id'.")
        trace_id = str(trace_id)
        if trace_id in seen:
            raise ValueError(
                f"Labels entry {i} repeats id {trace_id!r}. Each trace gets exactly one "
                f"label; two would silently double its weight in the ranking."
            )
        seen.add(trace_id)

        label = entry.get("label")
        if label is None or str(label).strip() == "":
            blank.append(trace_id)
            continue
        label = str(label).strip().lower()
        if label not in LABELS:
            raise ValueError(
                f"Labels entry {i} ({trace_id}) has label {label!r}, which is not one of "
                f"{list(LABELS)}. The set is closed so the LLM judge can be scored against it."
            )

        raw_sev = entry.get("severity")
        if raw_sev is None or str(raw_sev).strip() == "":
            severity = 0
        else:
            try:
                severity = int(raw_sev)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Labels entry {i} ({trace_id}) has severity {raw_sev!r}, which is not "
                    f"a whole number. Legal values are {list(SEVERITIES)}."
                ) from None
            if severity not in SEVERITIES:
                raise ValueError(
                    f"Labels entry {i} ({trace_id}) has severity {severity}, outside "
                    f"{list(SEVERITIES)}. Severity is how much it hurts, not how often "
                    f"it happens -- frequency is counted for you."
                )

        labels.append(
            HumanLabel(
                id=trace_id,
                label=label,
                category=str(entry.get("category") or "").strip(),
                severity=severity,
                note=str(entry.get("note") or "").strip(),
            )
        )

    if blank:
        shown = ", ".join(blank[:5]) + (", ..." if len(blank) > 5 else "")
        raise ValueError(
            f"{len(blank)} of {len(raw)} rows have no 'label' yet ({shown}). Ranking a "
            f"partly-read sample would report a taxonomy of whichever traces happened to "
            f"be read first. Finish the reading, or delete the rows you have not done."
        )
    return labels


def load_labels(path: Path) -> list[HumanLabel]:
    if not path.exists():
        raise FileNotFoundError(
            f"No labels at {path}. Run `python -m rag_app codes --init` to write a blank "
            f"sheet, then read the traces and fill it in."
        )
    return parse_labels(yaml.safe_load(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryGroup:
    category: str
    trace_ids: list[str]
    severities: list[int]
    notes: list[str]

    @property
    def count(self) -> int:
        return len(self.trace_ids)

    @property
    def mean_severity(self) -> float:
        return (sum(self.severities) / len(self.severities)) if self.severities else 0.0

    @property
    def weight(self) -> float:
        """Frequency x severity. Both factors stay visible in `describe()`."""
        return self.count * self.mean_severity


@dataclass
class Taxonomy:
    groups: list[CategoryGroup]
    n_labelled: int
    n_uncategorised: int
    label_counts: dict[str, int]

    @property
    def top(self) -> CategoryGroup | None:
        return self.groups[0] if self.groups else None

    def describe(self) -> str:
        counts = "  ".join(f"{k}={self.label_counts.get(k, 0)}" for k in LABELS)
        lines = [
            f"{self.n_labelled} traces labelled:  {counts}",
            f"{self.n_uncategorised} carried no category "
            f"(a taxonomy of problems, not of traces)",
            "",
        ]
        if not self.groups:
            lines.append(
                "  No categories yet. Group the notes you wrote into named problem types "
                "and put the name in each row's `category`."
            )
            return "\n".join(lines)

        header = f"  {'rank':<6}{'weight':<9}{'count':<8}{'mean sev':<11}category"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for rank, g in enumerate(self.groups, start=1):
            lines.append(
                f"  {rank:<6}{g.weight:<9.1f}{g.count:<8}{g.mean_severity:<11.1f}{g.category}"
            )
        lines.append("")
        lines.append("  weight = count x mean severity. Both columns are shown because they")
        lines.append("  pull in different directions: one catastrophe and nine annoyances")
        lines.append("  can weigh the same and are not the same problem.")
        lines.append("")

        for rank, g in enumerate(self.groups, start=1):
            lines.append(
                f"  [{rank}] {g.category}  ({g.count} traces, "
                f"mean severity {g.mean_severity:.1f})"
            )
            for tid, note in zip(g.trace_ids, g.notes):
                lines.append(f"      {tid}  {note}" if note else f"      {tid}")
            lines.append("")

        top = self.groups[0]
        lines.append("-" * 72)
        lines.append(f"  FIX TARGET: {top.category}")
        lines.append("")
        lines.append("  Before changing anything, write down what you expect to happen:")
        lines.append("")
        lines.append("    The change I will make:      ...")
        lines.append(
            f"    What I expect it to fix:     ... of the {top.count} traces in this group"
        )
        lines.append("    What it might break:         ...")
        lines.append(
            "    How I will know:             `eval --snapshot before` / `--snapshot after`"
        )
        lines.append("")
        lines.append("  Writing the prediction first is what makes the after-measurement")
        lines.append("  capable of surprising you. A prediction written afterwards never is.")
        return "\n".join(lines)


def build_taxonomy(labels: list[HumanLabel]) -> Taxonomy:
    """Group by category and rank by frequency x severity."""
    grouped: dict[str, list[HumanLabel]] = {}
    uncategorised = 0
    label_counts: dict[str, int] = {}

    for lab in labels:
        label_counts[lab.label] = label_counts.get(lab.label, 0) + 1
        if not lab.category:
            uncategorised += 1
            continue
        grouped.setdefault(lab.category, []).append(lab)

    groups = [
        CategoryGroup(
            category=name,
            trace_ids=[x.id for x in rows],
            severities=[x.severity for x in rows],
            notes=[x.note for x in rows],
        )
        for name, rows in grouped.items()
    ]
    # Ties broken by count, then by name, so the ordering is stable across runs.
    groups.sort(key=lambda g: (-g.weight, -g.count, g.category))

    return Taxonomy(
        groups=groups,
        n_labelled=len(labels),
        n_uncategorised=uncategorised,
        label_counts=label_counts,
    )


def taxonomy_to_json(tax: Taxonomy) -> str:
    return json.dumps(
        {
            "n_labelled": tax.n_labelled,
            "n_uncategorised": tax.n_uncategorised,
            "label_counts": tax.label_counts,
            "groups": [
                {
                    "rank": i,
                    "category": g.category,
                    "count": g.count,
                    "mean_severity": round(g.mean_severity, 3),
                    "weight": round(g.weight, 3),
                    "trace_ids": g.trace_ids,
                }
                for i, g in enumerate(tax.groups, start=1)
            ],
            "fix_target": tax.top.category if tax.top else None,
        },
        indent=2,
    )
