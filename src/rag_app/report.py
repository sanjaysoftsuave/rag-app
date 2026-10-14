"""`python -m rag_app report`: every number, rendered FROM the snapshot files.

WHY THIS EXISTS
----------------
In Week 11, a cost figure was typed into CLAUDE.md and README.md attributed to
`logs --replay-gold --generate`. That command had never run; the figure was a
different snapshot's (the Week 10 agent run), rounded and relabelled. Nothing
in the repo could have caught it, because the number lived in prose and the
snapshot that would have contradicted it was never consulted.

The structural fix is that numbers are not typed. This module reads the
snapshot JSON that the measuring commands wrote, and renders them with their
provenance: when, what changed, how many questions, what one question is
worth, and — the field whose absence made the mistake possible — whether a
real model produced them at all. Prose links to the generated report.

NOTHING HERE RE-DERIVES A DIFF
-------------------------------
For pairs that belong together it embeds `before_after.compare(...).describe()`
verbatim. A second diff implementation is a second place for "better" and
"worse" to disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_app.before_after import Snapshot, compare, load_snapshot

# Pairs that only mean something side by side. Missing members are skipped.
KNOWN_PAIRS = (
    ("single-v1", "team-v1", "Single ReAct agent vs manager + 2 specialists"),
    ("cache-cold", "cache-warm", "Answer cache: empty vs populated"),
    ("rt-before", "rt-after", "Injection defences: off vs on"),
    # Two things change at once here, and the report must not hide it: the
    # matcher fix (recall) and generated -> retrieval only (refusal accuracy
    # then measures the gate alone, which is where CS-9999 shows up).
    ("main-v1", "main-v1-fold",
     "Main path: real model vs gate only, after the whitespace-fold matcher fix"),
)

KEY_METRICS = {
    "eval": ("hit_rate_at_k", "recall_at_k", "mrr", "refusal_accuracy", "false_refusals",
             "answer_accuracy", "substring_accuracy", "judge_accuracy"),
    "agent": ("task_success", "refusal_accuracy", "mean_llm_calls", "mean_tokens",
              "mean_cost_usd", "mean_wall_clock_s"),
    "redteam": ("injection_success_rate", "forged_citation_rate", "leak_rate",
                "false_refusal_rate", "detector_false_positive_rate"),
    "requests": ("n_requests", "cache_hit_rate", "mean_latency_s", "mean_tokens",
                 "mean_cost_usd", "p99_cost_usd", "planted_excluded"),
}

# Kinds whose numbers only exist if a model produced the answers. An eval
# without --generate is different: its retrieval metrics are real either way.
NEEDS_A_MODEL = {"agent", "redteam", "requests"}


@dataclass(frozen=True)
class Located:
    snapshot: Snapshot
    corpus: str  # "your corpus" | "sample corpus"


def kind_of(snapshot: Snapshot) -> str:
    m = snapshot.metrics
    if "hit_rate_at_k" in m:
        return "eval"
    if "task_success" in m:
        return "agent"
    if "injection_success_rate" in m:
        return "redteam"
    if "cache_hit_rate" in m:
        return "requests"
    return "other"


def provenance(snapshot: Snapshot) -> str:
    """One honest phrase about where the numbers came from."""
    kind = kind_of(snapshot)
    generated = snapshot.metrics.get("generated")
    if generated is None:
        return "PROVENANCE UNKNOWN — snapshot predates the `generated` flag"
    if generated:
        return "real model"
    if kind in NEEDS_A_MODEL:
        return "SCRIPTED STAND-IN — NOT A REAL MODEL"
    return "retrieval only — no answers generated (retrieval metrics are real)"


def load_dir(folder: Path, corpus: str) -> list[Located]:
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            out.append(Located(load_snapshot(path), corpus))
        except (ValueError, KeyError, TypeError):
            continue  # not a snapshot; a report must not die on a stray file
    return out


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}" if 0 < abs(value) < 0.01 else f"{value:.4g}"
    return str(value)


def _section(loc: Located) -> list[str]:
    s = loc.snapshot
    kind = kind_of(s)
    n = s.n_answerable or s.n_questions
    lines = [
        f"### `{s.label}` — {kind}, {loc.corpus}",
        "",
        f"- **Source:** {provenance(s)}",
        f"- **Created:** {s.created}",
    ]
    note = s.settings.get("run_note")
    if note:
        lines.append(f"- **Note:** {note}")
    if n:
        lines.append(
            f"- **Base:** {s.n_questions} questions, {s.n_answerable} answerable — one "
            f"answerable question is {100 / n:.1f}% of any answerable-only rate"
        )
    lines += ["", "| metric | value |", "|---|---|"]
    for key in KEY_METRICS.get(kind, ()):
        if key in s.metrics:
            lines.append(f"| `{key}` | {_fmt(s.metrics[key])} |")
    lines.append("")
    return lines


def not_yet_measured(located: list[Located]) -> list[str]:
    real = {kind_of(l.snapshot) for l in located if l.snapshot.metrics.get("generated")}
    answers = any(
        kind_of(l.snapshot) == "eval" and l.snapshot.metrics.get("generated") for l in located
    )
    judged = any(l.snapshot.metrics.get("judge_accuracy") is not None for l in located)
    ragas = any(
        l.snapshot.metrics.get("ragas_faithfulness") is not None for l in located
    )
    gaps = []
    if not answers:
        gaps.append("Answer accuracy on the main `ask()` path — run `eval --generate --snapshot …`.")
    if "requests" not in real:
        gaps.append("Real cost per request — run `logs --replay-gold --generate --snapshot cache-cold`.")
    if "redteam" not in real:
        gaps.append(
            "Injection resistance of a REAL model — every red-team number above (if any) is a "
            "scripted stand-in. Run `redteam --generate`."
        )
    if not judged:
        gaps.append("LLM-judge accuracy — `eval --generate --judge` has not produced a snapshot.")
    gaps.append(
        "Judge validation against human labels — `judge` writes no snapshot, and it needs the "
        "20-trace label sheet filled in first (`python -m rag_app codes` reports progress)."
    )
    if not ragas:
        gaps.append("RAGAS — `eval --generate --ragas` has not produced a snapshot.")
    return gaps


def render_report(located: list[Located]) -> str:
    lines = [
        "# Evaluation report",
        "",
        "Generated by `python -m rag_app report` from the snapshot files in `data/eval/`",
        "(your corpus) and `data/sample/eval/` (the bundled sample). **Do not edit by hand** —",
        "re-run the command. Every number below says whether a real model produced it.",
        "",
    ]
    if not located:
        lines += ["No snapshots yet. Every measuring command takes `--snapshot LABEL`.", ""]
    by_label = {l.snapshot.label: l for l in located}

    pair_lines: list[str] = []
    for before, after, title in KNOWN_PAIRS:
        if before in by_label and after in by_label:
            a, b = by_label[before].snapshot, by_label[after].snapshot
            pair_lines += [
                f"### {title}",
                "",
                f"`{before}`: {provenance(a)} · `{after}`: {provenance(b)}",
                "",
                "```",
                compare(a, b).describe(),
                "```",
                "",
            ]
    if pair_lines:
        lines += ["## Comparisons", "", *pair_lines]

    if located:
        lines += ["## Every snapshot", ""]
        for loc in located:
            lines += _section(loc)

    lines += ["## Not yet measured", ""]
    lines += [f"- {gap}" for gap in not_yet_measured(located)]
    lines.append("")
    return "\n".join(lines)
