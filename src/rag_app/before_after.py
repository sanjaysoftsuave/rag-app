"""Before/after measurement: what a change actually bought.

A leaf module over plain dicts. It deliberately imports nothing from
`evaluate.py` — a snapshot's `metrics` is literally
`json.loads(report_to_json(report))`, so any metric added to `EvalReport`
appears in future snapshots with no second edit, and a test enforces that so
the two cannot drift apart.

WHAT IT GUARDS AGAINST
----------------------
Two failure modes, both of which look completely normal while producing
meaningless numbers:

  * comparing two runs against DIFFERENT gold sets. Every delta is then
    measuring the questions, not the change. `gold_fingerprint` catches it.

  * reading a delta table with no statement of what was changed. A number
    without a cause is not a finding, so `describe()` prints the changed
    settings FIRST and the deltas second.

AND ONE THING IT REFUSES TO DO
------------------------------
It never declares a winner on a delta smaller than one question. On a 9-question
answerable set, one question flipping is 11.1%; anything below that is noise
with a decimal point on it, and `describe()` says so in its header.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direction per metric. Absent from this map means "no verdict" rather than a
# guessed one — inventing a direction for an unknown metric is how a snapshot
# tool starts reporting regressions as improvements.
HIGHER_IS_BETTER: dict[str, bool] = {
    "hit_rate_at_k": True,
    "recall_at_k": True,
    "mrr": True,
    "hit_rate_at_n": True,
    "recall_at_n": True,
    "rerank_lift": True,
    "refusal_accuracy": True,
    "false_refusals": False,
    "answer_accuracy": True,
    "substring_accuracy": True,
    "judge_accuracy": True,
    "judge_unscored": False,
    "geval_mean": True,
    "geval_stdev": False,
    "ragas_faithfulness": True,
    "ragas_answer_relevancy": True,
    "ragas_context_precision": True,
    "ragas_context_recall": True,
}

SNAPSHOT_DIRNAME = "eval"


def snapshot_dir(cfg) -> Path:
    return cfg.tickets_dir.parent / SNAPSHOT_DIRNAME


def snapshot_path(cfg, label: str) -> Path:
    return snapshot_dir(cfg) / f"{label}.json"


def gold_fingerprint(questions) -> str:
    """A stable digest of the gold set's content.

    Question text plus every expectation, so editing a `must_contain` counts as
    a different gold set — because it is one, and the deltas across it are not
    comparable.
    """
    payload = json.dumps(
        [
            {
                "question": q.question,
                "must_contain": sorted(q.must_contain),
                "expect_in_chunk": sorted(q.expect_in_chunk),
                "unanswerable": q.unanswerable,
            }
            for q in questions
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def settings_of(cfg) -> dict[str, Any]:
    """Only what changes the numbers. A snapshot is a claim about a configuration."""
    preset = cfg.chunk_presets[cfg.default_preset]
    return {
        "preset": cfg.default_preset,
        "chunk_size": preset.chunk_size,
        "overlap": preset.overlap,
        "bi_encoder_model": cfg.bi_encoder_model,
        "cross_encoder_model": cfg.cross_encoder_model,
        "retrieve_k": cfg.retrieve_k,
        "rerank_n": cfg.rerank_n,
        "score_threshold": cfg.score_threshold,
        "rerank_score_scale": cfg.rerank_score_scale,
        "retrieval_mode": cfg.retrieval.mode,
        "query_mode": cfg.retrieval.query_mode,
        "mmr": cfg.retrieval.mmr,
        "mmr_lambda": cfg.retrieval.mmr_lambda,
        "llm_model": cfg.llm.model,
        "judge_model": cfg.evaluation.judge_model,
    }


@dataclass(frozen=True)
class Snapshot:
    label: str
    created: str
    settings: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    n_questions: int = 0
    n_answerable: int = 0
    gold_fingerprint: str = ""

    def describe(self) -> str:
        return (
            f"{self.label}  ({self.created}, {self.n_questions} questions, "
            f"gold {self.gold_fingerprint})"
        )


def make_snapshot(label: str, cfg, metrics: dict, questions) -> Snapshot:
    answerable = sum(1 for q in questions if q.answerable)
    return Snapshot(
        label=label,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        settings=settings_of(cfg),
        metrics=dict(metrics),
        n_questions=len(questions),
        n_answerable=answerable,
        gold_fingerprint=gold_fingerprint(questions),
    )


def write_snapshot(snap: Snapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": snap.label,
                "created": snap.created,
                "settings": snap.settings,
                "metrics": snap.metrics,
                "n_questions": snap.n_questions,
                "n_answerable": snap.n_answerable,
                "gold_fingerprint": snap.gold_fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> Snapshot:
    if not path.exists():
        raise FileNotFoundError(
            f"No snapshot at {path}. Save one with `eval --snapshot <label>` before "
            f"comparing."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Snapshot(
        label=str(raw.get("label", path.stem)),
        created=str(raw.get("created", "")),
        settings=dict(raw.get("settings") or {}),
        metrics=dict(raw.get("metrics") or {}),
        n_questions=int(raw.get("n_questions", 0)),
        n_answerable=int(raw.get("n_answerable", 0)),
        gold_fingerprint=str(raw.get("gold_fingerprint", "")),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: Any = None
    after: Any = None

    @property
    def comparable(self) -> bool:
        return isinstance(self.before, (int, float)) and isinstance(self.after, (int, float))

    @property
    def delta(self) -> float | None:
        return (self.after - self.before) if self.comparable else None

    @property
    def verdict(self) -> str:
        """"" for a metric with no known direction — never a guess."""
        if not self.comparable or self.name not in HIGHER_IS_BETTER:
            return ""
        d = self.delta
        if d == 0:
            return "same"
        return "better" if (d > 0) == HIGHER_IS_BETTER[self.name] else "worse"

    def render(self) -> str:
        def fmt(v):
            if v is None:
                return "n/a"
            return f"{v:.4g}" if isinstance(v, float) else str(v)

        if not self.comparable:
            tag = " (new)" if self.before is None and self.after is not None else ""
            return f"  {self.name:<26}{fmt(self.before):>10} -> {fmt(self.after):>10}{tag}"
        return (
            f"  {self.name:<26}{fmt(self.before):>10} -> {fmt(self.after):>10}"
            f"   {self.delta:+.4g}   {self.verdict}"
        )


@dataclass
class Comparison:
    before: Snapshot
    after: Snapshot

    @property
    def rows(self) -> list[MetricDelta]:
        names = list(self.before.metrics) + [
            k for k in self.after.metrics if k not in self.before.metrics
        ]
        return [
            MetricDelta(n, self.before.metrics.get(n), self.after.metrics.get(n))
            for n in names
        ]

    @property
    def changed_settings(self) -> list[tuple[str, Any, Any]]:
        keys = set(self.before.settings) | set(self.after.settings)
        return [
            (k, self.before.settings.get(k), self.after.settings.get(k))
            for k in sorted(keys)
            if self.before.settings.get(k) != self.after.settings.get(k)
        ]

    @property
    def warnings(self) -> list[str]:
        out = []
        if (
            self.before.gold_fingerprint
            and self.after.gold_fingerprint
            and self.before.gold_fingerprint != self.after.gold_fingerprint
        ):
            out.append(
                f"!! the gold set changed between snapshots "
                f"({self.before.gold_fingerprint} -> {self.after.gold_fingerprint}) - "
                f"these deltas measure the questions, not the change"
            )
        if self.before.n_questions != self.after.n_questions:
            out.append(
                f"!! different question counts ({self.before.n_questions} -> "
                f"{self.after.n_questions}); rates are over different denominators"
            )
        if not self.changed_settings:
            out.append(
                "!! no configuration difference between these snapshots - any delta is "
                "run-to-run variation, not a change you made"
            )
        return out

    @property
    def one_question_worth(self) -> float:
        n = self.after.n_answerable or self.after.n_questions
        return (1.0 / n) if n else 0.0

    def describe(self) -> str:
        lines = [
            f"{self.before.label}  ->  {self.after.label}",
            f"  {self.before.created}  ->  {self.after.created}",
            "",
        ]
        # What you changed comes first: a delta table with no stated
        # intervention is a number without a cause.
        lines.append("WHAT CHANGED")
        if self.changed_settings:
            for name, b, a in self.changed_settings:
                lines.append(f"  {name:<26}{b!s:>14}  ->  {a!s}")
        else:
            lines.append("  (nothing in the recorded settings)")
        lines.append("")

        lines.append("WHAT IT BOUGHT")
        for row in self.rows:
            lines.append(row.render())
        lines.append("")

        pct = self.one_question_worth
        lines.append(
            f"  {self.after.n_answerable} answerable questions, so one question flipping "
            f"is {pct:.1%}."
        )
        lines.append(
            "  A delta smaller than that is one question's worth of noise, not a trend."
        )
        for w in self.warnings:
            lines.append("")
            lines.append(f"  {w}")
        return "\n".join(lines)


def compare(before: Snapshot, after: Snapshot) -> Comparison:
    return Comparison(before=before, after=after)


def comparison_to_json(cmp: Comparison) -> str:
    return json.dumps(
        {
            "before": cmp.before.label,
            "after": cmp.after.label,
            "changed_settings": [
                {"name": n, "before": b, "after": a} for n, b, a in cmp.changed_settings
            ],
            "metrics": [
                {
                    "name": r.name,
                    "before": r.before,
                    "after": r.after,
                    "delta": r.delta,
                    "verdict": r.verdict,
                }
                for r in cmp.rows
            ],
            "one_question_worth": round(cmp.one_question_worth, 4),
            "warnings": cmp.warnings,
        },
        indent=2,
    )
