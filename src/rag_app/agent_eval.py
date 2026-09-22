"""Trajectory-level evaluation: judging HOW the agent got there, not only what it said.

WHY QUESTION -> ANSWER IS NOT ENOUGH
--------------------------------------
`data/gold.yaml` grades an answer. An agent can produce the right answer by a
route you would never ship: eight steps to answer a one-hop question, a tool it
should not have needed, an answer assembled without ever citing the document it
came from. It can also produce a wrong answer by a perfect route, when the
document simply does not say.

So an agent task carries expectations about the TRAJECTORY as well as the text:
which tools should appear, which must not, how many steps it should take, and
which sources the answer must cite.

READ SUCCESS AND EFFICIENCY TOGETHER
-------------------------------------
Neither number means much alone, in the same way `refusal_accuracy` is
meaningless without `false_refusals`. An agent that always answers in one step
scores perfectly on budget adherence and may be answering from its own weights;
an agent that always takes six steps is thorough and unaffordable.
`stop_reason_counts` is the histogram that says which budget is actually
binding — usually the most actionable number in the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_app.config import AppConfig

TASKS_FILENAME = "agent_tasks.yaml"


@dataclass(frozen=True)
class AgentTask:
    task: str
    must_contain: list[str] = field(default_factory=list)
    expect_sources: list[str] = field(default_factory=list)
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    max_steps: int = 0
    min_steps: int = 0
    unanswerable: bool = False
    note: str = ""
    # Order matters sometimes: you cannot read a source before you know it
    # exists. `expect_tools` is a SET claim, this is an ORDER claim, and they
    # are kept apart deliberately — see matches_sequence().
    expect_sequence: list[str] = field(default_factory=list)
    sequence_match: str = "subsequence"

    @property
    def answerable(self) -> bool:
        return not self.unanswerable


def percentile(values: list[float], p: float) -> float:
    """Nearest rank: sorted(values)[ceil(p/100 * n) - 1]. No interpolation.

    `statistics.quantiles` would interpolate between the 7th and 8th of eight
    samples and hand back a number implying a distribution that does not exist.
    Nearest rank returns a value that was actually measured, which on a corpus
    this size is the only honest kind of answer.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(1, min(rank, len(ordered))) - 1]


def percentile_is_max(n: int, p: float) -> bool:
    """True when nearest-rank p over n samples can only ever BE the maximum.

    At p99 this holds for every n below 100, so the report says so rather than
    printing a number that implies a tail it does not have.
    """
    return n > 0 and math.ceil(p / 100 * n) >= n


@dataclass(frozen=True)
class CostStat:
    name: str
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return (sum(self.values) / self.n) if self.n else 0.0

    @property
    def p99(self) -> float:
        return percentile(self.values, 99)


SEQUENCE_MATCHES = ("subsequence", "prefix", "exact")


def matches_sequence(used: list[str], expected: list[str], mode: str) -> bool:
    """Does the tool order satisfy the task's order claim?

    Vacuously True on an empty `expected`, matching `answered`'s convention:
    a task that claims nothing cannot fail the claim.

      subsequence  the expected names appear in that relative order, with
                   anything allowed in between. THE DEFAULT.
      prefix       the run OPENS with exactly these, in order.
      exact        the run is exactly these, duplicates and all.

    WHY SUBSEQUENCE IS THE DEFAULT. An order expectation is a claim about
    DEPENDENCY — "you cannot read a source before you know it exists" — not
    about count. `exact` punishes a harmless corroborating search, and on a
    corpus of eight tasks a metric that fires on harmless extras sits near zero
    and tells you nothing about order. Subsequence isolates the ordering claim
    from the set claim, which is the whole reason the two metrics are reported
    side by side. `exact` stays available for the rare task where the count
    genuinely is the point, and the task file records which rule it asked for.
    """
    if not expected:
        return True
    if mode == "exact":
        return used == expected
    if mode == "prefix":
        return used[: len(expected)] == expected
    # subsequence: greedy two-pointer
    it = iter(used)
    return all(name in it for name in expected)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def parse_agent_tasks(raw: Any) -> list[AgentTask]:
    """Same error discipline as `parse_gold`: name the entry that is wrong.

    Silently skipping a malformed task would quietly shrink the denominator of
    every metric below, which is how an evaluation harness starts flattering the
    thing it measures.
    """
    if not isinstance(raw, list):
        raise ValueError("The agent task set must be a list of entries.")
    tasks: list[AgentTask] = []
    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Agent task {i} is not a mapping: {entry!r}")
        text = entry.get("task") or entry.get("question")
        if not text:
            raise ValueError(f"Agent task {i} has no 'task'.")
        unanswerable = bool(entry.get("unanswerable", False))
        must = _as_list(entry.get("must_contain"))
        tools = _as_list(entry.get("expect_tools"))
        sequence = _as_list(entry.get("expect_sequence"))
        if not unanswerable and not (must or tools or sequence):
            raise ValueError(
                f"Agent task {i} ({text!r}) is answerable but names nothing to check. "
                f"Add 'must_contain', 'expect_tools' or 'expect_sequence', or mark it "
                f"'unanswerable: true'."
            )
        match = str(entry.get("sequence_match", "subsequence"))
        if match not in SEQUENCE_MATCHES:
            raise ValueError(
                f"Agent task {i} ({text!r}) has sequence_match {match!r}, which is not "
                f"one of {list(SEQUENCE_MATCHES)}. The rule decides whether an extra "
                f"corroborating tool call counts as wrong order, so it cannot be guessed."
            )
        tasks.append(
            AgentTask(
                task=str(text),
                must_contain=must,
                expect_sources=_as_list(entry.get("expect_sources")),
                expect_tools=tools,
                forbid_tools=_as_list(entry.get("forbid_tools")),
                max_steps=int(entry.get("max_steps", 0)),
                min_steps=int(entry.get("min_steps", 0)),
                unanswerable=unanswerable,
                note=str(entry.get("note", "")),
                expect_sequence=sequence,
                sequence_match=match,
            )
        )
    return tasks


def tasks_path(cfg: AppConfig) -> Path:
    return cfg.tickets_dir.parent / TASKS_FILENAME


def load_agent_tasks(cfg: AppConfig, path: Path | None = None) -> list[AgentTask]:
    target = path or tasks_path(cfg)
    if not target.exists():
        raise FileNotFoundError(
            f"No agent tasks at {target}. Trajectory evaluation needs tasks whose correct "
            f"route you already know — see agent_tasks.example.yaml at the repo root."
        )
    return parse_agent_tasks(yaml.safe_load(target.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task: AgentTask
    result: Any  # AgentResult

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.result.steps if s.tool]

    @property
    def answered(self) -> bool:
        low = (self.result.text or "").lower()
        return all(s.lower() in low for s in self.task.must_contain)

    @property
    def cited(self) -> bool:
        return all(s in self.result.sources for s in self.task.expect_sources)

    @property
    def success(self) -> bool:
        if not self.task.answerable:
            # `_stop()` sets refused=True for EVERY budget trip, so `refused`
            # alone credited a task that merely ran out of steps as a correct
            # refusal. A refusal has to be a DECISION, not an exhaustion.
            #
            # Deliberately `in REFUSAL_STOPS` rather than `refused and not
            # failed`: `no-evidence` sets failed=True and is the BEST kind of
            # refusal — the gate caught an ungrounded answer.
            from rag_app.agent import REFUSAL_STOPS

            return self.result.stop_reason in REFUSAL_STOPS
        return self.answered and self.cited

    @property
    def tool_choice_ok(self) -> bool:
        used = set(self.tools_used)
        if any(t not in used for t in self.task.expect_tools):
            return False
        return not any(t in used for t in self.task.forbid_tools)

    @property
    def failures(self):
        """Every failure mode this trajectory exhibits. See failure_modes.py."""
        from rag_app.failure_modes import classify_failures

        return classify_failures(self.task, self.result)

    @property
    def path_failures(self):
        from rag_app.failure_modes import PATH_MODES

        return [f for f in self.failures if f.mode in PATH_MODES]

    @property
    def path_ok(self) -> bool:
        """Defined ONLY over the flag set, so wrong-tool / wrong-sequence /
        step-target-missed have one definition that cannot drift from the
        classifier's — the explain-computes / ui-renders split, applied here."""
        return not self.path_failures

    @property
    def sequence_checked(self) -> bool:
        return bool(self.task.expect_sequence)

    @property
    def sequence_ok(self) -> bool:
        return matches_sequence(
            self.tools_used, self.task.expect_sequence, self.task.sequence_match
        )

    @property
    def right_answer_wrong_path(self) -> bool:
        """The headline. An answer you would ship, by a route you would not."""
        return self.success and not self.path_ok

    @property
    def wrong_answer_right_path(self) -> bool:
        """The mirror. Points at the corpus or the task file, not the agent."""
        return not self.success and self.path_ok

    @property
    def budget_ok(self) -> bool:
        """Finished on its own terms, and within the task's own step window.

        "Its own terms" INCLUDES refusing: a correct refusal stops at
        `no-evidence`/`model-refused`, which is a decision, not an exhausted
        budget. Requiring `final-answer` made every unanswerable task
        structurally unreachable and understated the metric by its whole
        denominator.
        """
        from rag_app.agent import SELF_TERMINATED

        if self.result.stop_reason not in SELF_TERMINATED:
            return False
        steps = len(self.result.steps)
        if self.task.max_steps and steps > self.task.max_steps:
            return False
        if self.task.min_steps and steps < self.task.min_steps:
            return False
        return True


@dataclass
class AgentEvalReport:
    results: list[TaskResult] = field(default_factory=list)
    generated: bool = False

    @property
    def answerable(self) -> list[TaskResult]:
        return [r for r in self.results if r.task.answerable]

    @property
    def unanswerable(self) -> list[TaskResult]:
        return [r for r in self.results if not r.task.answerable]

    def _frac(self, values: list[bool]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    @property
    def task_success(self) -> float:
        return self._frac([r.success for r in self.answerable])

    @property
    def refusal_accuracy(self) -> float:
        """Delegates to `success` so the definition of "a correct refusal" lives
        in exactly one place and the two numbers cannot drift apart."""
        return self._frac([r.success for r in self.unanswerable])

    @property
    def false_refusals(self) -> list[TaskResult]:
        return [r for r in self.answerable if r.result.refused]

    @property
    def tool_choice_accuracy(self) -> float:
        return self._frac([r.tool_choice_ok for r in self.results])

    @property
    def budget_adherence(self) -> float:
        return self._frac([r.budget_ok for r in self.results])

    def _mean(self, fn) -> float:
        return (sum(fn(r) for r in self.results) / len(self.results)) if self.results else 0.0

    @property
    def mean_steps(self) -> float:
        return self._mean(lambda r: len(r.result.steps))

    @property
    def mean_tool_calls(self) -> float:
        return self._mean(lambda r: r.result.tool_calls)

    @property
    def mean_llm_calls(self) -> float:
        return self._mean(lambda r: r.result.llm_calls)

    @property
    def hallucinated_citation_rate(self) -> float:
        return self._frac([bool(r.result.hallucinated_citations) for r in self.results])

    @property
    def no_evidence_rate(self) -> float:
        return self._frac([r.result.stop_reason == "no-evidence" for r in self.results])

    # ---- order, and the gap ------------------------------------------------

    @property
    def sequence_checked(self) -> list[TaskResult]:
        return [r for r in self.results if r.sequence_checked]

    @property
    def tool_sequence_accuracy(self) -> float:
        """Denominator: tasks that DECLARE an order. Deliberately different from
        `tool_choice_accuracy`'s all-tasks denominator — that difference is the
        point, and describe() states both."""
        return self._frac([r.sequence_ok for r in self.sequence_checked])

    @property
    def right_tools_wrong_order(self) -> list[TaskResult]:
        return [r for r in self.sequence_checked if r.tool_choice_ok and not r.sequence_ok]

    @property
    def tool_choice_checked(self) -> list[TaskResult]:
        return [
            r for r in self.results if r.task.expect_tools or r.task.forbid_tools
        ]

    @property
    def outcome_trajectory_gap(self) -> list[TaskResult]:
        return [r for r in self.results if r.right_answer_wrong_path]

    @property
    def clean_path_failures(self) -> list[TaskResult]:
        return [r for r in self.results if r.wrong_answer_right_path]

    @property
    def successful(self) -> list[TaskResult]:
        return [r for r in self.results if r.success]

    @property
    def gap_rate(self) -> float:
        """Of the answers you would have SHIPPED, how many arrived by a route
        you would not ship. Denominator is successful tasks, not all tasks."""
        return self._frac([r.right_answer_wrong_path for r in self.successful])

    @property
    def outcome_path_matrix(self) -> dict[tuple[bool, bool], int]:
        counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for r in self.results:
            counts[(r.success, r.path_ok)] += 1
        return counts

    @property
    def failure_counts(self) -> list[tuple[str, int]]:
        from rag_app.failure_modes import rank_modes

        return rank_modes([r.failures for r in self.results])

    @property
    def stopped_on_budget(self) -> list[TaskResult]:
        from rag_app.agent import BUDGET_STOPS

        return [r for r in self.results if r.result.stop_reason in BUDGET_STOPS]

    @property
    def cost_stats(self) -> dict[str, CostStat]:
        """Cost over ALL tasks, refusals included — a refusal costs money too,
        and saying so is the point."""
        dims = {
            "steps": lambda r: len(r.result.steps),
            "llm calls": lambda r: r.result.llm_calls,
            "tool calls": lambda r: r.result.tool_calls,
            "prompt chars": lambda r: r.result.prompt_chars,
            "wall clock (s)": lambda r: r.result.elapsed_s,
        }
        return {
            name: CostStat(name, [float(fn(r)) for r in self.results])
            for name, fn in dims.items()
        }

    def _cost_lines(self) -> list[str]:
        if not self.results:
            return []
        stats = self.cost_stats
        lines = [
            "",
            f"  COST PER TASK  (over all {len(self.results)} tasks, refusals included)",
            "",
            f"    {'':<18}{'mean':>10}{'p99':>10}",
        ]
        for name, stat in stats.items():
            lines.append(f"    {name:<18}{stat.mean:>10,.1f}{stat.p99:>10,.1f}")
        lines.append("")
        if percentile_is_max(len(self.results), 99):
            lines.append(
                f"    p99 over {len(self.results)} tasks IS the maximum — nearest rank, no"
            )
            lines.append(
                "    interpolation. It only starts being a percentile above 100 tasks."
            )
        lines.append(
            "    Cost is calls, characters and seconds. A dollar figure needs a tokenizer"
        )
        lines.append("    and a price table this project does not carry.")
        return lines

    @property
    def stop_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.result.stop_reason] = counts.get(r.result.stop_reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _sequence_lines(self) -> list[str]:
        rows = self.right_tools_wrong_order
        if not rows:
            return []
        lines = ["", f"  right tools, wrong order ({len(rows)}):"]
        for r in rows:
            lines.append(f"    - {r.task.task}")
            lines.append(
                f"        expected {' -> '.join(r.task.expect_sequence)}"
                f" ({r.task.sequence_match}); got [{', '.join(r.tools_used) or 'nothing'}]"
            )
        return lines

    def _gap_lines(self) -> list[str]:
        if not self.results:
            return []
        m = self.outcome_path_matrix
        lines = [
            "",
            "  OUTCOME vs TRAJECTORY  (the gap this section exists to find)",
            "",
            "                    path clean   path bad",
            f"    answer right   {m[(True, True)]:>10}{m[(True, False)]:>11}",
            f"    answer wrong   {m[(False, True)]:>10}{m[(False, False)]:>11}",
        ]

        gap = self.outcome_trajectory_gap
        if gap:
            lines.append("")
            lines.append(
                f"    RIGHT ANSWER, WRONG PATH  ({len(gap)} of "
                f"{len(self.successful)} scored successful):"
            )
            for r in gap:
                lines.append(f"      - {r.task.task}")
                for flag in r.path_failures:
                    lines.append(f"          {flag.describe()}")

        clean = self.clean_path_failures
        if clean:
            lines.append("")
            lines.append(f"    WRONG ANSWER, RIGHT PATH  ({len(clean)}):")
            for r in clean:
                lines.append(f"      - {r.task.task}  (stopped: {r.result.stop_reason})")
            lines.append(
                "          a clean route that still failed points at the corpus or the"
            )
            lines.append("          task file, not at the agent.")

        lines.append("")
        lines.append(
            f"    At {len(self.results)} tasks these are lists, not rates. Read the tasks;"
        )
        lines.append("    the percentage is a summary of four numbers you can see above.")
        return lines

    def _failure_mode_lines(self) -> list[str]:
        from rag_app.failure_modes import FAILURE_MODES, INFERRED_MODES

        ranked = self.failure_counts
        if not ranked:
            return []
        lines = ["", "  FAILURE MODES  (ranked by how many trajectories exhibit them)", ""]
        for mode, count in ranked:
            tag = " *" if mode in INFERRED_MODES else ""
            lines.append(f"    {count:>3}  {mode:<20}{tag:<3}{FAILURE_MODES[mode]}")
        lines.append("")
        lines.append(
            "    Rows OVERLAP — one trajectory can exhibit several modes — so they sum"
        )
        lines.append("    above the number of failing tasks. That is the finding, not a bug.")
        if any(mode in INFERRED_MODES for mode, _ in ranked):
            lines.append(
                "    * inferred from a proxy, and under-counts: a plausible search string"
            )
            lines.append(
                "      the model invented looks exactly like one it chose well."
            )
        return lines

    def describe(self) -> str:
        lines = [
            f"Agent trajectory evaluation "
            f"({len(self.answerable)} answerable + {len(self.unanswerable)} unanswerable)",
            "",
            f"  task success        {self.task_success:6.1%}   answer correct AND expected sources cited",
            f"  tool choice         {self.tool_choice_accuracy:6.1%}   order-insensitive; every expected tool used, none forbidden"
            f"   (of {len(self.results)}; {len(self.tool_choice_checked)} declare one)",
            f"  tool sequence       {self.tool_sequence_accuracy:6.1%}   order-sensitive; expected order present"
            f"   (of {len(self.sequence_checked)} that declare one)",
            f"  budget adherence    {self.budget_adherence:6.1%}   finished on its own terms, within the step target",
            f"  refusal accuracy    {self.refusal_accuracy:6.1%}   of {len(self.unanswerable)} unanswerable",
            f"  false refusals      {len(self.false_refusals):6d}   answerable tasks refused",
            f"  hallucinated cites  {self.hallucinated_citation_rate:6.1%}",
            f"  answered with no evidence {self.no_evidence_rate:5.1%}   the agent-only failure mode",
            "",
            f"  mean steps {self.mean_steps:.1f}   mean tool calls {self.mean_tool_calls:.1f}   "
            f"mean LLM calls {self.mean_llm_calls:.1f}",
            "",
            "  Stop reasons (which budget is actually binding):",
        ]
        for reason, count in self.stop_reason_counts.items():
            lines.append(f"    {reason or '(none)':<20} {count}")
        lines.extend(self._cost_lines())
        lines.extend(self._sequence_lines())
        lines.extend(self._gap_lines())
        lines.extend(self._failure_mode_lines())
        lines.append("")
        lines.append(
            "  Read success and efficiency together: a perfect trajectory can still"
        )
        lines.append(
            "  produce a wrong answer, and a messy one a right answer."
        )
        if self.false_refusals:
            lines.append("")
            lines.append("  Refused but answerable:")
            for r in self.false_refusals:
                lines.append(f"    - {r.task.task}  ({r.result.stop_reason})")
        return "\n".join(lines)


@dataclass(frozen=True)
class TaskGoldView:
    """An `AgentTask` in the shape `before_after.gold_fingerprint` reads.

    It lives here and not in `before_after.py`, which is documented as a leaf
    module over plain dicts and must not learn about its callers.

    The ROUTE expectations are packed into `expect_in_chunk` deliberately.
    Without them the digest would not change when `expect_tools` is edited, and
    two snapshots taken across a task-file edit would compare as if they had
    measured the same task set — which is the one thing the fingerprint exists
    to prevent. The sequence is joined into ONE string because
    `gold_fingerprint` sorts the list, and sorting would destroy its order.
    `note` is excluded: commentary, not ground truth.
    """

    question: str
    must_contain: list[str]
    expect_in_chunk: list[str]
    unanswerable: bool

    @property
    def answerable(self) -> bool:
        return not self.unanswerable


def agent_gold_view(tasks: list[AgentTask]) -> list[TaskGoldView]:
    views = []
    for t in tasks:
        route = [f"tool:{x}" for x in sorted(t.expect_tools)]
        route += [f"forbid:{x}" for x in sorted(t.forbid_tools)]
        route += [f"source:{x}" for x in sorted(t.expect_sources)]
        if t.expect_sequence:
            route.append("seq:" + ">".join(t.expect_sequence))
            route.append(f"match:{t.sequence_match}")
        route.append(f"max_steps:{t.max_steps}")
        route.append(f"min_steps:{t.min_steps}")
        views.append(
            TaskGoldView(
                question=t.task,
                must_contain=list(t.must_contain),
                expect_in_chunk=route,
                unanswerable=t.unanswerable,
            )
        )
    return views


def task_result_to_dict(r: TaskResult) -> dict[str, Any]:
    return {
        "task": r.task.task,
        "note": r.task.note,
        "answerable": r.task.answerable,
        "success": r.success,
        "answered": r.answered,
        "cited": r.cited,
        "stop_reason": r.result.stop_reason,
        "refused": r.result.refused,
        "failed": r.result.failed,
        "tools_used": r.tools_used,          # ordered, duplicates preserved
        "expect_tools": list(r.task.expect_tools),
        "forbid_tools": list(r.task.forbid_tools),
        "expect_sequence": list(r.task.expect_sequence),
        "sequence_match": r.task.sequence_match,
        "tool_choice_ok": r.tool_choice_ok,
        "sequence_ok": r.sequence_ok,
        "sequence_checked": r.sequence_checked,
        "budget_ok": r.budget_ok,
        "path_ok": r.path_ok,
        "right_answer_wrong_path": r.right_answer_wrong_path,
        "steps": len(r.result.steps),
        "llm_calls": r.result.llm_calls,
        "tool_calls": r.result.tool_calls,
        "prompt_chars": r.result.prompt_chars,
        "elapsed_s": round(r.result.elapsed_s, 3),
        "hallucinated_citations": list(r.result.hallucinated_citations),
        "failures": [
            {"mode": f.mode, "evidence": f.evidence, "proven": f.proven}
            for f in r.failures
        ],
    }


def agent_metrics(report: AgentEvalReport) -> dict[str, Any]:
    """SCALARS ONLY — this is what a snapshot consumes.

    `make_snapshot` does `dict(metrics)` with no validation, so feeding it the
    full report would push a nested list of task objects into
    `Snapshot.metrics`, and `Comparison.rows` would render it as a
    non-comparable blob. `agent_report_to_json` is built FROM this so the two
    cannot drift.

    Failure counts are flattened to one key per mode and emitted EVERY run even
    at zero: `Comparison.rows` unions key sets, so a mode that drops to zero
    would otherwise vanish into a missing row instead of showing `3 -> 0` — and
    those keys are the before-and-after number on the top failure.
    """
    from rag_app.failure_modes import FAILURE_MODES

    counts = dict(report.failure_counts)
    stats = report.cost_stats
    payload = {
        "n_tasks": len(report.results),
        "n_answerable": len(report.answerable),
        "n_sequence_checked": len(report.sequence_checked),
        "n_tool_choice_checked": len(report.tool_choice_checked),
        "generated": report.generated,
        "task_success": round(report.task_success, 4),
        "tool_choice_accuracy": round(report.tool_choice_accuracy, 4),
        "tool_sequence_accuracy": round(report.tool_sequence_accuracy, 4),
        "budget_adherence": round(report.budget_adherence, 4),
        "refusal_accuracy": round(report.refusal_accuracy, 4),
        "false_refusals": len(report.false_refusals),
        "stopped_on_budget": len(report.stopped_on_budget),
        "hallucinated_citation_rate": round(report.hallucinated_citation_rate, 4),
        "no_evidence_rate": round(report.no_evidence_rate, 4),
        "gap_rate": round(report.gap_rate, 4),
        "right_answer_wrong_path": len(report.outcome_trajectory_gap),
        "wrong_answer_right_path": len(report.clean_path_failures),
    }
    for mode in FAILURE_MODES:
        payload[f"failures_{mode.replace('-', '_')}"] = counts.get(mode, 0)
    for name, stat in stats.items():
        key = name.replace(" ", "_").replace("(s)", "s").strip("_")
        payload[f"mean_{key}"] = round(stat.mean, 2)
        payload[f"p99_{key}"] = round(stat.p99, 2)
    return payload


def agent_report_to_json(report: AgentEvalReport) -> str:
    return json.dumps(
        {
            **agent_metrics(report),
            "tasks": [task_result_to_dict(r) for r in report.results],
            "gap": [r.task.task for r in report.outcome_trajectory_gap],
            "clean_path_failures": [r.task.task for r in report.clean_path_failures],
            "right_tools_wrong_order": [
                r.task.task for r in report.right_tools_wrong_order
            ],
            "stop_reason_counts": report.stop_reason_counts,
        },
        indent=2,
    )


def evaluate_agent(
    tasks: list[AgentTask],
    cfg: AppConfig,
    *,
    tools,
    llm_fn=None,
    clock=None,
    memory=None,
    generated: bool = False,
    runner=None,
) -> AgentEvalReport:
    from rag_app.agent import run_agent

    run = runner or run_agent
    results = [
        TaskResult(
            task=task,
            # `forbid_tools` stops being a post-hoc metric here: the tool is
            # actually denied at call time. `tools_used` still records the
            # ATTEMPT (a denied call appends a Step), so tool_choice_ok keeps
            # its meaning and no existing number moves.
            result=run(
                task.task, cfg, tools=tools.restrict(task.forbid_tools),
                llm_fn=llm_fn, clock=clock, memory=memory,
            ),
        )
        for task in tasks
    ]
    return AgentEvalReport(results=results, generated=generated)
