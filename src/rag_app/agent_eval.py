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

    @property
    def answerable(self) -> bool:
        return not self.unanswerable


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
        if not unanswerable and not (must or tools):
            raise ValueError(
                f"Agent task {i} ({text!r}) is answerable but names nothing to check. "
                f"Add 'must_contain' or 'expect_tools', or mark it 'unanswerable: true'."
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
            return self.result.refused
        return self.answered and self.cited

    @property
    def tool_choice_ok(self) -> bool:
        used = set(self.tools_used)
        if any(t not in used for t in self.task.expect_tools):
            return False
        return not any(t in used for t in self.task.forbid_tools)

    @property
    def budget_ok(self) -> bool:
        """Finished on its own terms, and within the task's own step target."""
        if self.result.stop_reason != "final-answer":
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
        return self._frac([r.result.refused for r in self.unanswerable])

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

    @property
    def stop_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.result.stop_reason] = counts.get(r.result.stop_reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def describe(self) -> str:
        lines = [
            f"Agent trajectory evaluation "
            f"({len(self.answerable)} answerable + {len(self.unanswerable)} unanswerable)",
            "",
            f"  task success        {self.task_success:6.1%}   answer correct AND expected sources cited",
            f"  tool choice         {self.tool_choice_accuracy:6.1%}   used every expected tool, no forbidden one",
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


def agent_report_to_json(report: AgentEvalReport) -> str:
    return json.dumps(
        {
            "task_success": round(report.task_success, 4),
            "tool_choice_accuracy": round(report.tool_choice_accuracy, 4),
            "budget_adherence": round(report.budget_adherence, 4),
            "refusal_accuracy": round(report.refusal_accuracy, 4),
            "false_refusals": len(report.false_refusals),
            "hallucinated_citation_rate": round(report.hallucinated_citation_rate, 4),
            "no_evidence_rate": round(report.no_evidence_rate, 4),
            "mean_steps": round(report.mean_steps, 2),
            "mean_tool_calls": round(report.mean_tool_calls, 2),
            "mean_llm_calls": round(report.mean_llm_calls, 2),
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
            result=run(task.task, cfg, tools=tools, llm_fn=llm_fn, clock=clock, memory=memory),
        )
        for task in tasks
    ]
    return AgentEvalReport(results=results, generated=generated)
