"""Workflow vs Agent, measured on your own questions.

THE TOPIC'S WHOLE POINT, MADE RUNNABLE
---------------------------------------
"Workflow or agent?" is usually argued in the abstract. Here both arms exist
over the same corpus, the same retrieval stages and the same grounding rules, so
the question becomes arithmetic: same questions, same grader, count the calls.

THE EXPECTED VERDICT, WRITTEN DOWN BEFORE MEASURING
-----------------------------------------------------
On a small single-document corpus the agent should LOSE. It will spend several
LLM calls to answer what one search already answers, and the extra steps cannot
improve an answer that was already correct. It should win only where the
workflow's single pass is structurally unable to help: multi-hop questions, and
"which document says X".

That prediction is stated here so the harness can refute it. A comparison whose
expected result was never written down is a comparison that will agree with
whatever you hoped.

ON COST, HONESTLY
-----------------
Reported as LLM calls and prompt characters, not dollars. A real cost figure
needs a tokenizer and a per-model price table this project deliberately does not
carry, and a fabricated one would be worse than none. Calls and characters are
both exactly countable and both proportional to spend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from rag_app.config import AppConfig


@dataclass(frozen=True)
class ArmResult:
    arm: str
    question: str
    text: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    ok: bool | None = None       # graded against gold; None when ungraded
    llm_calls: int = 0
    tool_calls: int = 0
    steps: int = 0
    prompt_chars: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""


class _Counter:
    """Counts calls through an injected fn without instrumenting the callee.

    Same technique as the UI's stopwatch proxies: neither `ask()` nor
    `run_agent()` grows a line of measurement code.
    """

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0
        self.prompt_chars = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        first = args[0] if args else None
        if isinstance(first, list):  # agent: a message list
            self.prompt_chars += sum(len(m.get("content", "")) for m in first)
        elif isinstance(first, str):  # generation: a question
            self.prompt_chars += len(first)
        return self.fn(*args, **kwargs)


def _grade(gold, text: str, refused: bool, stop_reason: str = "") -> bool:
    """Reuses the gold set's own rule, so both arms are graded identically.

    `stop_reason` is the agent-arm fix for the same bug `TaskResult.success`
    had: `_stop()` marks every budget trip `refused=True`, so an agent that ran
    out of steps on an unanswerable question was credited with a correct
    refusal. The empty default means the workflow arm — whose gates are all
    decisions (`no-candidates`, `below-threshold`, `model-refused`) — is
    unaffected with no special case.
    """
    if not gold.answerable:
        from rag_app.agent import BUDGET_STOPS

        return refused and stop_reason not in BUDGET_STOPS
    low = (text or "").lower()
    return all(s.lower() in low for s in gold.must_contain)


@dataclass
class ComparisonReport:
    rows: list[ArmResult] = field(default_factory=list)
    arms: tuple[str, ...] = ()
    generated: bool = False

    def per_arm(self, arm: str) -> list[ArmResult]:
        return [r for r in self.rows if r.arm == arm]

    def _stat(self, arm: str, attr: str) -> float:
        values = [getattr(r, attr) for r in self.per_arm(arm)]
        return sum(values) if values else 0

    def describe(self) -> str:
        lines = [
            f"Workflow vs Agent over {len(self.per_arm(self.arms[0]))} questions"
            if self.arms
            else "Workflow vs Agent",
            "",
        ]
        if not self.generated:
            lines.append(
                "  DRY RUN - a scripted stand-in drove both arms, so these numbers are "
                "shape,\n  not behaviour. Pass --generate for the real thing."
            )
            lines.append("")

        header = f"  {'':<24}" + "".join(f"{a:>12}" for a in self.arms)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        def row(label, fn, fmt="{:>12}"):
            cells = "".join(fmt.format(fn(a)) for a in self.arms)
            lines.append(f"  {label:<24}" + cells)

        graded = [r for r in self.rows if r.ok is not None]
        if graded:
            row(
                "answered correctly",
                lambda a: f"{sum(1 for r in self.per_arm(a) if r.ok)}/"
                f"{len(self.per_arm(a))}",
            )
        row("LLM calls", lambda a: int(self._stat(a, "llm_calls")))
        row("tool calls", lambda a: int(self._stat(a, "tool_calls")))
        row("prompt chars", lambda a: f"{int(self._stat(a, 'prompt_chars')):,}")
        row(
            "median latency",
            lambda a: f"{median([r.elapsed_s for r in self.per_arm(a)] or [0]):.2f}s",
        )
        lines.append("")

        # The delta is the point, so state it rather than leaving arithmetic to
        # the reader.
        if len(self.arms) >= 2:
            base, other = self.arms[0], self.arms[1]
            base_calls = self._stat(base, "llm_calls") or 1
            factor = self._stat(other, "llm_calls") / base_calls
            lines.append(
                f"  {other} spent {factor:.1f}x the LLM calls of {base}."
            )
        lines.append(
            "  Cost is reported as calls and prompt characters. A dollar figure needs a"
        )
        lines.append(
            "  tokenizer and a price table this project does not carry."
        )
        return "\n".join(lines)


def comparison_to_json(report: ComparisonReport) -> str:
    return json.dumps(
        {
            "arms": list(report.arms),
            "generated": report.generated,
            "totals": {
                arm: {
                    "questions": len(report.per_arm(arm)),
                    "correct": sum(1 for r in report.per_arm(arm) if r.ok),
                    "llm_calls": int(report._stat(arm, "llm_calls")),
                    "tool_calls": int(report._stat(arm, "tool_calls")),
                    "prompt_chars": int(report._stat(arm, "prompt_chars")),
                }
                for arm in report.arms
            },
            "rows": [
                {
                    "arm": r.arm,
                    "question": r.question,
                    "ok": r.ok,
                    "llm_calls": r.llm_calls,
                    "stop_reason": r.stop_reason,
                }
                for r in report.rows
            ],
        },
        indent=2,
    )


DRY_ANSWER = "(dry run: no LLM was called)"


def compare_arms(
    gold,
    cfg: AppConfig,
    *,
    arms: tuple[str, ...] = ("workflow", "agent"),
    preset: str | None = None,
    embedder=None,
    reranker=None,
    store=None,
    tools=None,
    generate_fn=None,
    llm_fn=None,
    clock=None,
    generated: bool = False,
) -> ComparisonReport:
    """Run the same questions through `ask()` and through `run_agent()`."""
    from rag_app.agent import run_agent
    from rag_app.pipeline import ask

    rows: list[ArmResult] = []

    for question in gold:
        if "workflow" in arms:
            counter = _Counter(generate_fn or (lambda q, c, k: DRY_ANSWER))
            started = time.monotonic()
            answer = ask(
                question.question, preset=preset, config=cfg,
                embedder=embedder, reranker=reranker, store=store,
                generate_fn=counter,
            )
            rows.append(
                ArmResult(
                    arm="workflow",
                    question=question.question,
                    text=answer.text,
                    sources=answer.sources,
                    refused=answer.refused,
                    ok=_grade(question, answer.text, answer.refused),
                    llm_calls=counter.calls if answer.used_llm else 0,
                    prompt_chars=counter.prompt_chars,
                    steps=1,
                    elapsed_s=time.monotonic() - started,
                    stop_reason=answer.gate,
                )
            )

        for arm in [a for a in arms if a in ("agent", "langgraph")]:
            runner = run_agent
            if arm == "langgraph":
                from rag_app.agent_langgraph import run_agent_langgraph

                runner = run_agent_langgraph
            counter = _Counter(
                llm_fn or (lambda m, c: f"Thought: dry\nAction: final_answer\nAction Input: {DRY_ANSWER}")
            )
            started = time.monotonic()
            result = runner(
                question.question, cfg, tools=tools, llm_fn=counter, clock=clock
            )
            rows.append(
                ArmResult(
                    arm=arm,
                    question=question.question,
                    text=result.text,
                    sources=result.sources,
                    refused=result.refused,
                    ok=_grade(question, result.text, result.refused, result.stop_reason),
                    llm_calls=result.llm_calls,
                    tool_calls=result.tool_calls,
                    steps=len(result.steps),
                    prompt_chars=result.prompt_chars,
                    elapsed_s=time.monotonic() - started,
                    stop_reason=result.stop_reason,
                )
            )

    return ComparisonReport(rows=rows, arms=arms, generated=generated)
