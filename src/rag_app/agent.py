"""The ReAct loop: a second control flow over the same retrieval and the same gate.

WORKFLOW VS AGENT, CONCRETELY
------------------------------
`pipeline.ask()` is the workflow arm: retrieve, rerank, gate, generate, once.
Fixed shape, one LLM call, fully predictable cost. This module is the agent arm:
the model decides what to look up, in what order, and when it has enough. Same
tools underneath, same grounding rules, different control flow — which is what
makes `compare.py` a fair comparison rather than two unrelated systems.

The honest expectation, written down before measuring: on a small single-PDF
corpus the agent will usually LOSE. More calls, more latency, and no better
answer on a question one search already answers. It should win only on multi-hop
questions and on "which document says X". `compare` exists to check that on your
corpus, not to flatter the agent.

THE GATE IS RELOCATED, NOT DROPPED
-----------------------------------
`pipeline.ask()` enforces four guarantees. An agent that answers from tool
output has to enforce the same ones somewhere else, plus one that only the agent
shape can violate:

  1. score gate      -> moved INTO search_documents (tools.py). Below threshold
                        the tool returns no excerpt at all, so the model cannot
                        even see the rejected text. Stricter than ask().
  2. citation rules  -> generate.CITATION_RULES, shared verbatim, not paraphrased.
  3. refusal strips  -> applied to the final answer here, exactly as ask() does.
     sources
  4. invented        -> checked against the union of evidence from EVERY step,
     citations          not just the last one.
  5. NO-EVIDENCE     -> new. A final answer produced without a single excerpt
     GATE               ever being returned is forced to DONT_KNOW. An agent
                        answering from its own weights is the failure only this
                        shape produces, and nothing in ask() ever needed to
                        catch it.

Memory text is deliberately NOT citable: `cited_sources()` only accepts labels
present in `evidence`, so a citation sourced from a recalled turn is reported as
hallucinated rather than credited.

WHY STATELESS RE-PROMPTING
---------------------------
The prompt is rebuilt from scratch each iteration rather than appended to a
growing message list. That is the classic ReAct formulation, it means the exact
prompt string is printable at every step (the same value `explain.py` protects
for retrieval), and it makes summarization "compress a string" instead of
surgery on a message list.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_app.config import AppConfig
from rag_app.generate import (
    CITATION_RULES,
    DONT_KNOW,
    cited_sources,
    is_refusal,
    normalize,
    sources_from_contexts,
)
from rag_app.llm import chat_messages
from rag_app.store import ScoredChunk
from rag_app.tools import FINAL_ANSWER, ToolRegistry

SYSTEM = (
    "You answer questions about a document collection by using tools.\n\n"
    "Work in a loop. Each turn, emit exactly this, and nothing else:\n\n"
    "Thought: <what you know so far and what you need next>\n"
    "Action: <one tool name, or final_answer>\n"
    "Action Input: <the input for that action>\n\n"
    "Then stop and wait. An Observation will be given to you. Never write an "
    "Observation yourself.\n\n"
    "When you have enough evidence, use Action: final_answer, and put the complete "
    "answer in Action Input.\n\n"
    "ANSWER ONLY FROM WHAT THE TOOLS RETURNED. If the tools returned nothing "
    "relevant, your final answer must be exactly this and nothing else: "
    f"{DONT_KNOW}\n\n"
    f"{CITATION_RULES}\n\n"
    "Recalled context from earlier in the conversation is NOT a document source and "
    "must never be cited."
)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentAction:
    thought: str = ""
    tool: str = ""
    tool_input: str = ""
    is_final: bool = False
    parse_error: str = ""


@dataclass(frozen=True)
class Step:
    index: int
    thought: str
    tool: str
    tool_input: str
    observation: str
    ok: bool
    raw: str = ""
    prompt_chars: int = 0


@dataclass
class AgentResult:
    """The agent's `Answer`. `stop_reason` is its `gate`."""

    question: str
    text: str = ""
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""
    failed: bool = False
    refused: bool = False
    used_llm: bool = False
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0
    prompt_chars: int = 0
    hallucinated_citations: list[str] = field(default_factory=list)
    evidence: list[ScoredChunk] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"Answer: {self.text}",
            f"Sources: {', '.join(self.sources) if self.sources else '(none)'}",
            (
                f"({len(self.steps)} steps, {self.tool_calls} tool calls, "
                f"{self.llm_calls} LLM calls, {self.elapsed_s:.1f}s, "
                f"{self.prompt_chars} prompt chars; stop={self.stop_reason})"
            ),
        ]
        if self.hallucinated_citations:
            lines.append(
                f"!! HALLUCINATED CITATIONS (not in any tool output): "
                f"{', '.join(self.hallucinated_citations)}"
            )
        if self.failed:
            lines.append(f"!! {self.meta.get('stop_detail', self.stop_reason)}")
        lines.append("")
        lines.append("Trajectory:")
        for s in self.steps:
            lines.append(f"  {s.index}. Thought: {s.thought}")
            lines.append(f"     Action: {s.tool}  Input: {s.tool_input}")
            observation = s.observation.replace("\n", " ")[:160]
            lines.append(f"     Observation: {observation}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_KEYS = ("thought", "action input", "action", "observation")


def _strip_marker(line: str) -> str:
    line = line.strip()
    for prefix in ("- ", "* ", "> "):
        if line.startswith(prefix):
            line = line[len(prefix):]
    return line.replace("**", "").replace("__", "")


def parse_action(raw: str) -> AgentAction:
    """Read one model turn. Tolerant of formatting, strict about structure.

    Tolerated: any case, markdown bold, list markers, code fences, a multi-line
    Action Input, and a JSON object as the input.

    NOT tolerated, on purpose:

      * a reply with no `Action:` line is a parse failure, never an implicit
        final answer. Treating loose prose as an answer is precisely how
        ungrounded text gets past the gate.

      * anything after a model-written `Observation:` is discarded. Models
        pre-fill fake observations, and keeping them would let the model invent
        its own tool results and then reason over them.
    """
    text = (raw or "").strip()
    if not text:
        return AgentAction(parse_error="the model returned nothing")

    lines = [ln for ln in text.splitlines() if ln.strip() != "```"]
    lines = [ln for ln in lines if not ln.strip().startswith("```")]

    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        stripped = _strip_marker(line)
        lowered = stripped.lower()
        matched = None
        for key in _KEYS:
            if lowered.startswith(key + ":"):
                matched = key
                break
        if matched == "observation":
            break  # a hallucinated observation: everything after it is fiction
        if matched:
            current = matched
            fields.setdefault(current, []).append(stripped[len(matched) + 1 :].strip())
        elif current is not None:
            fields[current].append(line.rstrip())

    thought = "\n".join(fields.get("thought", [])).strip()
    action = "\n".join(fields.get("action", [])).strip().strip("`").strip()
    action_input = "\n".join(fields.get("action input", [])).strip()

    if not action:
        return AgentAction(
            thought=thought,
            parse_error="no 'Action:' line",
        )

    action = action.split()[0].strip(".,:").lower() if action.split() else ""

    # A JSON object input for a single-string tool is flattened when
    # unambiguous. Anything else is left as text for the tool to reject.
    if action_input.startswith("{") and action_input.endswith("}"):
        import json

        try:
            payload = json.loads(action_input)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict) and len(payload) == 1:
                action_input = str(next(iter(payload.values())))

    return AgentAction(
        thought=thought,
        tool=action,
        tool_input=action_input,
        is_final=action == FINAL_ANSWER,
    )


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class Budget:
    """Every limit the loop can hit, and the reason string for each.

    `clock` is injected so a wall-clock budget is testable without sleeping —
    the suite is three seconds and must stay that way.
    """

    def __init__(self, cfg: AppConfig, clock=None):
        self.cfg = cfg.agent
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.steps = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.parse_failures = 0

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def check(self, next_prompt_chars: int | None = None) -> tuple[str, str] | None:
        """Return (stop_reason, detail) if a budget is spent, else None."""
        c = self.cfg
        if self.steps >= c.max_steps:
            return (
                "max-steps",
                f"stopped after {self.steps} steps - agent.max_steps={c.max_steps} "
                f"reached without a final answer",
            )
        if self.elapsed >= c.wall_clock_seconds:
            return (
                "wall-clock",
                f"stopped after {self.elapsed:.1f}s - agent.wall_clock_seconds="
                f"{c.wall_clock_seconds}",
            )
        if self.llm_calls >= c.max_llm_calls:
            return (
                "max-llm-calls",
                f"stopped after {self.llm_calls} LLM calls - agent.max_llm_calls="
                f"{c.max_llm_calls}",
            )
        if next_prompt_chars is not None and next_prompt_chars > c.max_prompt_chars:
            return (
                "token-budget",
                f"the next prompt would be {next_prompt_chars} chars, over "
                f"agent.max_prompt_chars={c.max_prompt_chars}",
            )
        return None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _call(messages: list[dict[str, str]], cfg: AppConfig) -> str:
    return chat_messages(messages, cfg.llm, cfg)


def build_prompt(
    question: str,
    tools: ToolRegistry,
    steps: list[Step],
    memory_block: str = "",
) -> list[dict[str, str]]:
    """Rebuild the whole user message from scratch. See the module docstring."""
    parts = [f"TOOLS\n{tools.describe_for_prompt()}", f"\nQUESTION\n{question}"]
    if memory_block:
        parts.append(
            "\nRECALLED CONTEXT - NOT a document source; never cite it\n" + memory_block
        )
    if steps:
        scratch = []
        for s in steps:
            scratch.append(f"Thought: {s.thought}")
            scratch.append(f"Action: {s.tool}")
            scratch.append(f"Action Input: {s.tool_input}")
            scratch.append(f"Observation: {s.observation}")
        parts.append("\nWORK SO FAR\n" + "\n".join(scratch))
    parts.append("\nContinue. Emit Thought / Action / Action Input.")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def finalize(
    question: str,
    text: str,
    evidence: list[ScoredChunk],
    result: AgentResult,
) -> AgentResult:
    """The gate. Guarantees 3, 4 and 5 live here."""
    if not evidence:
        # Guarantee 5. An agent that answered without ever retrieving anything
        # answered from its own weights, which is exactly what a RAG gate is for.
        result.text = DONT_KNOW
        result.sources = []
        result.stop_reason = "no-evidence"
        result.refused = True
        result.failed = True
        result.meta["stop_detail"] = (
            "the agent produced a final answer without any tool ever returning an "
            "excerpt, so the answer came from the model rather than the documents"
        )
        return result

    if is_refusal(text):
        # Guarantee 3: a refusal carries no sources, or the documents get credit
        # for a non-answer.
        result.text = text
        result.sources = []
        result.stop_reason = "model-refused"
        result.refused = True
        return result

    # Guarantee 4, over the union of evidence from every step.
    grounded, invented = cited_sources(text, evidence)
    result.text = text
    result.sources = grounded or sources_from_contexts(evidence)
    result.hallucinated_citations = invented
    result.meta["cited"] = bool(grounded)
    result.stop_reason = "final-answer"
    return result


def _stop(result: AgentResult, reason: str, detail: str) -> AgentResult:
    """Every budget trip: DONT_KNOW, no sources, failed, and the number named.

    The trajectory is preserved in `steps`; it just does not become an answer.
    Attaching partial prose to a truncated run would be the exact dishonesty the
    failed=True + describe() precedent exists to prevent.
    """
    result.text = DONT_KNOW
    result.sources = []
    result.stop_reason = reason
    result.failed = True
    result.refused = True
    result.meta["stop_detail"] = detail
    return result


def run_agent(
    question: str,
    cfg: AppConfig,
    *,
    tools: ToolRegistry,
    llm_fn=None,
    memory=None,
    clock=None,
    max_steps: int | None = None,
) -> AgentResult:
    """Run the ReAct loop. Never raises: every failure becomes a stop_reason."""
    if max_steps is not None:
        from dataclasses import replace

        cfg = replace(cfg, agent=replace(cfg.agent, max_steps=max_steps))

    caller = llm_fn or _call
    budget = Budget(cfg, clock=clock)
    result = AgentResult(question=question)
    steps: list[Step] = []
    evidence: list[ScoredChunk] = []
    seen_actions: dict[tuple[str, str], int] = {}
    memory_block = memory.recall(question) if memory is not None else ""

    while True:
        trip = budget.check()
        if trip:
            result.steps = steps
            result.evidence = evidence
            return _stop(result, *trip)

        messages = build_prompt(question, tools, steps, memory_block)
        prompt_chars = sum(len(m["content"]) for m in messages)
        # Checked BEFORE the call, so an over-budget prompt costs nothing.
        trip = budget.check(next_prompt_chars=prompt_chars)
        if trip:
            result.steps = steps
            result.evidence = evidence
            return _stop(result, *trip)

        try:
            raw = caller(messages, cfg)
        except Exception as exc:
            result.steps = steps
            result.evidence = evidence
            result.llm_calls = budget.llm_calls
            return _stop(
                result, "llm-error", f"the LLM call failed: {type(exc).__name__}: {exc}"
            )

        budget.llm_calls += 1
        budget.steps += 1
        result.used_llm = True
        result.prompt_chars += prompt_chars

        action = parse_action(raw)

        if action.parse_error:
            budget.parse_failures += 1
            observation = (
                f"Malformed output: {action.parse_error}. Reply with exactly three "
                f"lines: Thought:, Action:, Action Input:. Available actions: "
                f"{', '.join(tools.names())}, {FINAL_ANSWER}."
            )
            steps.append(
                Step(len(steps) + 1, action.thought, "", "", observation, False,
                     raw=raw, prompt_chars=prompt_chars)
            )
            if budget.parse_failures > cfg.agent.max_parse_failures:
                result.steps = steps
                result.evidence = evidence
                result.llm_calls = budget.llm_calls
                result.tool_calls = budget.tool_calls
                result.elapsed_s = budget.elapsed
                return _stop(
                    result,
                    "unparseable",
                    f"the model failed to produce a parseable action "
                    f"{budget.parse_failures} times",
                )
            continue

        if action.is_final:
            result.steps = steps
            result.evidence = evidence
            result.llm_calls = budget.llm_calls
            result.tool_calls = budget.tool_calls
            result.elapsed_s = budget.elapsed
            return finalize(question, action.tool_input, evidence, result)

        tool = tools.get(action.tool)
        if tool is None:
            observation = (
                f"Unknown tool {action.tool!r}. Available: "
                f"{', '.join(tools.names())}, {FINAL_ANSWER}."
            )
            steps.append(
                Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                     observation, False, raw=raw, prompt_chars=prompt_chars)
            )
            continue

        key = (action.tool, normalize(action.tool_input))
        seen_actions[key] = seen_actions.get(key, 0) + 1
        if seen_actions[key] > cfg.agent.repeat_action_limit:
            result.steps = steps
            result.evidence = evidence
            result.llm_calls = budget.llm_calls
            result.tool_calls = budget.tool_calls
            result.elapsed_s = budget.elapsed
            return _stop(
                result,
                "repeated-action",
                f"the model called {action.tool} with the same input "
                f"{seen_actions[key]} times and is not making progress",
            )
        if seen_actions[key] > 1:
            # Warn once before stopping: models usually recover from being told.
            observation = (
                "You already ran this exact call; its result will not change. Choose a "
                "different tool or input, or give your final answer."
            )
            steps.append(
                Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                     observation, False, raw=raw, prompt_chars=prompt_chars)
            )
            continue

        if budget.tool_calls >= cfg.agent.max_tool_calls:
            result.steps = steps
            result.evidence = evidence
            result.llm_calls = budget.llm_calls
            result.tool_calls = budget.tool_calls
            result.elapsed_s = budget.elapsed
            return _stop(
                result,
                "max-tool-calls",
                f"stopped after {budget.tool_calls} tool calls - "
                f"agent.max_tool_calls={cfg.agent.max_tool_calls}",
            )

        observation = tool.run(action.tool_input)
        budget.tool_calls += 1
        evidence.extend(getattr(tool, "last_evidence", []) or [])
        # Tools return text; the evidence they contributed is recovered from the
        # store by matching the labels they printed. See _evidence_from.
        evidence.extend(_evidence_from(observation, tools))
        steps.append(
            Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                 observation, True, raw=raw, prompt_chars=prompt_chars)
        )


def _evidence_from(observation: str, tools: ToolRegistry) -> list[ScoredChunk]:
    """Recover citable chunks from a tool observation.

    Tools return strings because that is all a model can read. The gate needs
    objects, so the excerpt blocks are parsed back out. Only text a tool actually
    printed becomes evidence — which is what makes `cited_sources()` able to call
    a memory-sourced or invented citation hallucinated.
    """
    from rag_app.chunking import Chunk

    out: list[ScoredChunk] = []
    blocks = observation.split("\n\n")
    for block in blocks:
        lines = block.split("\n", 1)
        head = lines[0].strip()
        if len(lines) == 2 and head.startswith("[") and head.endswith("]"):
            label = head[1:-1]
            out.append(
                ScoredChunk(Chunk(f"tool::{len(out)}", label, lines[1], {}), 0.0)
            )
    return out
