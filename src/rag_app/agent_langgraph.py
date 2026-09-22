"""The same agent, built on LangGraph. A labelled exhibit, not the backbone.

WHAT THIS IS FOR
----------------
"Should we use a framework?" is normally settled by assertion. Here both
implementations exist over the same tools, the same prompt and the same
grounding gate, so the only difference is the machinery — which makes the cost
of the framework measurable rather than argued.

To keep that honest, this module reuses `tools.py`, `agent.parse_action`,
`agent.build_prompt` and `agent.finalize` UNCHANGED, and returns the same
`AgentResult`. A LangGraph arm with its own tools and its own prompt would be
measuring two different agents and calling the difference "the framework".
`test_agent_langgraph.py` pins the equivalence: identical scripted replies must
produce an identical tool sequence, final text and stop reason.

WHAT LANGGRAPH ACTUALLY BUYS
-----------------------------
  * a graph you can print (`graph.get_graph().draw_ascii()`) — a real artifact
    the plain `while` loop has no equivalent of
  * checkpointing, so state survives between invocations
  * interrupt/resume, and human-in-the-loop approval before a tool runs
  * streaming of intermediate state
  * declarative state merging through reducers

WHAT IT COSTS
-------------
  * a large transitive dependency tree, into a venv this project documents as
    hand-repaired
  * a message abstraction between you and the exact prompt string, which
    directly conflicts with this repo's habit of showing the prepared text
  * termination logic scattered into a router instead of a readable loop

THE VERDICT FOR THIS APP
------------------------
The plain loop wins. It is a few hundred lines you can step through in a
debugger, and none of what LangGraph adds — persistence, resumption, parallel
branches, approval gates — is something this app needs today. That verdict is
written down so the comparison can overturn it rather than confirm it.

Imports are inside function bodies so this module is importable, and the CLI can
offer `--impl langgraph` and reject it politely, on a machine where the extra was
never installed.
"""

from __future__ import annotations

from typing import Any, TypedDict

from rag_app.agent import (
    AgentResult,
    Budget,
    Step,
    build_prompt,
    finalize,
    _call,
    _evidence_from,
    _stop,
)
from rag_app.config import AppConfig
from rag_app.generate import normalize
from rag_app.tools import FINAL_ANSWER, ToolRegistry

EXTRA_HINT = 'LangGraph is not installed. Run: pip install -e ".[agents]"'


class AgentState(TypedDict, total=False):
    """The graph's state. Declared at MODULE scope on purpose.

    LangGraph resolves each node's annotations with `get_type_hints` against
    module globals, and `from __future__ import annotations` turns them into
    strings — so a TypedDict defined inside `build_graph` raises NameError at
    graph-build time. `TypedDict` comes from `typing`, so this costs no
    framework import.
    """

    question: str
    steps: list
    evidence: list
    stop_reason: str
    stop_detail: str
    text: str
    llm_calls: int
    tool_calls: int
    prompt_chars: int
    parse_failures: int
    seen: dict
    # Carried between the two nodes. They must be DECLARED here: LangGraph
    # filters any key the schema does not name, so a node cannot simply hand a
    # local to the next one the way the plain loop does. Part of the framework's
    # cost, and worth seeing rather than smoothing over.
    action: Any
    raw: str


def _require():
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(EXTRA_HINT) from exc


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("langgraph") is not None


def build_graph(cfg: AppConfig, tools: ToolRegistry, *, llm_fn=None, clock=None):
    """A two-node StateGraph: think -> act -> think, with budgets in the router."""
    _require()
    from langgraph.graph import END, StateGraph

    caller = llm_fn or _call
    budget = Budget(cfg, clock=clock)

    def think(state: AgentState) -> AgentState:
        from rag_app.agent import parse_action

        messages = build_prompt(state["question"], tools, state["steps"])
        prompt_chars = sum(len(m["content"]) for m in messages)
        trip = budget.check(next_prompt_chars=prompt_chars)
        if trip:
            return {**state, "stop_reason": trip[0], "stop_detail": trip[1]}

        try:
            raw = caller(messages, cfg)
        except Exception as exc:
            return {
                **state,
                "stop_reason": "llm-error",
                "stop_detail": f"the LLM call failed: {type(exc).__name__}: {exc}",
            }

        budget.llm_calls += 1
        budget.steps += 1
        action = parse_action(raw)
        return {
            **state,
            "llm_calls": budget.llm_calls,
            "prompt_chars": state.get("prompt_chars", 0) + prompt_chars,
            "action": action,
            "raw": raw,
        }

    def act(state: AgentState) -> AgentState:
        action = state["action"]
        steps = list(state["steps"])
        evidence = list(state["evidence"])
        seen = dict(state.get("seen") or {})

        if action.parse_error:
            failures = state.get("parse_failures", 0) + 1
            observation = (
                f"Malformed output: {action.parse_error}. Reply with exactly three "
                f"lines: Thought:, Action:, Action Input:."
            )
            steps.append(Step(len(steps) + 1, action.thought, "", "", observation, False))
            if failures > cfg.agent.max_parse_failures:
                return {
                    **state, "steps": steps, "parse_failures": failures,
                    "stop_reason": "unparseable",
                    "stop_detail": f"the model failed to produce a parseable action "
                                   f"{failures} times",
                }
            return {**state, "steps": steps, "parse_failures": failures}

        tool = tools.get(action.tool)
        if tool is None:
            observation = (
                f"Unknown tool {action.tool!r}. Available: "
                f"{', '.join(tools.names())}, {FINAL_ANSWER}."
            )
            steps.append(
                Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                     observation, False)
            )
            return {**state, "steps": steps}

        key = f"{action.tool}::{normalize(action.tool_input)}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > cfg.agent.repeat_action_limit:
            return {
                **state, "steps": steps, "seen": seen,
                "stop_reason": "repeated-action",
                "stop_detail": f"the model called {action.tool} with the same input "
                               f"{seen[key]} times and is not making progress",
            }
        if seen[key] > 1:
            observation = (
                "You already ran this exact call; its result will not change. Choose a "
                "different tool or input, or give your final answer."
            )
            steps.append(
                Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                     observation, False)
            )
            return {**state, "steps": steps, "seen": seen}

        if budget.tool_calls >= cfg.agent.max_tool_calls:
            return {
                **state, "steps": steps, "seen": seen,
                "stop_reason": "max-tool-calls",
                "stop_detail": f"stopped after {budget.tool_calls} tool calls",
            }

        observation = tool.run(action.tool_input)
        budget.tool_calls += 1
        got, _planted = _evidence_from(observation, tools)
        evidence.extend(got)
        steps.append(
            Step(len(steps) + 1, action.thought, action.tool, action.tool_input,
                 observation, True)
        )
        return {
            **state, "steps": steps, "evidence": evidence, "seen": seen,
            "tool_calls": budget.tool_calls,
        }

    def route(state: AgentState) -> str:
        """Termination lives here rather than in a readable `while`. This is
        precisely the readability cost the module docstring names.

        The router deliberately does NOT check the budget itself. `think`
        already checks it at its top, exactly where the plain loop checks at the
        top of each iteration — and an extra check here changed BEHAVIOUR, not
        just bookkeeping: it ended the run between a successful `think` and its
        `act`, so a budget-stopped graph run performed one fewer tool call than
        the identical plain run on the identical script.

        That was invisible while the parity test only covered the happy path.
        The equivalence this module claims ("same tools, same prompt, same gate,
        only the control flow differs") is worth exactly as much as the tests
        that pin it, so the budget is checked in one place per arm and
        `test_both_implementations_agree_on_cost_after_a_budget_stop` holds it
        there.
        """
        if state.get("stop_reason"):
            return END
        action = state.get("action")
        if action is not None and action.is_final:
            return END
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.set_entry_point("think")
    graph.add_conditional_edges("think", route, {"act": "act", END: END})
    graph.add_edge("act", "think")
    return graph.compile(), budget


def run_agent_langgraph(
    question: str,
    cfg: AppConfig,
    *,
    tools: ToolRegistry,
    llm_fn=None,
    memory=None,
    clock=None,
    max_steps: int | None = None,
) -> AgentResult:
    """Same signature and same AgentResult as `agent.run_agent`."""
    if max_steps is not None:
        from dataclasses import replace

        cfg = replace(cfg, agent=replace(cfg.agent, max_steps=max_steps))

    _require()
    graph, budget = build_graph(cfg, tools, llm_fn=llm_fn, clock=clock)
    state: dict[str, Any] = {
        "question": question, "steps": [], "evidence": [],
        "stop_reason": "", "text": "", "llm_calls": 0, "tool_calls": 0,
        "prompt_chars": 0, "parse_failures": 0, "seen": {},
    }
    final = graph.invoke(state, {"recursion_limit": cfg.agent.max_steps * 4 + 10})

    result = AgentResult(question=question)
    result.steps = final.get("steps", [])
    result.evidence = final.get("evidence", [])
    result.llm_calls = final.get("llm_calls", 0)
    result.tool_calls = final.get("tool_calls", 0)
    result.prompt_chars = final.get("prompt_chars", 0)
    result.used_llm = result.llm_calls > 0
    result.elapsed_s = budget.elapsed
    result.meta["implementation"] = "langgraph"

    if final.get("stop_reason"):
        return _stop(result, final["stop_reason"], final.get("stop_detail", ""))

    action = final.get("action")
    if action is not None and action.is_final:
        # The SAME gate as the plain loop, not a reimplementation.
        return finalize(question, action.tool_input, result.evidence, result)
    return _stop(result, "max-steps", "the graph ended without a final answer")


def draw(cfg: AppConfig, tools: ToolRegistry) -> str:
    """The one thing the plain loop genuinely cannot do: print itself.

    Mermaid rather than ASCII: `draw_ascii()` needs `grandalf`, and adding a
    dependency to render a demonstration of a dependency's cost would be a
    poor trade.
    """
    graph, _ = build_graph(cfg, tools)
    return graph.get_graph().draw_mermaid()
