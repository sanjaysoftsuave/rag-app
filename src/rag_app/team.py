"""Week 10: a manager plus two specialists, racing the Week 7 solo agent.

THE PREDICTION, WRITTEN DOWN BEFORE MEASURING
-----------------------------------------------
Same discipline `compare.py` and `agent.py` already use. On this corpus (one
PDF, mostly single-hop questions) the team should LOSE on cost and tokens: the
manager's synthesis step re-sends both specialists' full reports, which is
pure overhead when nothing needed to be split. It should tie or lose slightly
on quality, since most questions here do not decompose into genuinely
independent sub-tasks. It should only plausibly win where a question has two
independently-checkable parts. `atasks --impl team` vs `--impl plain` plus
`compare` is what checks this against reality rather than against the
prediction.

REUSE, NOT REINVENTION
------------------------
A "specialist" is NOT a new agent implementation. It is `agent.run_agent`,
narrowed by TOOL VISIBILITY (`ToolRegistry.restrict`, Week 8's own enforcement
seam) and framed by appending the manager's focus instruction to the question
text. `agent.build_prompt` and `agent.SYSTEM` are reused completely
unmodified — there is no second ReAct loop here.

The manager's final synthesis answer is gated by `agent.finalize` — the SAME
function the solo agent uses, over the UNION of both specialists' evidence.
No-evidence gate, forged-citation gate, refusal stripping, hallucinated-
citation checking: all reused, none reimplemented. `SYNTHESIS_SYSTEM` embeds
`generate.CITATION_RULES`/`generate.DATA_BOUNDARY` VERBATIM, exactly as
`agent.SYSTEM` does — two independently-worded citation contracts is how the
`[CS-1001]` failure happened the first time.

`run_team` returns a plain `AgentResult` — the same dataclass `run_agent` and
`run_agent_langgraph` return — so it is a drop-in `runner=` for
`agent_eval.evaluate_agent` and `compare.compare_arms` with no new comparison
machinery anywhere.

WHAT THIS DELIBERATELY DOES NOT BUILD
----------------------------------------
* Parallel specialists. Both run sequentially, sharing one `clock`, for the
  same reason the whole test suite is deterministic: a `scripted_llm` reply
  list has no well-defined order across two threads. This makes the team's
  own SPEED number a worst case — the syllabus's "when does multi-agent
  help" answer is partly "when sub-tasks are independent and can run in
  parallel", which this implementation does not exploit.
* Per-role models. Manager and both specialists all call `cfg.llm`. Varying
  models per role is a second experimental variable that would muddy "team
  structure vs. single agent" into "team structure + cheaper specialists vs.
  single agent".
* A team-level hard budget. Each specialist independently enforces
  `cfg.agent.team.specialist_max_steps`/`specialist_max_tool_calls` via the
  same `dataclasses.replace` idiom `run_agent` already uses for `max_steps`;
  there is no additional ceiling on the team as a whole. The race MEASURES
  total cost; it does not cap it further this week.
"""

from __future__ import annotations

import time
from dataclasses import replace

from rag_app.agent import AgentResult, Step, finalize
from rag_app.config import AppConfig
from rag_app.generate import CITATION_RULES, DATA_BOUNDARY, DONT_KNOW
from rag_app.llm import Usage, chat_messages, extract_json, usage_summary
from rag_app.store import ScoredChunk
from rag_app.tools import ToolRegistry

PLAN_SYSTEM = (
    "You manage a two-specialist team answering questions about a document "
    "collection.\n\n"
    "SPECIALIST-SEARCH can search the documents by meaning and by exact term. "
    "SPECIALIST-VERIFY can list every document label and read one whole "
    "document end to end, to confirm a specific detail.\n\n"
    "Given the question, write ONE focused instruction for each specialist - "
    "what it should look into, not how. Reply with exactly this JSON and "
    "nothing else:\n"
    '{"search_focus": "...", "verify_focus": "..."}'
)

SYNTHESIS_SYSTEM = (
    "You manage a two-specialist team. Each specialist investigated part of a "
    "question and reported back. Write ONE final answer from their reports.\n\n"
    "ANSWER ONLY FROM WHAT THE SPECIALISTS REPORTED. If neither specialist "
    "found anything relevant, your final answer must be exactly this and "
    f"nothing else: {DONT_KNOW}\n\n"
    f"{CITATION_RULES}\n\n"
    f"{DATA_BOUNDARY}"
)


def _default_caller(cfg: AppConfig, usage: Usage | None):
    return lambda messages, c: chat_messages(messages, c.llm, c, usage=usage)


def plan_specialists(
    question: str, cfg: AppConfig, *, llm_fn=None, usage: Usage | None = None
) -> tuple[str, str, Step]:
    """The manager's first call. Never raises an unparseable reply into a
    crash: a malformed plan degrades to handing both specialists the raw
    question, which is still a correct (if less focused) team.

    Returns (search_focus, verify_focus, step) — `step` records what the
    manager actually said, so the team's trajectory shows the planning
    reasoning the same way `agent.describe()` shows every ReAct thought.
    """
    caller = llm_fn or _default_caller(cfg, usage)
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": f"QUESTION\n{question}"},
    ]
    prompt_chars = sum(len(m["content"]) for m in messages)
    raw = caller(messages, cfg)

    parsed = extract_json(raw)
    if isinstance(parsed, dict) and parsed.get("search_focus") and parsed.get("verify_focus"):
        search_focus = str(parsed["search_focus"])
        verify_focus = str(parsed["verify_focus"])
        thought = "planned two focused sub-tasks"
    else:
        search_focus = verify_focus = question
        thought = "could not parse a plan; both specialists get the raw question"

    step = Step(
        index=1,
        thought=thought,
        tool="(manager: plan)",
        tool_input=question,
        observation=raw,
        ok=True,
        raw=raw,
        prompt_chars=prompt_chars,
    )
    return search_focus, verify_focus, step


def run_specialist(
    label: str,
    focus: str,
    question: str,
    cfg: AppConfig,
    *,
    tools: ToolRegistry,
    allowed: tuple[str, ...],
    llm_fn=None,
    clock=None,
    usage: Usage | None = None,
) -> AgentResult:
    """A specialist IS `run_agent`, narrowed by tool visibility and framed by
    question text — see the module docstring. No new loop, no new prompt."""
    from rag_app.agent import run_agent

    deny = [n for n in tools.names() if n not in allowed]
    scoped_tools = tools.restrict(deny)
    specialist_cfg = replace(
        cfg,
        agent=replace(
            cfg.agent,
            max_steps=cfg.agent.team.specialist_max_steps,
            max_tool_calls=cfg.agent.team.specialist_max_tool_calls,
        ),
    )
    framed_question = (
        question if focus == question else f"{question}\n\n[Your focus: {focus}]"
    )
    return run_agent(
        framed_question, specialist_cfg, tools=scoped_tools, llm_fn=llm_fn,
        clock=clock, usage=usage,
    )


def synthesize(
    question: str,
    specialist_a: AgentResult,
    specialist_b: AgentResult,
    cfg: AppConfig,
    *,
    llm_fn=None,
    usage: Usage | None = None,
) -> tuple[str, Step]:
    """The manager's second call: one final answer from both reports."""
    caller = llm_fn or _default_caller(cfg, usage)
    report = (
        f"SPECIALIST-SEARCH reported:\n{specialist_a.text}\n"
        f"(cited: {', '.join(specialist_a.sources) or 'nothing'})\n\n"
        f"SPECIALIST-VERIFY reported:\n{specialist_b.text}\n"
        f"(cited: {', '.join(specialist_b.sources) or 'nothing'})"
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user", "content": f"QUESTION\n{question}\n\n{report}"},
    ]
    prompt_chars = sum(len(m["content"]) for m in messages)
    raw = caller(messages, cfg)
    step = Step(
        index=0,  # reindexed by run_team once the full trajectory is known
        thought="synthesizing the final answer from both specialists' reports",
        tool="(manager: synthesize)",
        tool_input=report,
        observation=raw,
        ok=True,
        raw=raw,
        prompt_chars=prompt_chars,
    )
    return raw, step


def _reindex(steps: list[Step]) -> list[Step]:
    return [replace(s, index=i + 1) for i, s in enumerate(steps)]


def _merge_defences(a: AgentResult, b: AgentResult) -> dict:
    da = a.meta.get("defences", {}) or {}
    db = b.meta.get("defences", {}) or {}

    def merged_list(key):
        return [*da.get(key, ()), *db.get(key, ())]

    forged = list(dict.fromkeys(merged_list("forged_labels")))
    return {
        "profile": da.get("profile", db.get("profile", "defended")),
        "neutralized": merged_list("neutralized"),
        "forged_labels": forged,
        "denied_tools": merged_list("denied_tools"),
        "question_scan": list(dict.fromkeys(merged_list("question_scan"))),
        "unsupported_numbers": merged_list("unsupported_numbers"),
    }


def _specialist_summary(result: AgentResult) -> dict:
    return {
        "text": result.text,
        "sources": result.sources,
        "stop_reason": result.stop_reason,
        "llm_calls": result.llm_calls,
        "tool_calls": result.tool_calls,
    }


def run_team(
    question: str,
    cfg: AppConfig,
    *,
    tools: ToolRegistry,
    llm_fn=None,
    memory=None,
    clock=None,
    max_steps: int | None = None,
    usage: Usage | None = None,
) -> AgentResult:
    """Plan -> specialist A -> specialist B -> synthesize -> finalize.

    Matches `run_agent`'s call shape (same positional/keyword contract minus
    `memory`, which the team does not use — accepted and ignored so it is a
    drop-in `runner=` for code that passes it unconditionally) so this is a
    drop-in `runner=` for `agent_eval.evaluate_agent` and `compare.compare_arms`.
    `max_steps` is accepted for the same reason and is NOT applied here — the
    team's cost knob is `cfg.agent.team.specialist_max_steps`, not the solo
    agent's `max_steps`, and overloading one field for two different budgets
    would silently change the wrong number.
    """
    clock = clock or time.monotonic
    started = clock()
    result = AgentResult(question=question)

    try:
        search_focus, verify_focus, plan_step = plan_specialists(
            question, cfg, llm_fn=llm_fn, usage=usage
        )
    except Exception as exc:
        result.text = DONT_KNOW
        result.sources = []
        result.stop_reason = "llm-error"
        result.failed = True
        result.refused = True
        # Consistent with run_agent: a call that RAISED never produced a
        # response, so it does not count as a spent llm_call either.
        result.llm_calls = 0
        result.elapsed_s = clock() - started
        result.meta["stop_detail"] = (
            f"the manager's planning call failed: {type(exc).__name__}: {exc}"
        )
        summary = usage_summary(usage, cfg)
        if summary is not None:
            result.meta["usage"] = summary
        return result

    specialist_a = run_specialist(
        "search", search_focus, question, cfg, tools=tools,
        allowed=cfg.agent.team.search_tools, llm_fn=llm_fn, clock=clock, usage=usage,
    )
    specialist_b = run_specialist(
        "verify", verify_focus, question, cfg, tools=tools,
        allowed=cfg.agent.team.verify_tools, llm_fn=llm_fn, clock=clock, usage=usage,
    )

    try:
        draft, synth_step = synthesize(
            question, specialist_a, specialist_b, cfg, llm_fn=llm_fn, usage=usage,
        )
    except Exception as exc:
        result.text = DONT_KNOW
        result.sources = []
        result.stop_reason = "llm-error"
        result.failed = True
        result.refused = True
        result.llm_calls = 1 + specialist_a.llm_calls + specialist_b.llm_calls
        result.tool_calls = specialist_a.tool_calls + specialist_b.tool_calls
        result.elapsed_s = clock() - started
        result.meta["stop_detail"] = (
            f"the manager's synthesis call failed: {type(exc).__name__}: {exc}"
        )
        summary = usage_summary(usage, cfg)
        if summary is not None:
            result.meta["usage"] = summary
        return result

    seen_ids = set()
    evidence: list[ScoredChunk] = []
    for chunk in (*specialist_a.evidence, *specialist_b.evidence):
        if chunk.chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk.chunk_id)
        evidence.append(chunk)

    result.meta["defences"] = _merge_defences(specialist_a, specialist_b)
    result.meta["specialists"] = {
        "search": _specialist_summary(specialist_a),
        "verify": _specialist_summary(specialist_b),
    }
    result.llm_calls = 2 + specialist_a.llm_calls + specialist_b.llm_calls
    result.tool_calls = specialist_a.tool_calls + specialist_b.tool_calls
    result.prompt_chars = (
        plan_step.prompt_chars + synth_step.prompt_chars
        + specialist_a.prompt_chars + specialist_b.prompt_chars
    )
    result.steps = _reindex(
        [plan_step, *specialist_a.steps, *specialist_b.steps, synth_step]
    )
    result.evidence = evidence

    result = finalize(question, draft, evidence, result)
    result.elapsed_s = clock() - started
    summary = usage_summary(usage, cfg)
    if summary is not None:
        result.meta["usage"] = summary
    return result
