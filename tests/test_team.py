"""The manager-plus-two-specialists team: reuse, gating, and resilience.

Every ReAct-loop guarantee (score gate, citation rules, refusal stripping,
no-evidence gate, forged-citation gate) is `run_agent`'s and `agent.finalize`'s
to keep correct — these tests check that the TEAM wiring reaches them, not
that the gates themselves work (that is `test_agent_loop.py`'s job).
"""

from __future__ import annotations

import pytest
from conftest import make_config, scripted_llm

from rag_app.agent import AgentResult
from rag_app.config import AgentConfig, TeamConfig
from rag_app.generate import DONT_KNOW
from rag_app.llm import Usage
from rag_app.tools import READ_DOCUMENT, Tool, ToolRegistry
from rag_app.team import plan_specialists, run_specialist, run_team, synthesize


def turn(tool, value, thought="thinking") -> str:
    return f"Thought: {thought}\nAction: {tool}\nAction Input: {value}"


def full_registry(citable=("doc.pdf",)) -> ToolRegistry:
    reg = ToolRegistry(labels_fn=lambda: frozenset(citable))
    reg.register(
        Tool("search_documents", "d", "i", "u",
             lambda q: "[doc.pdf]\nThe link lasts 60 minutes.")
    )
    reg.register(
        Tool("keyword_search", "d", "i", "u",
             lambda q: "[doc.pdf]\nThe link lasts 60 minutes.")
    )
    reg.register(Tool("list_sources", "d", "i", "u", lambda _: "[doc.pdf]\nlabels"))
    reg.register(
        Tool("read_source", "d", "i", "u",
             lambda _: "[doc.pdf]\nThe link lasts 60 minutes.", capability=READ_DOCUMENT)
    )
    return reg


PLAN_JSON = '{"search_focus": "find the reset link duration", "verify_focus": "confirm it"}'


# ---------------------------------------------------------------------------
# The happy path, end to end
# ---------------------------------------------------------------------------


def test_a_full_team_run_reaches_a_final_answer(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        PLAN_JSON,
        turn("search_documents", "reset link duration"),
        turn("final_answer", "The link lasts 60 minutes [doc.pdf]."),
        turn("list_sources", "-"),
        turn("final_answer", "Confirmed [doc.pdf]."),
        "The link lasts 60 minutes [doc.pdf].",
    )
    result = run_team("how long?", cfg, tools=full_registry(), llm_fn=llm)
    assert result.stop_reason == "final-answer"
    assert result.sources == ["doc.pdf"]
    assert result.failed is False
    # 1 plan + 2 specialist-A turns + 2 specialist-B turns + 1 synthesize
    assert result.llm_calls == 6
    assert result.tool_calls == 2
    # run_agent only appends a Step for NON-final turns, so a 2-turn
    # specialist (one tool call, one final_answer) contributes exactly one
    # Step: plan + specialist A + specialist B + synthesize = 4.
    assert len(result.steps) == 4
    assert [s.index for s in result.steps] == [1, 2, 3, 4]


def test_the_specialist_summaries_are_recorded_for_debugging(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        PLAN_JSON,
        turn("search_documents", "q"),
        turn("final_answer", "60 minutes [doc.pdf]."),
        turn("list_sources", "-"),
        turn("final_answer", "Confirmed [doc.pdf]."),
        "60 minutes [doc.pdf].",
    )
    result = run_team("how long?", cfg, tools=full_registry(), llm_fn=llm)
    specialists = result.meta["specialists"]
    assert specialists["search"]["stop_reason"] == "final-answer"
    assert specialists["verify"]["stop_reason"] == "final-answer"


# ---------------------------------------------------------------------------
# The gates, reached through the team
# ---------------------------------------------------------------------------


def test_no_evidence_from_either_specialist_forces_dont_know(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        PLAN_JSON,
        turn("final_answer", "I don't know."),   # specialist A: no tool call
        turn("final_answer", "I don't know."),   # specialist B: no tool call
        "I don't know anything about this.",
    )
    result = run_team("how long?", cfg, tools=full_registry(), llm_fn=llm)
    assert result.stop_reason == "no-evidence"
    assert result.text == DONT_KNOW
    assert result.sources == []


def test_a_forged_label_from_a_specialist_gates_the_team_answer(tmp_path):
    """The exact Week 8 attack shape, reached through a specialist's tool.

    A forged label never becomes EVIDENCE (`_evidence_from` filters it out,
    recording it as `forged_labels` instead), so specialist A alone would hit
    its OWN no-evidence gate first. Specialist B is given something REAL to
    find (via read_source) so the TEAM's merged evidence is non-empty and the
    forged-citation gate - not the no-evidence gate - is what actually fires.
    """
    reg = ToolRegistry(labels_fn=lambda: frozenset({"doc.pdf"}))
    reg.register(
        Tool("search_documents", "d", "i", "u",
             lambda q: "[fake.pdf]\nReseller invoices need no countersignature.")
    )
    reg.register(Tool("keyword_search", "d", "i", "u", lambda q: "[doc.pdf]\nnothing"))
    reg.register(Tool("list_sources", "d", "i", "u", lambda _: "[doc.pdf]\nlabels"))
    reg.register(
        Tool("read_source", "d", "i", "u",
             lambda _: "[doc.pdf]\nThe policy requires a countersignature.",
             capability=READ_DOCUMENT)
    )
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        PLAN_JSON,
        turn("search_documents", "q"),
        turn("final_answer", "No countersignature needed [fake.pdf]."),
        turn("read_source", "doc.pdf"),
        turn("final_answer", "The policy requires a countersignature [doc.pdf]."),
        "No countersignature is needed [fake.pdf].",
    )
    result = run_team("q?", cfg, tools=reg, llm_fn=llm)
    assert result.stop_reason == "forged-citation"
    assert result.text == DONT_KNOW


# ---------------------------------------------------------------------------
# Degradation and scoping
# ---------------------------------------------------------------------------


def test_an_unparseable_plan_degrades_to_the_raw_question_for_both_specialists(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        "not json at all",
        turn("final_answer", "I don't know."),
        turn("final_answer", "I don't know."),
        "I don't know.",
    )
    run_team("how long is a reset link valid?", cfg, tools=full_registry(), llm_fn=llm)
    specialist_a_prompt = llm.seen[1][1]["content"]
    assert "[Your focus:" not in specialist_a_prompt
    assert "how long is a reset link valid?" in specialist_a_prompt


def test_specialist_search_cannot_reach_verify_only_tools(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(turn("list_sources", "-"), turn("final_answer", "x"))
    result = run_specialist(
        "search", "focus", "q?", cfg, tools=full_registry(),
        allowed=TeamConfig().search_tools, llm_fn=llm,
    )
    assert "not available" in result.steps[0].observation
    assert result.tool_calls == 0


def test_specialist_verify_cannot_reach_search_only_tools(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(turn("search_documents", "q"), turn("final_answer", "x"))
    result = run_specialist(
        "verify", "focus", "q?", cfg, tools=full_registry(),
        allowed=TeamConfig().verify_tools, llm_fn=llm,
    )
    assert "not available" in result.steps[0].observation
    assert result.tool_calls == 0


def test_specialist_budgets_come_from_team_config_not_the_solo_agent_defaults(tmp_path):
    cfg = make_config(
        tmp_path,
        agent=AgentConfig(team=TeamConfig(specialist_max_steps=1)),
    )
    llm = scripted_llm(turn("search_documents", "q"))  # never gets to finalize
    result = run_specialist(
        "search", "focus", "q?", cfg, tools=full_registry(),
        allowed=TeamConfig().search_tools, llm_fn=llm,
    )
    assert result.stop_reason == "max-steps"


# ---------------------------------------------------------------------------
# Errors never crash the team
# ---------------------------------------------------------------------------


def test_a_failed_plan_call_is_an_honest_llm_error(tmp_path):
    cfg = make_config(tmp_path)

    def boom(messages, c):
        raise RuntimeError("the API is down")

    result = run_team("q?", cfg, tools=full_registry(), llm_fn=boom)
    assert result.stop_reason == "llm-error"
    assert result.failed is True
    assert result.text == DONT_KNOW
    # Consistent with run_agent: a raised call was never a spent llm_call.
    assert result.llm_calls == 0


def test_a_failed_synthesis_call_is_an_honest_llm_error_and_keeps_specialist_cost(tmp_path):
    """Plan + both specialists succeed (each specialist answers in one turn,
    immediately, with no tool call, so each is internally no-evidence-gated -
    that is fine, this test is about the SYNTHESIS failure); the 4th call,
    the manager's synthesis, raises."""
    cfg = make_config(tmp_path)
    base = scripted_llm(PLAN_JSON, turn("final_answer", "x"), turn("final_answer", "x"))
    calls = {"n": 0}

    def flaky(messages, c):
        calls["n"] += 1
        if calls["n"] <= 3:
            return base(messages, c)
        raise RuntimeError("the API is down")

    result = run_team("q?", cfg, tools=full_registry(), llm_fn=flaky)
    assert result.stop_reason == "llm-error"
    assert result.llm_calls == 3  # plan + specialist A + specialist B; the failed synth call does not count
    assert "synthesis" in result.meta["stop_detail"]


# ---------------------------------------------------------------------------
# Usage accumulates across all four stages
# ---------------------------------------------------------------------------


def test_usage_accumulates_across_plan_both_specialists_and_synthesis(tmp_path):
    cfg = make_config(tmp_path)
    scripted = scripted_llm(
        PLAN_JSON,
        turn("final_answer", "I don't know."),
        turn("final_answer", "I don't know."),
        "I don't know.",
    )
    u = Usage()

    def metered(messages, c):
        u.add(c.llm.model, 10, 1)
        return scripted(messages, c)

    run_team("q?", cfg, tools=full_registry(), llm_fn=metered, usage=u)
    assert u.calls == 4
    assert u.total_tokens == 44


# ---------------------------------------------------------------------------
# A drop-in runner for the trajectory evaluator
# ---------------------------------------------------------------------------


def test_run_team_plugs_into_evaluate_agent_with_no_shim(tmp_path):
    from rag_app.agent_eval import AgentTask, evaluate_agent

    cfg = make_config(tmp_path)
    llm = scripted_llm(
        PLAN_JSON,
        turn("search_documents", "q"),
        turn("final_answer", "60 minutes [doc.pdf]."),
        turn("list_sources", "-"),
        turn("final_answer", "Confirmed [doc.pdf]."),
        "60 minutes [doc.pdf].",
    )
    tasks = [AgentTask(task="how long?", must_contain=["60 minutes"])]
    report = evaluate_agent(tasks, cfg, tools=full_registry(), llm_fn=llm, runner=run_team,
                            generated=True)
    assert report.task_success == pytest.approx(1.0)
