"""The LangGraph arm.

Skipped unless the optional extra is installed. The one test that matters is
`test_both_implementations_agree_on_the_same_scripted_replies` — it is the
comparison's entire evidence, and without it "same agent, different machinery"
is an assertion rather than a fact.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
from conftest import make_config, scripted_llm

from rag_app.agent import run_agent
from rag_app.agent_langgraph import EXTRA_HINT, available, run_agent_langgraph
from rag_app.tools import Tool, ToolRegistry


def turn(tool, value, thought="thinking") -> str:
    return f"Thought: {thought}\nAction: {tool}\nAction Input: {value}"


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\nthe body text")
    )
    reg.register(Tool("list_sources", "d", "i", "u", lambda _: "[doc.pdf]\nlabels"))
    return reg


SCRIPT = (
    turn("list_sources", "-"),
    turn("search_documents", "reset link"),
    turn("final_answer", "The link lasts 60 minutes [doc.pdf]."),
)


def test_the_module_imports_without_the_extra_installed():
    """The CLI must be able to offer --impl langgraph and reject it politely."""
    import rag_app.agent_langgraph as mod

    assert callable(mod.run_agent_langgraph)
    assert "pip install" in EXTRA_HINT


def test_available_reports_whether_the_extra_is_present():
    assert available() == (find_spec("langgraph") is not None)


pytestmark_reason = 'pip install -e ".[agents]"'


@pytest.mark.skipif(find_spec("langgraph") is None, reason=pytestmark_reason)
def test_both_implementations_agree_on_the_same_scripted_replies(tmp_path):
    """Same tools, same prompt, same gate, different control flow.

    If this fails, the two arms are different agents and every number the
    comparison produces is measuring the wrong thing.
    """
    cfg = make_config(tmp_path)
    plain = run_agent("how long?", cfg, tools=registry(), llm_fn=scripted_llm(*SCRIPT))
    graph = run_agent_langgraph(
        "how long?", cfg, tools=registry(), llm_fn=scripted_llm(*SCRIPT)
    )

    assert [s.tool for s in plain.steps] == [s.tool for s in graph.steps]
    assert plain.text == graph.text
    assert plain.sources == graph.sources
    assert plain.stop_reason == graph.stop_reason
    assert plain.tool_calls == graph.tool_calls


@pytest.mark.skipif(find_spec("langgraph") is None, reason=pytestmark_reason)
def test_the_langgraph_arm_enforces_the_same_no_evidence_gate(tmp_path):
    cfg = make_config(tmp_path)
    r = run_agent_langgraph(
        "q?", cfg, tools=registry(),
        llm_fn=scripted_llm(turn("final_answer", "30 days, obviously.")),
    )
    assert r.stop_reason == "no-evidence"
    assert r.sources == []


@pytest.mark.skipif(find_spec("langgraph") is None, reason=pytestmark_reason)
def test_the_langgraph_arm_stops_on_a_budget(tmp_path):
    cfg = make_config(tmp_path)
    r = run_agent_langgraph(
        "q?", cfg, tools=registry(),
        llm_fn=scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)]),
        max_steps=2,
    )
    assert r.failed is True
    assert r.stop_reason in ("max-steps", "repeated-action")


@pytest.mark.skipif(find_spec("langgraph") is None, reason=pytestmark_reason)
def test_the_graph_can_print_itself(tmp_path):
    """The one thing the plain loop genuinely cannot do."""
    from rag_app.agent_langgraph import draw

    cfg = make_config(tmp_path)
    art = draw(cfg, registry())
    assert "think" in art and "act" in art


@pytest.mark.skipif(find_spec("langgraph") is not None, reason="the extra IS installed")
def test_without_the_extra_the_error_names_the_install_command(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(RuntimeError, match="pip install"):
        run_agent_langgraph("q?", cfg, tools=registry(), llm_fn=scripted_llm("x"))


@pytest.mark.skipif(find_spec("langgraph") is None, reason=pytestmark_reason)
def test_both_implementations_agree_on_cost_after_a_budget_stop(tmp_path):
    """The test that would have caught BUG 3.

    The existing parity test only covers the happy path, and the plain loop
    accounted correctly there. It was the BUDGET-stopped exits that skipped
    accounting — so the two arms silently disagreed on exactly the runs that
    cost the most, and nothing noticed.
    """
    cfg = make_config(tmp_path)
    script = [turn("search_documents", f"q{i}") for i in range(10)]

    plain = run_agent(
        "q?", cfg, tools=registry(), llm_fn=scripted_llm(*script), max_steps=3
    )
    graph = run_agent_langgraph(
        "q?", cfg, tools=registry(), llm_fn=scripted_llm(*script), max_steps=3
    )

    assert plain.stop_reason == graph.stop_reason
    assert plain.llm_calls == graph.llm_calls
    assert plain.tool_calls == graph.tool_calls
    assert len(plain.steps) == len(graph.steps)
    assert plain.llm_calls > 0, "a budget stop that spent nothing is the bug"
