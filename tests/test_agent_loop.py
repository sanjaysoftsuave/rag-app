"""The ReAct loop, its budgets, and the relocated grounding gate.

The grounding tests here are the Week 7 sibling of `test_grounding.py`. They
matter more than the rest of the file: an agent that answers from its own
weights when retrieval returned nothing produces fluent, plausible, uncited
prose, and no pre-existing test in this repo would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    FakeClock,
    FakeEmbedder,
    FakeReranker,
    boom_llm,
    make_config,
    make_qdrant_store,
    scripted_llm,
)

from rag_app.agent import SYSTEM, AgentResult, Budget, build_prompt, finalize, run_agent
from rag_app.chunking import Chunk
from rag_app.generate import CITATION_RULES, DONT_KNOW
from rag_app.store import ScoredChunk
from rag_app.tools import Tool, ToolRegistry, build_registry


def turn(tool, value, thought="thinking") -> str:
    return f"Thought: {thought}\nAction: {tool}\nAction Input: {value}"


def static_tool(name="search_documents", observation="[doc.pdf]\nthe body text") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(name=name, description="d", input_desc="i", usage="u",
             run=lambda _: observation)
    )
    return reg


def seeded(tmp_path, texts=(("doc.pdf", "Password reset links expire after 60 minutes."),)):
    embedder = FakeEmbedder()
    chunks = [Chunk(f"{s}::{i}", s, t, {}) for i, (s, t) in enumerate(texts)]
    vectors = embedder.encode_documents([c.text for c in chunks])
    return embedder, make_qdrant_store(tmp_path, chunks, vectors)


# ---------------------------------------------------------------------------
# The loop actually loops
# ---------------------------------------------------------------------------


def test_a_one_step_agent_returns_the_final_answer(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(turn("final_answer", "The link lasts 60 minutes [doc.pdf]."))
    tools = static_tool()
    # Evidence has to exist for the answer to survive the no-evidence gate,
    # so give it one search first.
    llm = scripted_llm(
        turn("search_documents", "reset link"),
        turn("final_answer", "The link lasts 60 minutes [doc.pdf]."),
    )
    r = run_agent("how long?", cfg, tools=tools, llm_fn=llm)
    assert r.stop_reason == "final-answer"
    assert r.failed is False
    assert r.sources == ["doc.pdf"]
    assert r.llm_calls == 2
    assert r.tool_calls == 1


def test_the_observation_of_step_n_appears_in_the_prompt_of_step_n_plus_1(tmp_path):
    """The test that proves this is a loop and not two independent calls."""
    cfg = make_config(tmp_path)
    tools = static_tool(observation="[doc.pdf]\nDISTINCTIVE-OBSERVATION-TEXT")
    llm = scripted_llm(
        turn("search_documents", "q"),
        turn("final_answer", "done [doc.pdf]"),
    )
    run_agent("q?", cfg, tools=tools, llm_fn=llm)
    second_prompt = llm.seen[1][1]["content"]
    assert "DISTINCTIVE-OBSERVATION-TEXT" in second_prompt
    assert "WORK SO FAR" in second_prompt


def test_a_three_step_trajectory_records_every_thought_action_observation(tmp_path):
    cfg = make_config(tmp_path)
    reg = ToolRegistry()
    reg.register(Tool("list_sources", "d", "i", "u", lambda _: "[doc.pdf]\nlabels"))
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "[doc.pdf]\nbody"))
    llm = scripted_llm(
        turn("list_sources", "-", thought="what documents exist"),
        turn("read_source", "doc.pdf", thought="read it"),
        turn("final_answer", "answer [doc.pdf]", thought="done"),
    )
    r = run_agent("q?", cfg, tools=reg, llm_fn=llm)
    assert [s.tool for s in r.steps] == ["list_sources", "read_source"]
    assert r.steps[0].thought == "what documents exist"
    assert "labels" in r.steps[0].observation
    assert r.tool_calls == 2


def test_an_unknown_tool_becomes_an_observation_not_an_exception(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        turn("teleport", "x"),
        turn("search_documents", "q"),
        turn("final_answer", "done [doc.pdf]"),
    )
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert "Unknown tool" in r.steps[0].observation
    assert "search_documents" in r.steps[0].observation  # it lists the real ones
    assert r.stop_reason == "final-answer"


def test_a_tool_that_raises_becomes_an_observation(tmp_path):
    cfg = make_config(tmp_path)

    def explode(_):
        raise RuntimeError("disk on fire")

    reg = ToolRegistry()
    reg.register(Tool("search_documents", "d", "i", "u", explode))
    # _safe() wraps builders, not hand-made Tools, so wrap explicitly here
    from rag_app.tools import _safe

    reg = ToolRegistry()
    reg.register(Tool("search_documents", "d", "i", "u", _safe("search_documents", explode)))
    llm = scripted_llm(turn("search_documents", "q"), turn("final_answer", "x"))
    r = run_agent("q?", cfg, tools=reg, llm_fn=llm)
    assert "disk on fire" in r.steps[0].observation
    assert r.steps[0].ok is True  # the tool ran and reported; the loop survived


def test_an_llm_error_stops_visibly(tmp_path):
    cfg = make_config(tmp_path)
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=boom_llm)
    assert r.stop_reason == "llm-error"
    assert r.failed is True
    assert r.text == DONT_KNOW
    assert "the API is down" in r.meta["stop_detail"]


# ---------------------------------------------------------------------------
# Parse failures
# ---------------------------------------------------------------------------


def test_a_parse_failure_feeds_a_repair_observation_and_continues(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        "I think the answer is 30 days.",           # no Action line
        turn("search_documents", "q"),
        turn("final_answer", "done [doc.pdf]"),
    )
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert "Malformed output" in r.steps[0].observation
    assert r.stop_reason == "final-answer"


def test_too_many_parse_failures_stop_visibly(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm("prose", "more prose", "still prose", "and again")
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.stop_reason == "unparseable"
    assert r.failed is True
    assert r.text == DONT_KNOW


# ---------------------------------------------------------------------------
# Budgets: every trip is a visible failure
# ---------------------------------------------------------------------------


def test_max_steps_stops_and_names_the_budget(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm, max_steps=2)
    assert r.stop_reason == "max-steps"
    assert "agent.max_steps=2" in r.meta["stop_detail"]


def test_every_budget_trip_returns_dont_know_with_no_sources(tmp_path):
    """A truncated run has not answered. Partial prose with sources would lie."""
    cfg = make_config(tmp_path)
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm, max_steps=1)
    assert r.text == DONT_KNOW
    assert r.sources == []
    assert r.failed is True
    assert r.evidence  # the trajectory and its evidence are still preserved


def test_the_prompt_char_budget_stops_before_calling_the_llm(tmp_path):
    """Tripwire: an over-budget prompt must cost nothing."""
    from dataclasses import replace

    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, max_prompt_chars=10))
    called = {"llm": False}

    def tripwire(messages, config):
        called["llm"] = True
        return turn("final_answer", "x")

    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=tripwire)
    assert called["llm"] is False
    assert r.stop_reason == "token-budget"
    assert r.failed is True


def test_the_wall_clock_budget_uses_the_injected_clock(tmp_path):
    """No sleeping: the suite is three seconds and stays that way."""
    from dataclasses import replace

    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, wall_clock_seconds=5.0))
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm, clock=FakeClock(step=3.0))
    assert r.stop_reason == "wall-clock"


def test_max_tool_calls_stops(tmp_path):
    from dataclasses import replace

    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, max_tool_calls=1, max_steps=10))
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.stop_reason == "max-tool-calls"


def test_max_llm_calls_is_separate_from_max_steps(tmp_path):
    """A parse failure burns a call without producing progress."""
    from dataclasses import replace

    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, max_llm_calls=2, max_steps=99,
                                     max_parse_failures=99))
    llm = scripted_llm("prose", "prose", "prose", "prose")
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.stop_reason == "max-llm-calls"


def test_a_repeated_action_is_warned_once_then_stopped(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(*[turn("search_documents", "identical") for _ in range(5)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert "already ran this exact call" in r.steps[1].observation
    assert r.stop_reason == "repeated-action"


def test_describe_names_the_budget_that_tripped(tmp_path):
    cfg = make_config(tmp_path)
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    out = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm, max_steps=2).describe()
    assert "max_steps=2" in out
    assert "Trajectory:" in out


# ---------------------------------------------------------------------------
# Grounding: the five guarantees at the agent boundary
# ---------------------------------------------------------------------------


def test_a_final_answer_without_any_retrieval_is_forced_to_dont_know(tmp_path):
    """Guarantee 5. The failure only the agent shape can produce."""
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        turn("final_answer", "The refund window is 30 days, everyone knows that.")
    )
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.stop_reason == "no-evidence"
    assert r.text == DONT_KNOW
    assert r.sources == []
    assert r.failed is True
    assert "from the model rather than the documents" in r.meta["stop_detail"]


def test_the_agent_prompt_uses_the_same_citation_rules_as_generate():
    """Two prompts that both 'explain citations' in different words are two
    different contracts, and only one was debugged against the real failure."""
    assert CITATION_RULES in SYSTEM


def test_the_agent_prompt_forbids_citing_recalled_memory():
    assert "never cite" in SYSTEM.lower()


def test_the_agent_reports_invented_citations(tmp_path):
    """Guarantee 4, over the union of evidence from every step."""
    cfg = make_config(tmp_path)
    llm = scripted_llm(
        turn("search_documents", "q"),
        turn("final_answer", "The answer is 60 minutes [CS-1001]."),
    )
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.hallucinated_citations == ["CS-1001"]
    assert r.stop_reason == "final-answer"


def test_an_agent_refusal_carries_no_sources(tmp_path):
    """Guarantee 3: crediting documents for a non-answer is the thing to avoid."""
    cfg = make_config(tmp_path)
    llm = scripted_llm(turn("search_documents", "q"), turn("final_answer", DONT_KNOW))
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm)
    assert r.stop_reason == "model-refused"
    assert r.refused is True
    assert r.sources == []


def test_the_search_tool_returns_no_excerpt_below_the_threshold(tmp_path):
    """Guarantee 1, relocated into the tool: the model cannot see rejected text."""
    embedder, store = seeded(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.9)
    tools = build_registry(
        cfg, store=store, embedder=embedder, reranker=FakeReranker(0.1),
        names=["search_documents"],
    )
    observation = tools.get("search_documents").run("anything")
    store.close()
    assert "No excerpt scored above the relevance threshold" in observation
    assert "Password reset" not in observation   # the text never reaches the model


def test_below_threshold_search_contributes_no_evidence_so_the_gate_fires(tmp_path):
    embedder, store = seeded(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.9)
    tools = build_registry(
        cfg, store=store, embedder=embedder, reranker=FakeReranker(0.1),
        names=["search_documents"],
    )
    llm = scripted_llm(
        turn("search_documents", "q"),
        turn("final_answer", "It lasts 60 minutes [doc.pdf]."),
    )
    r = run_agent("q?", cfg, tools=tools, llm_fn=llm)
    store.close()
    assert r.stop_reason == "no-evidence"
    assert r.text == DONT_KNOW


def test_finalize_prefers_the_models_own_citations_over_the_fallback():
    r = AgentResult(question="q")
    evidence = [ScoredChunk(Chunk("a::0", "doc.pdf", "body", {}), 0.9)]
    out = finalize("q", "answer [doc.pdf]", evidence, r)
    assert out.sources == ["doc.pdf"]
    assert out.meta["cited"] is True


def test_an_uncited_answer_falls_back_to_the_evidence_and_records_it():
    r = AgentResult(question="q")
    evidence = [ScoredChunk(Chunk("a::0", "doc.pdf", "body", {}), 0.9)]
    out = finalize("q", "answer with no citation", evidence, r)
    assert out.sources == ["doc.pdf"]
    assert out.meta["cited"] is False


# ---------------------------------------------------------------------------
# Cost accounting on every exit path (BUG 3)
# ---------------------------------------------------------------------------


def test_a_budget_stop_still_reports_what_it_spent(tmp_path):
    """Three of six exits used to skip accounting, so the trajectories that
    spent the MOST reported spending nothing — and a p99, which below 100 tasks
    is just the maximum, would have been drawn from exactly those runs."""
    cfg = make_config(tmp_path)
    llm = scripted_llm(*[turn("search_documents", f"q{i}") for i in range(10)])
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=llm, max_steps=3)
    assert r.stop_reason == "max-steps"
    assert len(r.steps) == 3
    assert r.llm_calls == 3          # was 0
    assert r.tool_calls == 3         # was 0
    assert r.elapsed_s >= 0.0


def test_a_prompt_budget_stop_reports_zero_because_nothing_was_spent(tmp_path):
    """The one case where zero is the honest number: the budget is checked
    BEFORE the call, so an over-budget prompt really did cost nothing."""
    from dataclasses import replace

    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, max_prompt_chars=10))
    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=scripted_llm("unused"))
    assert r.stop_reason == "token-budget"
    assert r.llm_calls == 0
    assert r.steps == []


def test_an_llm_error_reports_the_calls_made_before_it(tmp_path):
    cfg = make_config(tmp_path)

    class Flaky:
        def __init__(self):
            self.n = 0

        def __call__(self, messages, config):
            self.n += 1
            if self.n > 2:
                raise RuntimeError("the API is down")
            return turn("search_documents", f"q{self.n}")

    r = run_agent("q?", cfg, tools=static_tool(), llm_fn=Flaky())
    assert r.stop_reason == "llm-error"
    assert r.llm_calls == 2
    assert r.tool_calls == 2


def test_every_stop_reason_is_classified_exactly_once():
    """A 12th stop reason added without classifying it fails the build rather
    than being silently mis-scored by every metric that reads these sets."""
    from rag_app.agent import BUDGET_STOPS, REFUSAL_STOPS, SELF_TERMINATED, STOP_REASONS

    assert len(STOP_REASONS) == len(set(STOP_REASONS))
    assert REFUSAL_STOPS | BUDGET_STOPS | {"final-answer"} == set(STOP_REASONS)
    assert not (REFUSAL_STOPS & BUDGET_STOPS)
    assert "final-answer" not in REFUSAL_STOPS and "final-answer" not in BUDGET_STOPS
    assert SELF_TERMINATED == REFUSAL_STOPS | {"final-answer"}


def test_a_refusal_is_a_decision_and_a_budget_trip_is_an_exhaustion():
    from rag_app.agent import BUDGET_STOPS, REFUSAL_STOPS

    assert "no-evidence" in REFUSAL_STOPS and "model-refused" in REFUSAL_STOPS
    for reason in ("max-steps", "wall-clock", "unparseable", "repeated-action",
                   "max-tool-calls", "token-budget", "max-llm-calls", "llm-error"):
        assert reason in BUDGET_STOPS
