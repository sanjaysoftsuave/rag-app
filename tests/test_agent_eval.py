"""Trajectory evaluation and the workflow-vs-agent comparison."""

from __future__ import annotations

import json

import pytest
from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store, scripted_llm

from rag_app.agent import AgentResult, Step
from rag_app.agent_eval import (
    AgentEvalReport,
    AgentTask,
    TaskResult,
    agent_report_to_json,
    evaluate_agent,
    load_agent_tasks,
    parse_agent_tasks,
)
from rag_app.chunking import Chunk
from rag_app.compare import ComparisonReport, compare_arms, comparison_to_json
from rag_app.evaluate import GoldQuestion
from rag_app.tools import Tool, ToolRegistry


def task(**kw) -> AgentTask:
    base = dict(task="a task", must_contain=["60 minutes"])
    base.update(kw)
    return AgentTask(**base)


def result(text="the link lasts 60 minutes", tools=("search_documents",),
           sources=("doc.pdf",), stop="final-answer", **kw) -> AgentResult:
    steps = [
        Step(i + 1, "t", name, "in", "obs", True) for i, name in enumerate(tools)
    ]
    return AgentResult(
        question="q", text=text, sources=list(sources), steps=steps,
        stop_reason=stop, tool_calls=len(steps), llm_calls=len(steps) + 1, **kw
    )


def report_of(pairs) -> AgentEvalReport:
    return AgentEvalReport(results=[TaskResult(t, r) for t, r in pairs])


# ---------------------------------------------------------------------------
# Task parsing
# ---------------------------------------------------------------------------


def test_a_task_needs_something_to_check():
    with pytest.raises(ValueError, match="names nothing to check"):
        parse_agent_tasks([{"task": "do something"}])


def test_an_unanswerable_task_needs_no_expectations():
    [t] = parse_agent_tasks([{"task": "the weather?", "unanswerable": True}])
    assert t.answerable is False


def test_expect_tools_alone_is_enough_to_check():
    [t] = parse_agent_tasks([{"task": "x", "expect_tools": "keyword_search"}])
    assert t.expect_tools == ["keyword_search"]


def test_a_malformed_task_names_its_position():
    with pytest.raises(ValueError, match="task 2"):
        parse_agent_tasks([{"task": "ok", "must_contain": "x"}, {"note": "no task"}])


def test_a_missing_task_file_points_at_the_example(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="agent_tasks.example.yaml"):
        load_agent_tasks(cfg)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_task_success_requires_every_must_contain():
    good = report_of([(task(), result("the link lasts 60 minutes"))])
    bad = report_of([(task(), result("the link expires eventually"))])
    assert good.task_success == pytest.approx(1.0)
    assert bad.task_success == pytest.approx(0.0)


def test_task_success_also_requires_the_expected_sources_to_be_cited():
    r = report_of([(task(expect_sources=["handbook.pdf"]), result(sources=["other.pdf"]))])
    assert r.task_success == pytest.approx(0.0)


def test_tool_choice_counts_expected_and_forbidden():
    expected = report_of([(task(expect_tools=["keyword_search"]),
                           result(tools=("search_documents",)))])
    assert expected.tool_choice_accuracy == pytest.approx(0.0)

    forbidden = report_of([(task(forbid_tools=["read_source"]),
                            result(tools=("search_documents", "read_source")))])
    assert forbidden.tool_choice_accuracy == pytest.approx(0.0)

    ok = report_of([(task(expect_tools=["search_documents"], forbid_tools=["read_source"]),
                     result(tools=("search_documents",)))])
    assert ok.tool_choice_accuracy == pytest.approx(1.0)


def test_budget_adherence_counts_only_final_answer_stops():
    stopped = report_of([(task(), result(stop="max-steps"))])
    assert stopped.budget_adherence == pytest.approx(0.0)


def test_budget_adherence_respects_the_tasks_own_step_target():
    too_many = report_of([
        (task(max_steps=1), result(tools=("a", "b", "c")))
    ])
    assert too_many.budget_adherence == pytest.approx(0.0)


def test_a_multi_hop_task_can_require_a_minimum_number_of_steps():
    """A one-shot answer to a two-hop question is suspicious, not efficient."""
    one_shot = report_of([(task(min_steps=2), result(tools=("search_documents",)))])
    assert one_shot.budget_adherence == pytest.approx(0.0)


def test_an_unanswerable_task_succeeds_by_refusing():
    refused = report_of([
        (task(must_contain=[], unanswerable=True), result(stop="no-evidence", refused=True))
    ])
    assert refused.refusal_accuracy == pytest.approx(1.0)


def test_false_refusals_are_listed_separately():
    r = report_of([(task(), result(refused=True))])
    assert len(r.false_refusals) == 1
    assert "Refused but answerable" in r.describe()


def test_the_no_evidence_rate_surfaces_the_agent_only_failure():
    r = report_of([(task(), result(stop="no-evidence"))])
    assert r.no_evidence_rate == pytest.approx(1.0)
    assert "agent-only failure mode" in r.describe()


def test_stop_reason_counts_show_which_budget_binds():
    r = report_of([
        (task(), result(stop="max-steps")),
        (task(), result(stop="max-steps")),
        (task(), result(stop="final-answer")),
    ])
    assert r.stop_reason_counts["max-steps"] == 2
    assert "Stop reasons" in r.describe()


def test_the_report_says_to_read_success_and_efficiency_together():
    assert "produce a wrong answer" in report_of([(task(), result())]).describe()


def test_describe_is_readable_with_zero_tasks():
    out = AgentEvalReport().describe()
    assert "0 answerable" in out


def test_json_carries_every_headline_metric():
    payload = json.loads(agent_report_to_json(report_of([(task(), result())])))
    for key in (
        "task_success", "tool_choice_accuracy", "budget_adherence",
        "no_evidence_rate", "mean_steps", "stop_reason_counts",
    ):
        assert key in payload


def test_evaluate_agent_runs_each_task_through_the_loop(tmp_path):
    cfg = make_config(tmp_path)
    reg = ToolRegistry()
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\n60 minutes"))
    llm = scripted_llm(
        "Thought: t\nAction: search_documents\nAction Input: q",
        "Thought: t\nAction: final_answer\nAction Input: it lasts 60 minutes [doc.pdf]",
    )
    report = evaluate_agent([task(expect_sources=["doc.pdf"])], cfg, tools=reg, llm_fn=llm)
    assert report.task_success == pytest.approx(1.0)
    assert report.budget_adherence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Workflow vs Agent
# ---------------------------------------------------------------------------


def build_store(tmp_path):
    embedder = FakeEmbedder()
    chunks = [Chunk("doc.pdf::0", "doc.pdf", "Password reset links expire after 60 minutes.", {})]
    vectors = embedder.encode_documents([c.text for c in chunks])
    return embedder, make_qdrant_store(tmp_path, chunks, vectors)


def test_both_arms_answer_the_same_questions(tmp_path):
    embedder, store = build_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    reg = ToolRegistry()
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\n60 minutes"))
    gold = [GoldQuestion(question="how long?", must_contain=["60 minutes"])]

    report = compare_arms(
        gold, cfg, preset="C", store=store, embedder=embedder,
        reranker=FakeReranker(0.9), tools=reg,
        generate_fn=lambda q, c, k: "It lasts 60 minutes [doc.pdf].",
        llm_fn=scripted_llm(
            "Thought: t\nAction: search_documents\nAction Input: q",
            "Thought: t\nAction: final_answer\nAction Input: It lasts 60 minutes [doc.pdf].",
        ),
        generated=True,
    )
    store.close()
    assert {r.arm for r in report.rows} == {"workflow", "agent"}
    assert all(r.ok for r in report.rows)


def test_the_workflow_arm_calls_the_generator_exactly_once_per_question(tmp_path):
    embedder, store = build_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    gold = [GoldQuestion(question="how long?", must_contain=["60 minutes"])]
    report = compare_arms(
        gold, cfg, arms=("workflow",), preset="C", store=store, embedder=embedder,
        reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "It lasts 60 minutes.",
        generated=True,
    )
    store.close()
    assert report.per_arm("workflow")[0].llm_calls == 1


def test_llm_calls_are_counted_per_arm_and_the_delta_is_stated(tmp_path):
    embedder, store = build_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    reg = ToolRegistry()
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\n60 minutes"))
    gold = [GoldQuestion(question="how long?", must_contain=["60 minutes"])]
    report = compare_arms(
        gold, cfg, preset="C", store=store, embedder=embedder,
        reranker=FakeReranker(0.9), tools=reg,
        generate_fn=lambda q, c, k: "60 minutes [doc.pdf]",
        llm_fn=scripted_llm(
            "Thought: t\nAction: search_documents\nAction Input: q",
            "Thought: t\nAction: final_answer\nAction Input: 60 minutes [doc.pdf]",
        ),
        generated=True,
    )
    store.close()
    out = report.describe()
    assert "LLM calls" in out
    assert "2.0x the LLM calls" in out       # agent spent 2, workflow spent 1
    assert "does not carry" in out           # the honest cost caveat


def test_a_dry_run_is_labelled_as_shape_not_behaviour():
    report = ComparisonReport(rows=[], arms=("workflow", "agent"), generated=False)
    assert "DRY RUN" in report.describe()


def test_comparison_json_carries_per_arm_totals(tmp_path):
    embedder, store = build_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    gold = [GoldQuestion(question="how long?", must_contain=["60 minutes"])]
    report = compare_arms(
        gold, cfg, arms=("workflow",), preset="C", store=store, embedder=embedder,
        reranker=FakeReranker(0.9), generate_fn=lambda q, c, k: "60 minutes", generated=True,
    )
    store.close()
    payload = json.loads(comparison_to_json(report))
    assert payload["totals"]["workflow"]["questions"] == 1
    assert payload["totals"]["workflow"]["correct"] == 1
