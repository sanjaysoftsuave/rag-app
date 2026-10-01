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
    implementation_verdict,
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


# ---------------------------------------------------------------------------
# BUG 1 / BUG 2 — a refusal is a decision, not an exhaustion
# ---------------------------------------------------------------------------


def test_a_task_that_blew_its_step_budget_is_not_scored_as_a_correct_refusal():
    """`_stop()` sets refused=True on every budget trip, so scoring on `refused`
    alone credited a run that merely ran out of steps. This inflated
    refusal_accuracy by however many unanswerable tasks were exhausting a budget.
    """
    r = report_of([(task(must_contain=[], unanswerable=True),
                    result(stop="max-steps", refused=True, failed=True))])
    assert r.refusal_accuracy == pytest.approx(0.0)


@pytest.mark.parametrize("stop", ["no-evidence", "model-refused"])
def test_a_refusal_by_decision_still_counts(stop):
    r = report_of([(task(must_contain=[], unanswerable=True),
                    result(stop=stop, refused=True))])
    assert r.refusal_accuracy == pytest.approx(1.0)


def test_no_evidence_counts_even_though_it_sets_failed():
    """It is the BEST kind of refusal — the gate caught an ungrounded answer —
    so `refused and not failed` would have been the wrong rule."""
    r = report_of([(task(must_contain=[], unanswerable=True),
                    result(stop="no-evidence", refused=True, failed=True))])
    assert r.refusal_accuracy == pytest.approx(1.0)


def test_refusal_accuracy_delegates_to_success_so_they_cannot_drift():
    rows = [(task(must_contain=[], unanswerable=True), result(stop="no-evidence"))]
    r = report_of(rows)
    assert r.refusal_accuracy == pytest.approx(1.0)
    assert r.unanswerable[0].success is True


def test_a_correct_refusal_counts_as_budget_adherent():
    """Requiring `final-answer` made every unanswerable task structurally
    unreachable, understating budget_adherence by its whole denominator."""
    r = report_of([(task(must_contain=[], unanswerable=True), result(stop="no-evidence"))])
    assert r.budget_adherence == pytest.approx(1.0)


def test_a_budget_trip_is_never_budget_adherent():
    r = report_of([(task(), result(stop="max-steps"))])
    assert r.budget_adherence == pytest.approx(0.0)
    assert len(r.stopped_on_budget) == 1


# ---------------------------------------------------------------------------
# Cost per task
# ---------------------------------------------------------------------------


def test_percentile_uses_nearest_rank_and_never_interpolates():
    from rag_app.agent_eval import percentile

    values = [1, 2, 3, 4, 5, 6, 7, 8]
    assert percentile(values, 99) == 8      # a value that was actually measured
    assert percentile(values, 50) == 4
    assert percentile([], 99) == 0.0
    assert percentile([5], 99) == 5


def test_p99_below_a_hundred_samples_is_the_maximum_and_the_report_says_so():
    from rag_app.agent_eval import percentile_is_max

    assert percentile_is_max(8, 99) is True
    assert percentile_is_max(99, 99) is True
    assert percentile_is_max(1000, 99) is False

    out = report_of([(task(), result())] * 8).describe()
    assert "IS the maximum" in out
    assert "nearest rank, no" in out


def test_cost_is_reported_over_every_task_including_refusals():
    rows = [
        (task(), result(tools=("a", "b", "c"))),
        (task(must_contain=[], unanswerable=True), result(stop="no-evidence")),
    ]
    stats = report_of(rows).cost_stats
    assert stats["steps"].n == 2            # the refusal is counted, not skipped
    assert stats["llm calls"].p99 >= stats["llm calls"].mean


def test_cost_reads_zero_without_a_usage_meter():
    """No meter means nothing was measured - the dim is present (every task
    contributes a row) but reads 0, not a guessed number."""
    out = report_of([(task(), result())]).describe()
    assert "COST PER TASK" in out
    assert "tokens" in out
    assert "cost usd" in out
    stats = report_of([(task(), result())]).cost_stats
    assert stats["tokens"].mean == 0.0
    assert stats["cost usd"].mean == 0.0


def test_cost_reports_a_real_dollar_figure_when_a_meter_was_used():
    """Week 10: real tokens off the API's own response, priced from a
    declared table - a computation, not the fabrication CLAUDE.md used to
    rule out."""
    metered = result(meta={"usage": {
        "prompt_tokens": 1000, "completion_tokens": 500,
        "total_tokens": 1500, "cost_usd": 0.01,
    }})
    report = report_of([(task(), metered)])
    assert report.cost_stats["tokens"].mean == 1500.0
    assert report.cost_stats["cost usd"].mean == pytest.approx(0.01)
    assert report.unpriced_tasks == []


def test_an_unpriced_model_reads_as_zero_cost_but_is_flagged_not_silently_guessed():
    unpriced = result(meta={"usage": {
        "prompt_tokens": 1000, "completion_tokens": 500,
        "total_tokens": 1500, "cost_usd": None,
    }})
    report = report_of([(task(), unpriced)])
    assert report.cost_stats["cost usd"].mean == 0.0  # the silent reading...
    assert len(report.unpriced_tasks) == 1            # ...but not a silent one
    out = report.describe()
    assert "no listed price" in out
    assert "token_prices" in out


def test_the_arena_does_not_credit_the_agent_for_running_out_of_steps():
    """BUG 1's second site: compare.py graded an unanswerable question on
    `refused` alone, and every budget trip sets it."""
    from rag_app.compare import _grade

    unanswerable = GoldQuestion(question="the weather?", unanswerable=True)
    assert _grade(unanswerable, "I don't know", True, "no-evidence") is True
    assert _grade(unanswerable, "I don't know", True, "max-steps") is False
    # the workflow arm passes no stop_reason and is unaffected
    assert _grade(unanswerable, "I don't know", True) is True


# ---------------------------------------------------------------------------
# Week 10: the honest implementation_verdict paragraph
# ---------------------------------------------------------------------------


def test_a_clear_quality_win_recommends_after_regardless_of_cost():
    before = {"task_success": 0.5, "n_answerable": 8, "mean_cost_usd": 0.001}
    after = {"task_success": 1.0, "n_answerable": 8, "mean_cost_usd": 0.01}
    out = implementation_verdict(before, after)
    assert "Quality improved" in out
    assert "VERDICT: keep AFTER" in out


def test_a_clear_quality_loss_recommends_before_even_if_cheaper():
    before = {"task_success": 1.0, "n_answerable": 8, "mean_cost_usd": 0.01}
    after = {"task_success": 0.5, "n_answerable": 8, "mean_cost_usd": 0.001}
    out = implementation_verdict(before, after)
    assert "Quality dropped" in out
    assert "VERDICT: keep BEFORE" in out


def test_a_quality_wash_within_noise_falls_back_to_cost():
    """At n=8 one task is 12.5% - a smaller delta is noise, not a finding."""
    before = {"task_success": 0.75, "n_answerable": 8, "mean_cost_usd": 0.01,
              "mean_tokens": 2000}
    after = {"task_success": 0.75, "n_answerable": 8, "mean_cost_usd": 0.03,
              "mean_tokens": 6000}
    out = implementation_verdict(before, after)
    assert "wash" in out
    assert "VERDICT: keep BEFORE" in out
    assert "3.0x" in out  # both cost and tokens tripled


def test_missing_metrics_on_either_side_never_guesses_a_verdict():
    out = implementation_verdict({}, {})
    assert "not enough measured" in out


def test_the_verdict_names_the_actual_multiples():
    before = {"task_success": 0.8, "n_answerable": 9, "mean_cost_usd": 0.002,
              "mean_tokens": 1000, "mean_llm_calls": 2.0, "mean_wall_clock_s": 1.5}
    after = {"task_success": 0.8, "n_answerable": 9, "mean_cost_usd": 0.008,
              "mean_tokens": 4000, "mean_llm_calls": 6.0, "mean_wall_clock_s": 3.0}
    out = implementation_verdict(before, after)
    assert "cost 4.0x" in out
    assert "tokens 4.0x" in out
    assert "LLM calls 3.0x" in out
    assert "wall clock 2.0x" in out
