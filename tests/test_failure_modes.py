"""The failure-mode classifier, and the outcome-vs-trajectory gap.

Every test here is pure over `(task, result)` — no store, no model, no config.
"""

from __future__ import annotations

import pytest

from rag_app.agent import AgentResult, Step
from rag_app.agent_eval import AgentEvalReport, AgentTask, TaskResult
from rag_app.failure_modes import (
    FAILURE_MODES,
    INFERRED_MODES,
    PATH_MODES,
    classify_failures,
    rank_modes,
)


def step(i, tool, tool_input="in", observation="obs", ok=True) -> Step:
    return Step(i, "t", tool, tool_input, observation, ok)


def result(steps=(), stop="final-answer", text="an answer", question="q?", **kw) -> AgentResult:
    return AgentResult(
        question=question, text=text, sources=["doc.pdf"], steps=list(steps),
        stop_reason=stop, **kw
    )


def task(**kw) -> AgentTask:
    base = dict(task="a task", must_contain=["x"])
    base.update(kw)
    return AgentTask(**base)


def modes(task_, result_) -> set[str]:
    return {f.mode for f in classify_failures(task_, result_)}


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------


def test_the_same_tool_with_the_same_input_twice_is_a_loop():
    r = result([step(1, "search_documents", "refunds"), step(2, "search_documents", "refunds")])
    assert "loop" in modes(task(), r)


def test_the_same_tool_with_different_inputs_is_not_a_loop():
    """Two searches for different things is research, not a loop."""
    r = result([step(1, "search_documents", "refunds"), step(2, "search_documents", "widgets")])
    assert "loop" not in modes(task(), r)


def test_the_loop_check_normalizes_the_input():
    r = result([step(1, "search_documents", "Refunds"), step(2, "search_documents", "refunds  ")])
    assert "loop" in modes(task(), r)


def test_stopping_on_the_repeat_limit_is_a_loop():
    r = result([], stop="repeated-action")
    r.meta["stop_detail"] = "called search_documents with the same input 3 times"
    flags = classify_failures(task(), r)
    assert "loop" in {f.mode for f in flags}
    assert "3 times" in next(f for f in flags if f.mode == "loop").evidence


# ---------------------------------------------------------------------------
# wrong tool / wrong sequence
# ---------------------------------------------------------------------------


def test_skipping_an_expected_tool_is_wrong_tool():
    r = result([step(1, "search_documents")])
    flags = classify_failures(task(expect_tools=["keyword_search"]), r)
    flag = next(f for f in flags if f.mode == "wrong-tool")
    assert "keyword_search" in flag.evidence
    assert flag.proven is True


def test_using_a_forbidden_tool_is_wrong_tool():
    r = result([step(1, "read_source")])
    assert "wrong-tool" in modes(task(forbid_tools=["read_source"]), r)


def test_the_expected_tools_in_the_wrong_order_is_wrong_sequence_not_wrong_tool():
    """The two claims are separate, which is why both metrics are reported."""
    r = result([step(1, "read_source"), step(2, "list_sources")])
    m = modes(task(expect_tools=["list_sources", "read_source"],
                   expect_sequence=["list_sources", "read_source"]), r)
    assert "wrong-sequence" in m
    assert "wrong-tool" not in m


def test_an_interleaved_extra_call_is_not_wrong_sequence_by_default():
    r = result([step(1, "list_sources"), step(2, "search_documents"), step(3, "read_source")])
    assert "wrong-sequence" not in modes(
        task(expect_sequence=["list_sources", "read_source"]), r
    )


def test_exact_match_rejects_the_interleaved_call():
    r = result([step(1, "list_sources"), step(2, "search_documents"), step(3, "read_source")])
    assert "wrong-sequence" in modes(
        task(expect_sequence=["list_sources", "read_source"], sequence_match="exact"), r
    )


# ---------------------------------------------------------------------------
# invented input — including what it cannot see
# ---------------------------------------------------------------------------


def test_a_hallucinated_citation_is_a_proven_invented_input():
    r = result([step(1, "search_documents")], hallucinated_citations=["CS-9999"])
    flag = next(f for f in classify_failures(task(), r) if f.mode == "invented-input")
    assert flag.proven is True
    assert "CS-9999" in flag.evidence


def test_calling_a_tool_that_does_not_exist_is_a_proven_invented_input():
    r = result([step(1, "teleport", observation="Unknown tool 'teleport'.", ok=False)])
    flag = next(f for f in classify_failures(task(), r) if f.mode == "invented-input")
    assert flag.proven is True


def test_looking_up_an_identifier_that_appeared_nowhere_is_inferred_not_proven():
    r = result([step(1, "keyword_search", "CS-9999")], question="what happened?")
    flag = next(f for f in classify_failures(task(), r) if f.mode == "invented-input")
    assert flag.proven is False
    assert "invented-input" in INFERRED_MODES


def test_an_identifier_from_the_question_is_not_invented():
    r = result([step(1, "keyword_search", "CS-1005")], question="what was CS-1005 about?")
    assert "invented-input" not in modes(task(), r)


def test_an_identifier_seen_in_an_earlier_observation_is_not_invented():
    r = result([
        step(1, "list_sources", "-", observation="[handbook.pdf]"),
        step(2, "read_source", "handbook.pdf"),
    ])
    assert "invented-input" not in modes(task(), r)


def test_an_ordinary_search_string_is_never_flagged_and_that_is_the_blind_spot():
    """The commonest invented input has no signal. This test documents the hole
    rather than pretending the classifier closes it."""
    r = result([step(1, "search_documents", "the refund window for enterprise customers")])
    assert "invented-input" not in modes(task(), r)


# ---------------------------------------------------------------------------
# give up / budget / step target
# ---------------------------------------------------------------------------


def test_refusing_an_answerable_task_is_a_quiet_give_up():
    assert "quiet-give-up" in modes(task(), result(stop="no-evidence"))


def test_refusing_an_unanswerable_task_is_not_a_failure():
    assert "quiet-give-up" not in modes(task(unanswerable=True), result(stop="no-evidence"))


def test_a_budget_trip_is_loud_not_quiet():
    """The distinction BUG 1 erased: exhaustion is not a decision."""
    r = result(stop="max-steps")
    r.meta["stop_detail"] = "stopped after 6 steps - agent.max_steps=6"
    m = modes(task(), r)
    assert "budget-exhausted" in m
    assert "quiet-give-up" not in m


def test_too_many_steps_misses_the_step_target():
    r = result([step(i, "search_documents", f"q{i}") for i in range(1, 6)])
    assert "step-target-missed" in modes(task(max_steps=3), r)


def test_too_few_steps_also_misses_it():
    """A two-hop task answered in one step probably answered from the model's
    own weights — right answer, wrong route."""
    r = result([step(1, "search_documents")])
    flags = classify_failures(task(min_steps=2), r)
    flag = next(f for f in flags if f.mode == "step-target-missed")
    assert "without the work the task describes" in flag.evidence


# ---------------------------------------------------------------------------
# Shape of the classification
# ---------------------------------------------------------------------------


def test_modes_co_occur_and_all_are_reported():
    """No precedence order: a loop that then exhausts max-steps is both."""
    r = result([step(1, "search_documents", "x"), step(2, "search_documents", "x")],
               stop="max-steps")
    m = modes(task(expect_tools=["keyword_search"]), r)
    assert {"loop", "budget-exhausted", "wrong-tool"} <= m


def test_a_clean_trajectory_has_no_flags():
    r = result([step(1, "search_documents")])
    assert classify_failures(task(expect_tools=["search_documents"]), r) == []


def test_every_mode_has_a_definition():
    r = result([step(1, "x", "y"), step(2, "x", "y")], stop="max-steps")
    for flag in classify_failures(task(min_steps=9), r):
        assert flag.mode in FAILURE_MODES


def test_quiet_give_up_is_not_a_path_mode():
    """It is an outcome problem. Including it would make 'right answer, wrong
    path' unsatisfiable by construction."""
    assert "quiet-give-up" not in PATH_MODES
    assert set(PATH_MODES) <= set(FAILURE_MODES)


def test_rank_modes_counts_trajectories_not_flags():
    from rag_app.failure_modes import FailureFlag

    ranked = rank_modes([
        [FailureFlag("loop", "a"), FailureFlag("loop", "b")],   # one trajectory
        [FailureFlag("loop", "c")],
        [FailureFlag("wrong-tool", "d")],
    ])
    assert ranked[0] == ("loop", 2)
    assert ("wrong-tool", 1) in ranked


def test_rank_modes_is_stable_for_ties():
    from rag_app.failure_modes import FailureFlag

    a = rank_modes([[FailureFlag("wrong-tool", "x")], [FailureFlag("loop", "y")]])
    b = rank_modes([[FailureFlag("loop", "y")], [FailureFlag("wrong-tool", "x")]])
    assert a == b


# ---------------------------------------------------------------------------
# The gap, at report level
# ---------------------------------------------------------------------------


def report(rows) -> AgentEvalReport:
    return AgentEvalReport(results=rows)


def test_a_right_answer_by_a_wrong_route_is_the_gap():
    r = TaskResult(
        task(must_contain=["widgets"], expect_tools=["keyword_search"]),
        result([step(1, "search_documents")], text="it was widgets"),
    )
    assert r.success is True
    assert r.path_ok is False
    assert r.right_answer_wrong_path is True
    assert report([r]).gap_rate == pytest.approx(1.0)


def test_a_wrong_answer_by_a_clean_route_is_the_mirror():
    r = TaskResult(
        task(must_contain=["widgets"], expect_tools=["search_documents"]),
        result([step(1, "search_documents")], text="no idea"),
    )
    assert r.wrong_answer_right_path is True
    assert report([r]).clean_path_failures == [r]


def test_the_gap_denominator_is_successful_tasks_not_all_tasks():
    good = TaskResult(task(must_contain=["x"], expect_tools=["search_documents"]),
                      result([step(1, "search_documents")], text="x"))
    gapped = TaskResult(task(must_contain=["x"], expect_tools=["keyword_search"]),
                        result([step(1, "search_documents")], text="x"))
    failed = TaskResult(task(must_contain=["x"], expect_tools=["search_documents"]),
                        result([step(1, "search_documents")], text="nope"))
    rep = report([good, gapped, failed])
    assert len(rep.successful) == 2
    assert rep.gap_rate == pytest.approx(0.5)   # 1 of 2 successful, not 1 of 3


def test_the_matrix_partitions_every_task():
    rows = [
        TaskResult(task(must_contain=["x"], expect_tools=["search_documents"]),
                   result([step(1, "search_documents")], text="x")),
        TaskResult(task(must_contain=["x"], expect_tools=["keyword_search"]),
                   result([step(1, "search_documents")], text="x")),
    ]
    assert sum(report(rows).outcome_path_matrix.values()) == 2


def test_describe_prints_the_gap_list_and_says_it_is_a_list_not_a_rate():
    rows = [TaskResult(
        task(task="the gapped one", must_contain=["x"], expect_tools=["keyword_search"]),
        result([step(1, "search_documents")], text="x"),
    )]
    out = report(rows).describe()
    assert "RIGHT ANSWER, WRONG PATH" in out
    assert "the gapped one" in out
    assert "expected keyword_search" in out     # the evidence sentence
    assert "these are lists, not rates" in out


def test_describe_ranks_the_failure_modes_and_warns_they_overlap():
    rows = [TaskResult(
        task(expect_tools=["keyword_search"]),
        result([step(1, "search_documents", "x"), step(2, "search_documents", "x")]),
    )]
    out = report(rows).describe()
    assert "FAILURE MODES" in out
    assert "Rows OVERLAP" in out


def test_describe_marks_the_inferred_mode_and_explains_the_blind_spot():
    rows = [TaskResult(
        task(), result([step(1, "keyword_search", "CS-9999")], question="what happened?")
    )]
    out = report(rows).describe()
    assert "inferred from a proxy" in out
    assert "under-counts" in out


def test_both_sequence_denominators_are_printed():
    rows = [
        TaskResult(task(expect_tools=["search_documents"]),
                   result([step(1, "search_documents")])),
        TaskResult(task(expect_sequence=["list_sources", "read_source"]),
                   result([step(1, "read_source"), step(2, "list_sources")])),
    ]
    out = report(rows).describe()
    assert "of 2; 1 declare one" in out          # tool choice denominator
    assert "of 1 that declare one" in out        # sequence denominator
    assert "right tools, wrong order" in out
