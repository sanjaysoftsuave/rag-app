"""Snapshots and the before/after diff."""

from __future__ import annotations

import json

import pytest
from conftest import make_config

from rag_app.before_after import (
    HIGHER_IS_BETTER,
    Snapshot,
    compare,
    comparison_to_json,
    gold_fingerprint,
    load_snapshot,
    make_snapshot,
    settings_of,
    snapshot_path,
    write_snapshot,
)
from rag_app.evaluate import GoldQuestion


def gold(*questions) -> list[GoldQuestion]:
    return [GoldQuestion(question=q, must_contain=["x"]) for q in questions]


def snap(label, metrics, settings=None, n=9, fp="aaa") -> Snapshot:
    return Snapshot(
        label=label, created="2026-01-01T00:00:00+00:00",
        settings=settings or {"retrieval_mode": "dense"},
        metrics=metrics, n_questions=n, n_answerable=n, gold_fingerprint=fp,
    )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_a_snapshot_round_trips_through_a_file(tmp_path):
    cfg = make_config(tmp_path)
    s = make_snapshot("before", cfg, {"mrr": 0.8}, gold("a", "b"))
    path = snapshot_path(cfg, "before")
    write_snapshot(s, path)
    back = load_snapshot(path)
    assert back.label == "before"
    assert back.metrics == {"mrr": 0.8}
    assert back.gold_fingerprint == s.gold_fingerprint


def test_a_snapshot_records_the_settings_that_produce_the_numbers(tmp_path):
    cfg = make_config(tmp_path)
    s = settings_of(cfg)
    for key in (
        "chunk_size", "overlap", "bi_encoder_model", "cross_encoder_model",
        "retrieve_k", "rerank_n", "score_threshold", "retrieval_mode",
        "query_mode", "mmr", "llm_model", "judge_model",
    ):
        assert key in s


def test_the_snapshot_metrics_come_from_report_to_json(tmp_path):
    """Prevents drift: a metric added to EvalReport must reach snapshots for free."""
    from rag_app.evaluate import EvalReport, report_to_json

    cfg = make_config(tmp_path)
    metrics = json.loads(report_to_json(EvalReport(preset="C", k=10, n=3, results=[])))
    s = make_snapshot("x", cfg, metrics, gold("a"))
    for key in metrics:
        assert key in s.metrics


def test_a_missing_snapshot_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="--snapshot"):
        load_snapshot(tmp_path / "nope.json")


def test_the_gold_fingerprint_changes_when_an_expectation_changes():
    a = [GoldQuestion(question="q", must_contain=["x"])]
    b = [GoldQuestion(question="q", must_contain=["y"])]
    assert gold_fingerprint(a) != gold_fingerprint(b)


def test_the_gold_fingerprint_is_stable_across_ordering_of_expectations():
    a = [GoldQuestion(question="q", must_contain=["x", "y"])]
    b = [GoldQuestion(question="q", must_contain=["y", "x"])]
    assert gold_fingerprint(a) == gold_fingerprint(b)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_a_changed_gold_set_is_flagged_loudly_and_the_deltas_still_print():
    c = compare(
        snap("before", {"mrr": 0.5}, fp="aaa"),
        snap("after", {"mrr": 0.9}, {"retrieval_mode": "hybrid"}, fp="bbb"),
    )
    out = c.describe()
    assert any("gold set changed" in w for w in c.warnings)
    assert "aaa -> bbb" in out
    assert "0.9" in out          # degrade visibly, do not refuse


def test_the_comparison_lists_which_settings_changed_first():
    c = compare(
        snap("before", {"mrr": 0.5}, {"retrieval_mode": "dense", "rerank_n": 3}),
        snap("after", {"mrr": 0.9}, {"retrieval_mode": "hybrid", "rerank_n": 3}),
    )
    assert c.changed_settings == [("retrieval_mode", "dense", "hybrid")]
    out = c.describe()
    assert out.index("WHAT CHANGED") < out.index("WHAT IT BOUGHT")


def test_identical_settings_are_flagged_as_run_to_run_variation():
    c = compare(snap("a", {"mrr": 0.5}), snap("b", {"mrr": 0.6}))
    assert any("no configuration difference" in w for w in c.warnings)


def test_a_metric_present_on_one_side_only_is_not_a_delta_from_zero():
    c = compare(snap("a", {"mrr": 0.5}), snap("b", {"mrr": 0.5, "judge_accuracy": 0.8}))
    row = next(r for r in c.rows if r.name == "judge_accuracy")
    assert row.delta is None
    assert "(new)" in row.render()


def test_higher_is_better_is_per_metric_not_global():
    """false_refusals going up is worse, not better."""
    c = compare(
        snap("a", {"false_refusals": 1}, {"retrieval_mode": "dense"}),
        snap("b", {"false_refusals": 3}, {"retrieval_mode": "hybrid"}),
    )
    row = next(r for r in c.rows if r.name == "false_refusals")
    assert row.delta == 2
    assert row.verdict == "worse"
    assert HIGHER_IS_BETTER["false_refusals"] is False


def test_a_metric_improving_reads_as_better():
    c = compare(snap("a", {"mrr": 0.5}), snap("b", {"mrr": 0.9}))
    assert next(r for r in c.rows if r.name == "mrr").verdict == "better"


def test_an_unknown_metric_gets_no_verdict_rather_than_a_guessed_one():
    c = compare(snap("a", {"mystery": 1}), snap("b", {"mystery": 2}))
    row = next(r for r in c.rows if r.name == "mystery")
    assert row.delta == 1
    assert row.verdict == ""


def test_an_unchanged_metric_reads_as_same():
    c = compare(snap("a", {"mrr": 0.5}), snap("b", {"mrr": 0.5}))
    assert next(r for r in c.rows if r.name == "mrr").verdict == "same"


def test_the_comparison_states_what_one_question_is_worth():
    out = compare(snap("a", {"mrr": 0.5}, n=9), snap("b", {"mrr": 0.6}, n=9)).describe()
    assert "one question flipping is 11.1%" in out
    assert "noise, not a trend" in out


def test_a_different_question_count_is_flagged():
    c = compare(snap("a", {"mrr": 0.5}, n=9), snap("b", {"mrr": 0.6}, n=4))
    assert any("different question counts" in w for w in c.warnings)


def test_json_carries_the_deltas_and_the_warnings():
    payload = json.loads(
        comparison_to_json(compare(snap("a", {"mrr": 0.5}), snap("b", {"mrr": 0.9})))
    )
    assert payload["before"] == "a"
    assert payload["metrics"][0]["delta"] == pytest.approx(0.4)
    assert payload["metrics"][0]["verdict"] == "better"
    assert payload["one_question_worth"] == pytest.approx(0.1111, abs=1e-3)
    assert payload["warnings"]


# ---------------------------------------------------------------------------
# Week 8: agent and defence metrics
# ---------------------------------------------------------------------------


def test_the_defence_switches_are_recorded_or_the_ab_reports_no_change(tmp_path):
    """Without these, a defended-vs-undefended comparison shows identical
    settings and warns 'no configuration difference' on its own headline."""
    from rag_app.redteam import undefended

    cfg = make_config(tmp_path)
    before = settings_of(undefended(cfg))
    after = settings_of(cfg)
    assert before["defences_neutralize"] is False
    assert after["defences_neutralize"] is True

    c = compare(snap("undefended", {"injection_success_rate": 0.71}, before),
                snap("defended", {"injection_success_rate": 0.43}, after))
    assert not any("no configuration difference" in w for w in c.warnings)
    assert ("defences_neutralize", False, True) in c.changed_settings


def test_an_injection_metric_reads_lower_as_better():
    c = compare(snap("a", {"injection_success_rate": 0.71}),
                snap("b", {"injection_success_rate": 0.43}))
    row = next(r for r in c.rows if r.name == "injection_success_rate")
    assert row.verdict == "better"


def test_a_denial_count_has_no_direction_and_therefore_no_verdict():
    """More denials is not inherently better: it may mean a nastier corpus or a
    detector false-positiving on a clean one."""
    c = compare(snap("a", {"denied_tool_calls": 0}), snap("b", {"denied_tool_calls": 9}))
    assert next(r for r in c.rows if r.name == "denied_tool_calls").verdict == ""


def test_wall_clock_has_no_direction():
    c = compare(snap("a", {"mean_wall_clock_s": 1.0}), snap("b", {"mean_wall_clock_s": 2.0}))
    assert next(r for r in c.rows if r.name == "mean_wall_clock_s").verdict == ""


def test_wrong_answer_right_path_has_no_direction_and_therefore_no_verdict():
    """A clean route that still failed points at the corpus or the task file,
    not the agent. Driving it to zero is not obviously good."""
    c = compare(snap("a", {"wrong_answer_right_path": 1}),
                snap("b", {"wrong_answer_right_path": 0}))
    assert next(r for r in c.rows if r.name == "wrong_answer_right_path").verdict == ""


def test_a_run_note_records_a_code_change_settings_cannot_see(tmp_path):
    """settings_of records CONFIGURATION; most real interventions are code."""
    cfg = make_config(tmp_path)
    s = make_snapshot("after", cfg, {"mrr": 0.9}, gold("a"), note="added DATA_BOUNDARY")
    assert s.settings["run_note"] == "added DATA_BOUNDARY"

    c = compare(make_snapshot("before", cfg, {"mrr": 0.5}, gold("a")), s)
    assert ("run_note", None, "added DATA_BOUNDARY") in c.changed_settings
    assert c.describe().index("WHAT CHANGED") < c.describe().index("WHAT IT BOUGHT")
