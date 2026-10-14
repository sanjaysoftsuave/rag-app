"""`report`: numbers rendered FROM snapshots, each saying where it came from."""

from __future__ import annotations

from rag_app.before_after import Snapshot
from rag_app.report import Located, kind_of, not_yet_measured, provenance, render_report


def snap(label, metrics, n=9, note="") -> Snapshot:
    return Snapshot(
        label=label, created="2026-10-01T00:00:00+00:00",
        settings={"run_note": note} if note else {},
        metrics=metrics, n_questions=n, n_answerable=n, gold_fingerprint="abc",
    )


def loc(s, corpus="your corpus") -> Located:
    return Located(s, corpus)


def test_kind_is_inferred_from_the_metrics_a_command_writes():
    assert kind_of(snap("e", {"hit_rate_at_k": 1.0})) == "eval"
    assert kind_of(snap("a", {"task_success": 0.8})) == "agent"
    assert kind_of(snap("r", {"injection_success_rate": 0.4})) == "redteam"
    assert kind_of(snap("c", {"cache_hit_rate": 0.5})) == "requests"


def test_a_scripted_red_team_run_is_labelled_not_a_real_model():
    """The exact claim the README used to make without this label."""
    s = snap("rt-after", {"injection_success_rate": 0.43, "generated": False})
    assert "NOT A REAL MODEL" in provenance(s)
    assert "NOT A REAL MODEL" in render_report([loc(s)])


def test_a_retrieval_only_eval_is_real_for_retrieval():
    """eval without --generate is not a stand-in: its retrieval metrics are real."""
    s = snap("e", {"hit_rate_at_k": 1.0, "generated": False})
    assert "retrieval only" in provenance(s)
    assert "NOT A REAL MODEL" not in provenance(s)


def test_a_snapshot_without_the_flag_says_its_provenance_is_unknown():
    assert "UNKNOWN" in provenance(snap("old", {"task_success": 0.8}))


def test_known_pairs_embed_the_before_after_diff_verbatim():
    a = snap("cache-cold", {"cache_hit_rate": 0.0, "mean_cost_usd": 0.002, "generated": True})
    b = snap("cache-warm", {"cache_hit_rate": 1.0, "mean_cost_usd": 0.0, "generated": True})
    out = render_report([loc(a), loc(b)])
    assert "Answer cache: empty vs populated" in out
    assert "WHAT IT BOUGHT" in out  # before_after.Comparison.describe(), not a second diff


def test_the_noise_floor_is_stated_for_every_snapshot():
    out = render_report([loc(snap("e", {"hit_rate_at_k": 1.0, "generated": True}, n=9))])
    assert "11.1%" in out


def test_gaps_are_listed_rather_than_left_silent():
    gaps = not_yet_measured([])
    assert any("Answer accuracy" in g for g in gaps)
    assert any("Real cost per request" in g for g in gaps)
    assert any("REAL model" in g for g in gaps)


def test_a_real_measurement_removes_its_gap():
    real_eval = snap("main-v1", {"hit_rate_at_k": 1.0, "answer_accuracy": 0.9, "generated": True})
    gaps = not_yet_measured([loc(real_eval)])
    assert not any("Answer accuracy" in g for g in gaps)


def test_an_empty_report_still_renders_and_says_how_to_start():
    out = render_report([])
    assert "No snapshots yet" in out
    assert "## Not yet measured" in out
