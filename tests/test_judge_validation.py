"""Measuring the measurer: trace loading, Cohen's kappa, and small-n honesty."""

from __future__ import annotations

import json
import math

import pytest
from conftest import FakeJudge, make_config, verdict_json

from rag_app.error_analysis import HumanLabel
from rag_app.judge import JUDGE_TRACE_SYSTEM
from rag_app.judge_validation import (
    Trace,
    ValidationReport,
    ValidationRow,
    agreement_of,
    load_traces,
    validate_judge,
    validation_to_json,
)


def lab(id, label, note="") -> HumanLabel:
    return HumanLabel(id=id, label=label, note=note)


def write_traces(tmp_path, rows, name="t.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def test_traces_load_into_scoredchunks_the_ragas_metrics_can_reuse(tmp_path):
    """Real ScoredChunks mean RAGAS runs on a trace with no adapter."""
    path = write_traces(tmp_path, [{
        "id": "cur-001", "question": "why?", "answer": "because",
        "gate": "answered", "used_llm": True, "best_score": 0.9,
        "sources": ["doc.pdf"],
        "reranked": [{"score": 0.9, "source": "doc.pdf", "text": "body"}],
    }])
    [t] = load_traces(path)
    assert t.id == "cur-001"
    assert t.contexts[0].chunk.source == "doc.pdf"
    assert t.contexts[0].chunk.text == "body"
    assert t.contexts[0].score == pytest.approx(0.9)

    # and it is directly usable by a RAGAS metric
    from rag_app.ragas_metrics import context_precision

    cfg = make_config(tmp_path)
    cp = context_precision(
        t.question, t.contexts, cfg,
        judge_fn=FakeJudge(json.dumps([{"position": 1, "useful": 1}])),
    )
    assert cp.score == pytest.approx(1.0)


def test_a_trace_without_contexts_still_loads(tmp_path):
    path = write_traces(tmp_path, [{"id": "a", "question": "q", "answer": "x"}])
    assert load_traces(path)[0].contexts == []


def test_a_malformed_trace_line_names_its_line(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"id":"a"}\nnope\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_traces(path)


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------


def test_raw_agreement_is_the_diagonal_over_n():
    a = agreement_of([("correct", "correct"), ("correct", "incorrect")])
    assert a.n == 2
    assert a.raw == pytest.approx(0.5)


def test_cohens_kappa_matches_a_hand_computed_value():
    """human [c,c,c,i] vs judge [c,c,i,i]:
        raw      = 3/4 = 0.75
        expected = (3/4)(2/4) + (0)(0) + (1/4)(2/4) = 0.375 + 0.125 = 0.5
        kappa    = (0.75 - 0.5) / (1 - 0.5) = 0.5
    """
    pairs = [
        ("correct", "correct"),
        ("correct", "correct"),
        ("correct", "incorrect"),
        ("incorrect", "incorrect"),
    ]
    a = agreement_of(pairs)
    assert a.raw == pytest.approx(0.75)
    assert a.expected == pytest.approx(0.5)
    assert a.kappa == pytest.approx(0.5)


def test_perfect_agreement_over_two_labels_is_kappa_one():
    a = agreement_of([("correct", "correct"), ("incorrect", "incorrect")])
    assert a.kappa == pytest.approx(1.0)


def test_kappa_is_zero_when_agreement_equals_chance():
    pairs = [
        ("correct", "correct"), ("correct", "incorrect"),
        ("incorrect", "correct"), ("incorrect", "incorrect"),
    ]
    assert agreement_of(pairs).kappa == pytest.approx(0.0)


def test_kappa_can_go_negative_when_raters_disagree_worse_than_chance():
    pairs = [("correct", "incorrect"), ("incorrect", "correct")]
    assert agreement_of(pairs).kappa < 0


def test_kappa_is_undefined_when_both_raters_used_one_label():
    """The likeliest real outcome on 20 traces from a working system."""
    a = agreement_of([("correct", "correct")] * 20)
    assert a.raw == pytest.approx(1.0)
    assert math.isnan(a.kappa)
    out = a.describe()
    assert "undefined" in out
    assert "NOT evidence the judge works" in out


def test_a_constant_judge_scores_high_raw_and_near_zero_kappa():
    """17 correct, 3 incorrect; the judge says correct every time."""
    pairs = [("correct", "correct")] * 17 + [("incorrect", "correct")] * 3
    a = agreement_of(pairs)
    assert a.raw == pytest.approx(0.85)
    assert a.kappa == pytest.approx(0.0, abs=1e-9)


def test_the_confusion_matrix_is_printed():
    out = agreement_of([("correct", "partial")]).describe()
    assert "human down, judge across" in out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def make_report(pairs, notes=None) -> ValidationReport:
    from rag_app.judge import Verdict

    rows = []
    for i, (human, judge) in enumerate(pairs):
        t = Trace(id=f"cur-{i:03d}", question=f"q{i}", answer="a")
        rows.append(
            ValidationRow(t, lab(t.id, human, (notes or {}).get(t.id, "")),
                          Verdict(judge, "the judge reasoned"))
        )
    return ValidationReport(rows=rows, judge_model="big/model")


def test_validation_uses_the_trace_prompt_not_the_gold_prompt(tmp_path):
    """The human had no reference; a gold-referenced judge would be a different task."""
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json("correct"))
    traces = [Trace(id="a", question="q", answer="x")]
    validate_judge(traces, [lab("a", "correct")], cfg, judge_fn=fake)
    assert fake.systems[0] == JUDGE_TRACE_SYSTEM


def test_unscored_rows_are_reported_separately_not_as_disagreement(tmp_path):
    cfg = make_config(tmp_path)
    traces = [Trace(id="a", question="q", answer="x")]
    report = validate_judge(
        traces, [lab("a", "correct")], cfg, judge_fn=FakeJudge("not json at all")
    )
    assert report.rows == []
    assert len(report.unscored) == 1
    assert "excluded from the matrix" in report.describe()


def test_a_label_for_a_missing_trace_is_reported_not_dropped(tmp_path):
    cfg = make_config(tmp_path)
    traces = [Trace(id="a", question="q", answer="x")]
    report = validate_judge(
        traces, [lab("a", "correct"), lab("ghost", "correct")], cfg,
        judge_fn=FakeJudge(verdict_json()),
    )
    assert report.missing_ids == ["ghost"]
    assert "Labelled but not in the trace file" in report.describe()


def test_an_unlabelled_trace_is_reported(tmp_path):
    cfg = make_config(tmp_path)
    traces = [Trace(id="a", question="q", answer="x"), Trace(id="b", question="q", answer="y")]
    report = validate_judge(traces, [lab("a", "correct")], cfg, judge_fn=FakeJudge(verdict_json()))
    assert report.unlabelled_ids == ["b"]


def test_the_report_states_what_twenty_labels_cannot_support():
    out = make_report([("correct", "correct")] * 20).describe()
    assert "CANNOT support" in out
    assert "the deliverable is the list below, not the number" in out


def test_the_disagreement_list_names_both_labels_and_both_reasons():
    report = make_report(
        [("correct", "incorrect")], notes={"cur-000": "the answer was actually fine"}
    )
    out = report.describe()
    assert "human=correct" in out
    assert "judge=incorrect" in out
    assert "the answer was actually fine" in out
    assert "the judge reasoned" in out


def test_agreeing_rows_are_not_listed_as_disagreements():
    assert make_report([("correct", "correct")]).disagreements == []


def test_json_reports_kappa_as_null_when_undefined():
    payload = json.loads(validation_to_json(make_report([("correct", "correct")] * 5)))
    assert payload["cohens_kappa"] is None
    assert payload["kappa_undefined"] is True
    assert payload["n"] == 5


def test_json_carries_the_disagreements():
    payload = json.loads(validation_to_json(make_report([("correct", "incorrect")])))
    assert payload["disagreements"][0]["human"] == "correct"
    assert payload["disagreements"][0]["judge"] == "incorrect"
