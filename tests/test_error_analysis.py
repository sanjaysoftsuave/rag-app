"""Week 5 scaffolding: the blank sheet, the label file, and the ranking.

Everything here is a pure function over data a human typed, so there is no
store, no embedder and no LLM anywhere in this file.
"""

from __future__ import annotations

import json

import pytest
import yaml

from rag_app.error_analysis import (
    LABELS,
    SEVERITIES,
    HumanLabel,
    TraceStub,
    build_taxonomy,
    load_labels,
    load_trace_stubs,
    parse_labels,
    taxonomy_to_json,
    write_labels_template,
)


def lab(id, label="incorrect", category="", severity=0, note="") -> HumanLabel:
    return HumanLabel(id=id, label=label, category=category, severity=severity, note=note)


def write_yaml(tmp_path, rows, name="labels.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Trace stubs
# ---------------------------------------------------------------------------


def test_trace_stubs_read_id_and_question_from_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"id": "cur-001", "question": "why?", "answer": "because"})
        + "\n"
        + json.dumps({"id": "cur-002", "question": "how?"})
        + "\n",
        encoding="utf-8",
    )
    stubs = load_trace_stubs(path)
    assert [s.id for s in stubs] == ["cur-001", "cur-002"]
    assert stubs[0].question == "why?"


def test_a_blank_line_in_the_trace_file_is_skipped(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"id": "a", "question": "q"}) + "\n\n", encoding="utf-8")
    assert len(load_trace_stubs(path)) == 1


def test_a_malformed_trace_line_names_its_line_number(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_trace_stubs(path)


def test_a_missing_trace_file_says_to_generate_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trace file"):
        load_trace_stubs(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# The blank sheet
# ---------------------------------------------------------------------------


def test_the_template_has_one_row_per_trace_and_carries_the_question(tmp_path):
    path = tmp_path / "labels.yaml"
    write_labels_template([TraceStub("cur-001", "why is it slow?")], path)
    text = path.read_text(encoding="utf-8")
    assert "- id: cur-001" in text
    assert "why is it slow?" in text
    assert "label:" in text and "category:" in text and "severity:" in text and "note:" in text


def test_the_template_carries_the_severity_rubric_so_judgements_stay_on_one_scale(tmp_path):
    path = tmp_path / "labels.yaml"
    write_labels_template([TraceStub("a", "q")], path)
    text = path.read_text(encoding="utf-8")
    for level in SEVERITIES:
        assert f"     {level}  " in text
    assert "harmful" in text and "useless" in text and "annoying" in text


def test_the_template_tells_you_to_write_the_note_before_the_category(tmp_path):
    """The whole method is notes-first. If the sheet does not say so, it will not happen."""
    path = tmp_path / "labels.yaml"
    write_labels_template([TraceStub("a", "q")], path)
    text = path.read_text(encoding="utf-8")
    assert "note` FIRST" in text
    assert "preconceptions" in text


def test_the_template_refuses_to_overwrite_an_hour_of_reading(tmp_path):
    path = tmp_path / "labels.yaml"
    path.write_text("- id: a\n  label: correct\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_labels_template([TraceStub("a", "q")], path)
    # and the existing content is untouched
    assert path.read_text(encoding="utf-8").startswith("- id: a")


def test_a_written_template_is_valid_yaml_that_parse_labels_rejects_as_unfilled(tmp_path):
    """The template must round-trip through YAML, and must not look finished."""
    path = tmp_path / "labels.yaml"
    write_labels_template([TraceStub("a", "q"), TraceStub("b", "q2")], path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert [r["id"] for r in raw] == ["a", "b"]
    with pytest.raises(ValueError, match="2 of 2 rows have no 'label'"):
        parse_labels(raw)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_labels_round_trip_through_a_file(tmp_path):
    path = write_yaml(
        tmp_path,
        [{"id": "a", "label": "incorrect", "category": "bad cite", "severity": 3, "note": "n"}],
    )
    labels = load_labels(path)
    assert labels == [lab("a", "incorrect", "bad cite", 3, "n")]


def test_severity_defaults_to_zero_when_omitted(tmp_path):
    labels = parse_labels([{"id": "a", "label": "correct"}])
    assert labels[0].severity == 0
    assert labels[0].category == ""


def test_an_unknown_label_names_the_closed_set(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        parse_labels([{"id": "a", "label": "wrongish"}])


def test_the_label_set_is_closed_because_the_judge_is_scored_against_it():
    with pytest.raises(ValueError, match="LLM judge"):
        parse_labels([{"id": "a", "label": "meh"}])


def test_a_severity_outside_the_rubric_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        parse_labels([{"id": "a", "label": "incorrect", "severity": 7}])


def test_a_non_numeric_severity_is_rejected():
    with pytest.raises(ValueError, match="not a whole number"):
        parse_labels([{"id": "a", "label": "incorrect", "severity": "bad"}])


def test_a_duplicate_id_is_rejected_because_it_would_double_a_weight():
    with pytest.raises(ValueError, match="repeats id"):
        parse_labels([{"id": "a", "label": "correct"}, {"id": "a", "label": "incorrect"}])


def test_an_entry_without_an_id_names_its_position():
    with pytest.raises(ValueError, match="entry 2 has no 'id'"):
        parse_labels([{"id": "a", "label": "correct"}, {"label": "correct"}])


def test_a_partly_filled_sheet_is_refused_with_a_count_not_a_first_failure():
    """Ranking half a sample would rank whichever traces were read first."""
    rows = [{"id": f"c{i}", "label": "correct"} for i in range(3)]
    rows += [{"id": f"b{i}"} for i in range(8)]
    with pytest.raises(ValueError, match="8 of 11 rows have no 'label'") as exc:
        parse_labels(rows)
    assert "b0" in str(exc.value)
    assert "..." in str(exc.value)


def test_an_empty_labels_file_points_at_the_init_command():
    with pytest.raises(ValueError, match="codes --init"):
        parse_labels(None)


def test_a_missing_labels_file_points_at_the_init_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="codes --init"):
        load_labels(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# The taxonomy
# ---------------------------------------------------------------------------


def test_the_weight_is_count_times_mean_severity():
    tax = build_taxonomy(
        [
            lab("a", category="bad cite", severity=3),
            lab("b", category="bad cite", severity=1),
        ]
    )
    g = tax.groups[0]
    assert g.count == 2
    assert g.mean_severity == pytest.approx(2.0)
    assert g.weight == pytest.approx(4.0)  # 2 x 2.0


def test_groups_rank_by_weight_not_by_frequency_alone():
    """Three annoyances must not outrank two catastrophes."""
    labels = [lab(f"n{i}", category="noisy", severity=1) for i in range(3)]
    labels += [lab(f"h{i}", category="harmful", severity=3) for i in range(2)]
    tax = build_taxonomy(labels)
    assert [g.category for g in tax.groups] == ["harmful", "noisy"]  # 6.0 vs 3.0
    assert tax.top.category == "harmful"


def test_frequency_can_still_win_when_severities_match():
    labels = [lab(f"a{i}", category="common", severity=2) for i in range(4)]
    labels += [lab("b", category="rare", severity=2)]
    tax = build_taxonomy(labels)
    assert [g.category for g in tax.groups] == ["common", "rare"]


def test_an_uncategorised_trace_is_counted_but_joins_no_group():
    """This is a taxonomy of problems, not of traces."""
    tax = build_taxonomy(
        [lab("a", label="correct"), lab("b", category="bad cite", severity=2)]
    )
    assert tax.n_labelled == 2
    assert tax.n_uncategorised == 1
    assert len(tax.groups) == 1


def test_a_zero_severity_category_sorts_to_the_bottom():
    tax = build_taxonomy(
        [lab("a", label="correct", category="clean lookup", severity=0),
         lab("b", category="real problem", severity=1)]
    )
    assert [g.category for g in tax.groups] == ["real problem", "clean lookup"]


def test_ordering_is_stable_for_tied_groups():
    labels = [lab("a", category="zebra", severity=2), lab("b", category="alpha", severity=2)]
    first = [g.category for g in build_taxonomy(labels).groups]
    second = [g.category for g in build_taxonomy(list(reversed(labels))).groups]
    assert first == second == ["alpha", "zebra"]


def test_label_counts_cover_every_legal_label():
    tax = build_taxonomy(
        [lab("a", label="correct"), lab("b", label="partial"), lab("c", label="incorrect")]
    )
    for name in LABELS:
        assert tax.label_counts[name] == 1


# ---------------------------------------------------------------------------
# describe() -- the deliverable a human reads
# ---------------------------------------------------------------------------


def test_describe_shows_count_and_severity_next_to_the_weight():
    """Collapsing the two into one number would hide the trade being made."""
    out = build_taxonomy([lab("a", category="bad cite", severity=3)]).describe()
    assert "weight" in out and "count" in out and "mean sev" in out
    assert "count x mean severity" in out


def test_describe_names_the_fix_target_and_asks_for_a_prediction_first():
    out = build_taxonomy(
        [lab("a", category="bad cite", severity=3, note="cited a source it did not use")]
    ).describe()
    assert "FIX TARGET: bad cite" in out
    assert "What I expect it to fix" in out
    assert "prediction written afterwards never is" in out


def test_describe_lists_every_trace_id_with_its_note():
    out = build_taxonomy(
        [lab("cur-007", category="bad cite", severity=3, note="invented a label")]
    ).describe()
    assert "cur-007" in out
    assert "invented a label" in out


def test_describe_with_no_categories_says_what_to_do_next():
    out = build_taxonomy([lab("a", label="correct")]).describe()
    assert "No categories yet" in out


def test_json_carries_the_ranking_and_the_fix_target():
    tax = build_taxonomy(
        [lab("a", category="bad cite", severity=3), lab("b", category="noisy", severity=1)]
    )
    payload = json.loads(taxonomy_to_json(tax))
    assert payload["fix_target"] == "bad cite"
    assert payload["groups"][0]["rank"] == 1
    assert payload["groups"][0]["weight"] == 3.0
    assert payload["groups"][0]["trace_ids"] == ["a"]
    assert payload["n_labelled"] == 2


def test_json_fix_target_is_none_when_nothing_is_categorised():
    payload = json.loads(taxonomy_to_json(build_taxonomy([lab("a", label="correct")])))
    assert payload["fix_target"] is None
