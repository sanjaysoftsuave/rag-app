"""Parsing one model turn.

Every test here is a format a real model actually emits. The parser is tolerant
about presentation and strict about structure, and the two strict rules —
no implicit final answer, no model-written observations — are grounding
guarantees, not style preferences.
"""

from __future__ import annotations

import pytest

from rag_app.agent import parse_action


def test_parses_thought_action_and_input():
    a = parse_action(
        "Thought: I should look this up\n"
        "Action: search_documents\n"
        "Action Input: password reset expiry"
    )
    assert a.thought == "I should look this up"
    assert a.tool == "search_documents"
    assert a.tool_input == "password reset expiry"
    assert a.is_final is False
    assert a.parse_error == ""


def test_final_answer_is_recognised():
    a = parse_action(
        "Thought: I have it\nAction: final_answer\nAction Input: The link lasts 60 minutes."
    )
    assert a.is_final is True
    assert a.tool_input == "The link lasts 60 minutes."


def test_action_input_may_be_multiline():
    a = parse_action(
        "Thought: done\n"
        "Action: final_answer\n"
        "Action Input: First line.\nSecond line.\nThird line."
    )
    assert a.tool_input == "First line.\nSecond line.\nThird line."


def test_markdown_bold_and_case_variations_are_tolerated():
    a = parse_action(
        "**Thought:** thinking\n**ACTION:** search_documents\n**Action Input:** query"
    )
    assert a.tool == "search_documents"
    assert a.tool_input == "query"


def test_list_markers_are_tolerated():
    a = parse_action("- Thought: t\n- Action: list_sources\n- Action Input: -")
    assert a.tool == "list_sources"


def test_code_fences_are_stripped():
    a = parse_action(
        "```\nThought: t\nAction: read_source\nAction Input: handbook.pdf\n```"
    )
    assert a.tool == "read_source"
    assert a.tool_input == "handbook.pdf"


def test_a_json_action_input_is_flattened_when_unambiguous():
    a = parse_action(
        'Thought: t\nAction: search_documents\nAction Input: {"query": "refund window"}'
    )
    assert a.tool_input == "refund window"


def test_an_ambiguous_json_input_is_left_for_the_tool_to_reject():
    a = parse_action(
        'Thought: t\nAction: search_documents\nAction Input: {"a": "1", "b": "2"}'
    )
    assert a.tool_input.startswith("{")


def test_a_trailing_period_on_the_action_is_tolerated():
    assert parse_action("Thought: t\nAction: list_sources.\nAction Input: -").tool == (
        "list_sources"
    )


def test_a_hallucinated_observation_is_discarded():
    """Models pre-fill fake observations. Keeping them lets the model invent its
    own tool results and then reason over them as if they were real."""
    a = parse_action(
        "Thought: t\n"
        "Action: search_documents\n"
        "Action Input: refunds\n"
        "Observation: The refund window is 900 days.\n"
        "Thought: so it is 900 days\n"
        "Action: final_answer\n"
        "Action Input: 900 days"
    )
    assert a.tool == "search_documents"
    assert a.is_final is False
    assert "900" not in a.tool_input


def test_a_missing_action_line_is_a_parse_failure_not_a_final_answer():
    """Treating loose prose as an answer is how ungrounded text passes the gate."""
    a = parse_action("The refund window is 30 days, based on what I know.")
    assert a.parse_error == "no 'Action:' line"
    assert a.is_final is False
    assert a.tool == ""


def test_a_thought_with_no_action_is_still_a_parse_failure():
    a = parse_action("Thought: I think the answer is 30 days.")
    assert a.parse_error
    assert a.thought == "I think the answer is 30 days."


def test_empty_output_is_a_parse_failure():
    assert parse_action("").parse_error
    assert parse_action("   ").parse_error


@pytest.mark.parametrize("missing_input", [
    "Thought: t\nAction: list_sources",
    "Thought: t\nAction: list_sources\nAction Input:",
])
def test_a_missing_action_input_parses_as_an_empty_string(missing_input):
    """The tool decides whether empty input is acceptable, not the parser."""
    a = parse_action(missing_input)
    assert a.tool == "list_sources"
    assert a.tool_input == ""
