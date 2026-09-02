"""Query transforms: rewrite and HyDE.

Both add an LLM call before retrieval, so every branch here is exercised
through the `transform_fn` seam — no network, same as `ask(generate_fn=...)`.
"""

from __future__ import annotations

import pytest

from rag_app.rewrite import (
    HYDE_SYSTEM,
    REWRITE_SYSTEM,
    QueryTransform,
    transform_query,
)

from conftest import make_config


def fake(reply: str):
    """A transform_fn returning `reply`, recording the system prompt it saw."""
    seen: dict[str, str] = {}

    def _call(system, question, cfg):
        seen["system"] = system
        seen["question"] = question
        return reply

    _call.seen = seen  # type: ignore[attr-defined]
    return _call


def boom(system, question, cfg):
    raise RuntimeError("the API is down")


# --- off -------------------------------------------------------------------


def test_off_is_a_passthrough_and_costs_nothing(tmp_path):
    called = []

    def _never(system, question, cfg):
        called.append(question)
        return "should not happen"

    t = transform_query("original question", make_config(tmp_path), transform_fn=_never)
    assert t.transformed == "original question"
    assert t.mode == "off"
    assert not t.changed
    assert called == [], "the LLM must not be called when the transform is off"


# --- rewrite / hyde --------------------------------------------------------


def test_rewrite_uses_the_rewrite_prompt_and_returns_the_restatement(tmp_path):
    f = fake("password reset link validity period")
    t = transform_query("how long does the link last?", make_config(tmp_path),
                        mode="rewrite", transform_fn=f)
    assert t.transformed == "password reset link validity period"
    assert t.changed and not t.failed
    assert f.seen["system"] == REWRITE_SYSTEM


def test_hyde_uses_the_hyde_prompt(tmp_path):
    f = fake("Password reset links expire after 60 minutes.")
    t = transform_query("how long does the link last?", make_config(tmp_path),
                        mode="hyde", transform_fn=f)
    assert t.transformed.startswith("Password reset links")
    assert f.seen["system"] == HYDE_SYSTEM


def test_the_original_question_is_always_preserved(tmp_path):
    """Retrieval searches for the transform; generation still answers what was
    actually asked. Losing the original would let a bad rewrite silently
    replace the user's question."""
    t = transform_query("what is the refund window?", make_config(tmp_path),
                        mode="hyde", transform_fn=fake("Refunds are processed in 5 days."))
    assert t.original == "what is the refund window?"
    assert t.transformed != t.original


# --- failure policy --------------------------------------------------------


def test_a_failed_transform_falls_back_to_the_original(tmp_path):
    """A degraded search beats no answer — but it must be visible."""
    t = transform_query("original", make_config(tmp_path), mode="rewrite", transform_fn=boom)
    assert t.transformed == "original"
    assert t.failed
    assert "failed" in t.describe()


@pytest.mark.parametrize("reply", ["", "   ", "\n"])
def test_an_empty_transform_falls_back_too(tmp_path, reply):
    t = transform_query("original", make_config(tmp_path), mode="hyde",
                        transform_fn=fake(reply))
    assert t.transformed == "original"
    assert t.failed


def test_a_missing_api_key_does_not_raise(tmp_path):
    """Enabling a transform without a key must degrade, not crash the app."""
    cfg = make_config(tmp_path, llm_api_key=None)
    t = transform_query("original", cfg, mode="rewrite")
    assert t.transformed == "original"
    assert t.failed


def test_an_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="query_mode"):
        transform_query("q", make_config(tmp_path), mode="magic")


# --- describe --------------------------------------------------------------


def test_describe_distinguishes_success_from_fallback():
    assert QueryTransform("q", "q", "off").describe() == "off"
    assert QueryTransform("q", "better q", "rewrite").describe() == "rewrite"
    assert "failed" in QueryTransform("q", "q", "hyde", failed=True).describe()
