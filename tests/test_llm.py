"""The shared client seam: injection, JSON survival, and spending arithmetic.

Nothing here touches the network. The point of `build_client` returning an
injected client before the key check is that a test never needs a key, and
`openai` is never imported — both are asserted below.
"""

from __future__ import annotations

import sys

import pytest
from conftest import make_config

from rag_app.llm import (
    DEFAULT_HEADERS,
    MISSING_KEY,
    CallBudget,
    build_client,
    chat_messages,
    chat_once,
    estimate_calls,
    extract_json,
)


class FakeClient:
    """Duck-types `client.chat.completions.create`, recording what it was sent."""

    def __init__(self, reply="ok"):
        self.reply = reply
        self.seen = {}
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.seen = kwargs

                class _Msg:
                    content = outer.reply

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def test_an_injected_client_is_returned_untouched_without_a_key(tmp_path):
    """A fake already carries whatever auth it needs; demanding a key is theatre."""
    cfg = make_config(tmp_path, llm_api_key=None)
    fake = FakeClient()
    assert build_client(cfg, cfg.llm, fake) is fake


def test_injecting_a_client_never_imports_the_openai_sdk(tmp_path):
    cfg = make_config(tmp_path, llm_api_key=None)
    sys.modules.pop("openai", None)
    chat_once("sys", "user", cfg.llm, cfg, client=FakeClient())
    assert "openai" not in sys.modules


def test_a_missing_key_names_the_env_file(tmp_path):
    cfg = make_config(tmp_path, llm_api_key=None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_client(cfg, cfg.llm, None)
    assert ".env" in MISSING_KEY


def test_chat_once_sends_a_system_and_a_user_message(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeClient("the reply")
    out = chat_once("SYS", "USER", cfg.llm, cfg, client=fake)
    assert out == "the reply"
    assert fake.seen["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert fake.seen["model"] == cfg.llm.model


def test_the_reply_is_stripped_and_none_becomes_empty(tmp_path):
    cfg = make_config(tmp_path)
    assert chat_once("s", "u", cfg.llm, cfg, client=FakeClient("  padded  ")) == "padded"
    assert chat_once("s", "u", cfg.llm, cfg, client=FakeClient(None)) == ""


def test_temperature_defaults_to_the_llm_config_and_can_be_overridden(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeClient()
    chat_once("s", "u", cfg.llm, cfg, client=fake)
    assert fake.seen["temperature"] == cfg.llm.temperature
    chat_once("s", "u", cfg.llm, cfg, client=fake, temperature=1.0)
    assert fake.seen["temperature"] == 1.0


def test_chat_messages_passes_a_full_message_list_through(tmp_path):
    """The agent's prompt grows, so a system+user pair cannot express it."""
    cfg = make_config(tmp_path)
    fake = FakeClient()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
    chat_messages(msgs, cfg.llm, cfg, client=fake)
    assert fake.seen["messages"] == msgs


def test_the_attribution_headers_are_defined_once():
    assert set(DEFAULT_HEADERS) == {"HTTP-Referer", "X-Title"}


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "correct"}',
        '```json\n{"verdict": "correct"}\n```',
        'Here is my evaluation:\n{"verdict": "correct"}',
        '{"verdict": "correct"}\nHope that helps!',
        '  \n {"verdict": "correct"} \n ',
    ],
)
def test_extract_json_survives_fences_and_preamble(text):
    assert extract_json(text) == {"verdict": "correct"}


def test_extract_json_handles_arrays():
    assert extract_json('prose [{"index": 1}] more') == [{"index": 1}]


@pytest.mark.parametrize("text", ["", "   ", "no json here", "{not: valid}", None])
def test_extract_json_returns_none_rather_than_guessing(text):
    """A half-parsed reply yields a number that looks like a measurement."""
    assert extract_json(text) is None


def test_a_bare_scalar_is_not_json_worth_returning():
    assert extract_json("5") is None


# ---------------------------------------------------------------------------
# Budget and estimate
# ---------------------------------------------------------------------------


def test_a_budget_refuses_the_call_that_would_exceed_it():
    b = CallBudget(limit=2)
    assert b.spend() is True
    assert b.spend() is True
    assert b.spend() is False
    assert b.used == 2  # the refused call reserved nothing


def test_a_zero_limit_is_unlimited():
    b = CallBudget(limit=0)
    for _ in range(50):
        assert b.spend() is True
    assert b.exhausted is False
    assert "no cap" in b.describe()


def test_a_multi_call_reservation_is_all_or_nothing():
    b = CallBudget(limit=5, used=3)
    assert b.spend(5) is False
    assert b.used == 3
    assert b.spend(2) is True


def test_the_call_estimate_matches_the_documented_arithmetic():
    """13 gold questions, 9 expected to reach the LLM, 4 with a reference answer."""
    est = estimate_calls(
        13, generate=True, judge=True, geval=True, ragas=True,
        geval_samples=5, n_answered=9, n_with_reference=4,
    )
    assert est.generation == 13
    assert est.judge == 9
    assert est.geval == 45          # 9 x 5 samples
    assert est.ragas == 40          # 9 x 4 prompts + 4 with a reference
    assert est.total == 107
    assert est.on_judge_model == 94


def test_an_estimate_with_nothing_requested_is_zero():
    est = estimate_calls(13)
    assert est.total == 0
    assert "none" in est.describe()


def test_the_estimate_defaults_to_every_question_reaching_the_llm():
    """The pre-flight number must not be able to surprise you upwards."""
    assert estimate_calls(10, judge=True).judge == 10


def test_the_estimate_names_both_models_so_the_expensive_half_is_visible():
    est = estimate_calls(
        5, generate=True, judge=True, n_answered=5,
        gen_model="cheap/model", judge_model="expensive/model",
    )
    out = est.describe()
    assert "cheap/model" in out and "expensive/model" in out
    assert "~10 LLM calls" in out
