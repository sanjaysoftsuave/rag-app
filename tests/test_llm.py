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

    def __init__(self, reply="ok", usage=None):
        self.reply = reply
        self.usage = usage  # Week 10: an object with .prompt_tokens/.completion_tokens, or None
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
                    usage = outer.usage

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


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


# ---------------------------------------------------------------------------
# Week 10: real tokens, declared cost
# ---------------------------------------------------------------------------


def test_usage_accumulates_across_models_separately():
    from rag_app.llm import Usage

    u = Usage()
    u.add("model-a", 100, 10)
    u.add("model-a", 50, 5)
    u.add("model-b", 1, 1)
    assert u.by_model == {"model-a": (150, 15), "model-b": (1, 1)}
    assert u.prompt_tokens == 151
    assert u.completion_tokens == 16
    assert u.total_tokens == 167
    assert u.calls == 3


def test_price_usage_is_none_when_any_model_used_is_unlisted():
    from rag_app.llm import Usage, price_usage

    u = Usage()
    u.add("priced/model", 1_000_000, 1_000_000)
    assert price_usage(u, {"priced/model": (1.0, 2.0)}) == pytest.approx(3.0)
    u.add("unpriced/model", 1, 1)
    assert price_usage(u, {"priced/model": (1.0, 2.0)}) is None


def test_usage_summary_is_none_without_a_meter_or_with_an_empty_one(tmp_path):
    from rag_app.llm import Usage, usage_summary

    cfg = make_config(tmp_path)
    assert usage_summary(None, cfg) is None
    assert usage_summary(Usage(), cfg) is None


def test_usage_summary_reports_tokens_and_cost(tmp_path):
    from rag_app.config import EvaluationConfig
    from rag_app.llm import Usage, usage_summary

    cfg = make_config(tmp_path, evaluation=EvaluationConfig(token_prices={"x": (1.0, 2.0)}))
    u = Usage()
    u.add("x", 1_000_000, 500_000)
    summary = usage_summary(u, cfg)
    assert summary == {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 500_000,
        "total_tokens": 1_500_000,
        "cost_usd": pytest.approx(2.0),
    }


def test_chat_messages_records_real_usage_when_a_meter_is_given(tmp_path):
    from rag_app.llm import Usage

    cfg = make_config(tmp_path)
    fake = FakeClient("ok", usage=_FakeUsage(prompt_tokens=42, completion_tokens=7))
    u = Usage()
    chat_messages([{"role": "user", "content": "hi"}], cfg.llm, cfg, client=fake, usage=u)
    assert u.by_model == {cfg.llm.model: (42, 7)}


def test_generate_answer_threads_usage_to_chat_messages(tmp_path):
    """Week 11: pipeline.ask()'s only token-cost gap, closed the same way
    agent.py's was in Week 10 — one additive kwarg, nothing existing moves."""
    from rag_app.generate import generate_answer
    from rag_app.llm import Usage

    cfg = make_config(tmp_path)
    fake = FakeClient("the answer", usage=_FakeUsage(prompt_tokens=10, completion_tokens=5))
    u = Usage()
    text = generate_answer("q?", [], cfg, client=fake, usage=u)
    assert text == "the answer"
    assert u.by_model == {cfg.llm.model: (10, 5)}


def test_generate_answer_with_no_meter_is_unaffected(tmp_path):
    from rag_app.generate import generate_answer

    cfg = make_config(tmp_path)
    fake = FakeClient("the answer")
    assert generate_answer("q?", [], cfg, client=fake) == "the answer"


def test_chat_messages_with_no_meter_never_touches_usage(tmp_path):
    """The overwhelming majority of callers pass nothing; must stay a no-op."""
    cfg = make_config(tmp_path)
    fake = FakeClient("ok", usage=_FakeUsage(prompt_tokens=42, completion_tokens=7))
    out = chat_messages([{"role": "user", "content": "hi"}], cfg.llm, cfg, client=fake)
    assert out == "ok"


def test_a_client_with_no_usage_attribute_degrades_silently(tmp_path):
    """Some fakes (and some providers) never set .usage; must not raise."""
    from rag_app.llm import Usage

    cfg = make_config(tmp_path)
    fake = FakeClient("ok")  # usage=None
    u = Usage()
    chat_messages([{"role": "user", "content": "hi"}], cfg.llm, cfg, client=fake, usage=u)
    assert u.calls == 0
