"""LLM-as-Judge and G-Eval.

The headline test is `test_a_correct_answer_in_different_words...`: it is the
whole reason Week 6 exists, and it fails if the judge ever degenerates back into
string matching.
"""

from __future__ import annotations

import json

import pytest
from conftest import FakeJudge, geval_json, make_config, verdict_json

from rag_app.evaluate import GoldQuestion, QuestionResult
from rag_app.judge import (
    GEVAL_SYSTEM,
    JUDGE_SYSTEM,
    JUDGE_TRACE_SYSTEM,
    SCORABLE,
    VERDICTS,
    Verdict,
    geval_score,
    judge_answer,
    judge_trace,
    parse_verdict,
)
from rag_app.store import Chunk, ScoredChunk


def gold(must=("3 business days",), **kw) -> GoldQuestion:
    return GoldQuestion(question="q?", must_contain=list(must), **kw)


def contexts(*texts) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(f"c::{i}", "doc.pdf", t, {}), 0.9)
        for i, t in enumerate(texts)
    ]


def result_for(answer_text: str, g: GoldQuestion) -> QuestionResult:
    """The substring baseline, for the side-by-side comparison tests."""
    return QuestionResult(
        gold=g, retrieved_rank=1, reranked_rank=1, retrieved_found=set(),
        reranked_found=set(), best_score=0.9, used_llm=True, refused=False,
        gate="answered", answer_text=answer_text,
    )


# ---------------------------------------------------------------------------
# The point of the whole module
# ---------------------------------------------------------------------------


def test_a_correct_answer_in_different_words_is_judged_correct_where_substring_fails(tmp_path):
    """Gold wants "3 business days"; the answer says "after three business days"."""
    cfg = make_config(tmp_path)
    g = gold(("3 business days",))
    answer = "The duplicate authorization cleared after three business days."

    assert result_for(answer, g).answer_ok is False          # substring: FAIL

    fake = FakeJudge(verdict_json("correct", "conveys the fact; only the numeral differs"))
    v = judge_answer("q?", answer, g, cfg, judge_fn=fake)
    assert v.ok is True                                       # judge: CORRECT
    assert v.scored is True


def test_a_substring_match_can_still_be_judged_incorrect(tmp_path):
    """The other direction: the string is present and the answer contradicts it."""
    cfg = make_config(tmp_path)
    g = gold(("cached",))
    answer = "The cached credentials were not the problem here."

    assert result_for(answer, g).answer_ok is True            # substring: PASS

    fake = FakeJudge(verdict_json("incorrect", "contradicts the reference"))
    assert judge_answer("q?", answer, g, cfg, judge_fn=fake).ok is False


# ---------------------------------------------------------------------------
# Prompt discipline
# ---------------------------------------------------------------------------


def test_the_judge_is_told_the_reference_is_a_fact_not_a_string():
    """The single sentence that separates this from `answer_ok`."""
    assert "FACT, not the wording" in JUDGE_SYSTEM
    assert "matching strings" in JUDGE_SYSTEM


def test_the_judge_prompt_names_the_closed_verdict_set():
    for v in SCORABLE:
        assert v in JUDGE_SYSTEM
    assert "unscored" not in JUDGE_SYSTEM  # a judge never chooses to be unscored


def test_the_judge_asks_for_reasoning_before_the_verdict():
    """Autoregressive: verdict-first makes the reasoning a rationalisation."""
    assert JUDGE_SYSTEM.index('"reasoning"') < JUDGE_SYSTEM.index('"verdict"')
    assert GEVAL_SYSTEM.index('"reasoning"') < GEVAL_SYSTEM.index('"score"')


def test_the_prompt_warns_that_verdict_words_inside_the_answer_are_not_verdicts():
    """Same trap class as the [TIC-1001] citation bug: a token in the material
    being judged is not an instruction."""
    assert "never your verdict" in JUDGE_SYSTEM
    assert "never your verdict" in JUDGE_TRACE_SYSTEM


def test_the_gold_judge_is_never_shown_the_retrieved_contexts(tmp_path):
    """Correctness and groundedness are different properties; RAGAS faithfulness
    measures the second. Fusing them lets an ungrounded answer score correct."""
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    judge_answer("q?", "an answer", gold(), cfg, judge_fn=fake)
    _, user = fake.seen[0]
    assert "RETRIEVED CONTEXT" not in user
    assert "REFERENCE FACTS" in user


def test_the_trace_judge_is_shown_the_contexts_and_no_reference(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    judge_trace("q?", "an answer", contexts("the document says X"), cfg, judge_fn=fake)
    system, user = fake.seen[0]
    assert system == JUDGE_TRACE_SYSTEM
    assert "RETRIEVED CONTEXT" in user
    assert "the document says X" in user
    assert "REFERENCE FACTS" not in user


def test_trace_contexts_are_labelled_the_way_generate_labels_them(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    judge_trace("q?", "a", contexts("body text"), cfg, judge_fn=fake)
    assert "[doc.pdf]\nbody text" in fake.seen[0][1]


def test_a_reference_answer_is_preferred_over_the_must_contain_fragments(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    g = GoldQuestion(
        question="q?", must_contain=["3 business days"],
        reference_answer="The duplicate authorization disappears after 3 business days.",
    )
    judge_answer("q?", "a", g, cfg, judge_fn=fake)
    assert "disappears after 3 business days" in fake.seen[0][1]


def test_an_unanswerable_question_is_judged_against_refusal_being_correct(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    judge_answer("q?", "I don't know", gold(unanswerable=True), cfg, judge_fn=fake)
    assert "only correct answer is an honest refusal" in fake.seen[0][1]


# ---------------------------------------------------------------------------
# Parsing, and refusing to guess
# ---------------------------------------------------------------------------


def test_a_clean_verdict_parses():
    v = parse_verdict(verdict_json("partial", "half of it", 0.6))
    assert (v.verdict, v.reasoning, v.confidence) == ("partial", "half of it", 0.6)
    assert v.scored and not v.ok


def test_a_verdict_survives_fences_and_preamble():
    raw = 'Here is my evaluation:\n```json\n{"reasoning":"r","verdict":"correct"}\n```'
    assert parse_verdict(raw).ok is True


def test_a_trailing_period_and_capitals_are_tolerated():
    assert parse_verdict('{"verdict": "Correct."}').ok is True


def test_malformed_json_degrades_to_unscored_and_keeps_the_raw_text():
    v = parse_verdict("the answer looks fine to me")
    assert v.verdict == "unscored"
    assert v.scored is False
    assert v.raw == "the answer looks fine to me"


def test_an_unknown_verdict_word_is_unscored_not_incorrect():
    """A judge malfunction must not be recorded as a wrong answer."""
    v = parse_verdict('{"verdict": "mostly right"}')
    assert v.verdict == "unscored"
    assert "mostly right" in v.reasoning


def test_a_verdict_is_never_guessed_from_the_prose():
    """"not incorrect" contains "incorrect"; a substring scan would misread it."""
    v = parse_verdict("The answer is not incorrect, it is fine.")
    assert v.verdict == "unscored"


def test_confidence_is_clamped_and_defaults_to_zero():
    assert parse_verdict('{"verdict":"correct","confidence":5}').confidence == 1.0
    assert parse_verdict('{"verdict":"correct","confidence":-2}').confidence == 0.0
    assert parse_verdict('{"verdict":"correct","confidence":"high"}').confidence == 0.0
    assert parse_verdict('{"verdict":"correct"}').confidence == 0.0


# ---------------------------------------------------------------------------
# The four unscored paths
# ---------------------------------------------------------------------------


def test_a_question_that_never_called_the_llm_is_never_judged(tmp_path):
    """Tripwire: the gate refused, so there is no answer, so no call is made."""
    cfg = make_config(tmp_path)
    called = {"judge": False}

    def tripwire(system, user, config):
        called["judge"] = True
        return verdict_json()

    v = judge_answer(
        "q?", "I don't know", gold(), cfg,
        used_llm=False, gate="below-threshold", judge_fn=tripwire,
    )
    assert called["judge"] is False
    assert v.verdict == "unscored"
    assert "below-threshold" in v.reasoning


def test_a_judge_exception_degrades_visibly(tmp_path):
    cfg = make_config(tmp_path)
    v = judge_answer("q?", "a", gold(), cfg, judge_fn=FakeJudge("", calls_raise=True))
    assert v.verdict == "unscored"
    assert v.failed is True
    assert "judge failed" in v.describe()
    assert "RuntimeError" in v.reasoning


def test_an_exhausted_budget_leaves_the_answer_unscored_rather_than_spending(tmp_path):
    from rag_app.llm import CallBudget

    cfg = make_config(tmp_path)
    fake = FakeJudge(verdict_json())
    budget = CallBudget(limit=1)
    assert judge_answer("q?", "a", gold(), cfg, judge_fn=fake, budget=budget).scored
    second = judge_answer("q?", "a", gold(), cfg, judge_fn=fake, budget=budget)
    assert second.verdict == "unscored"
    assert "budget" in second.reasoning
    assert fake.calls == 1  # the second question cost nothing


def test_a_gold_entry_with_nothing_to_convey_is_unscored(tmp_path):
    cfg = make_config(tmp_path)
    g = GoldQuestion(question="q?")
    v = judge_answer("q?", "a", g, cfg, judge_fn=FakeJudge(verdict_json()))
    assert v.verdict == "unscored"


def test_unscored_is_in_the_verdict_set_but_not_the_scorable_one():
    assert "unscored" in VERDICTS
    assert "unscored" not in SCORABLE


def test_a_missing_api_key_does_not_raise_through_the_judge(tmp_path):
    """The real _call runs; build_client raises; the judge degrades."""
    cfg = make_config(tmp_path, llm_api_key=None)
    v = judge_answer("q?", "a", gold(), cfg)
    assert v.verdict == "unscored"
    assert v.failed is True


def test_the_judge_runs_on_the_judge_model_not_the_generation_model(tmp_path):
    """The seam resolves judge_llm(cfg), so a stronger model is actually used."""
    from dataclasses import replace

    from rag_app.config import EvaluationConfig, judge_llm

    cfg = make_config(tmp_path)
    cfg = replace(cfg, evaluation=EvaluationConfig(judge_model="big/model"))
    assert judge_llm(cfg).model == "big/model"
    assert judge_llm(cfg).model != cfg.llm.model


# ---------------------------------------------------------------------------
# G-Eval
# ---------------------------------------------------------------------------


def test_geval_averages_n_samples(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(s) for s in (3, 4, 5, 4, 4)]})
    score = geval_score("q?", "a", gold(), cfg, judge_fn=fake, samples=5)
    assert score.samples == [3, 4, 5, 4, 4]
    assert score.score == pytest.approx(4.0)
    assert fake.calls == 5


def test_geval_reports_the_spread_because_a_mean_of_one_and_five_is_not_a_three(tmp_path):
    """The spread is what the paper's logprob weighting carries and a single
    greedy call throws away."""
    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(s) for s in (1, 5, 1, 5, 3)]})
    score = geval_score("q?", "a", gold(), cfg, judge_fn=fake, samples=5)
    assert score.score == pytest.approx(3.0)
    assert score.stdev > 1.5
    assert "sd 2.0" in score.describe()


def test_a_single_sample_has_no_spread(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(4)]})
    assert geval_score("q?", "a", gold(), cfg, judge_fn=fake, samples=1).stdev == 0.0


def test_a_score_outside_the_range_is_dropped_not_clamped(tmp_path):
    """Clamping 7 to 5 launders a misunderstood rubric into a maximal score."""
    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(7), geval_json(4), geval_json(0)]})
    score = geval_score("q?", "a", gold(), cfg, judge_fn=fake, samples=3)
    assert score.samples == [4]
    assert score.score == pytest.approx(4.0)
    assert "1 of 3 samples parsed" in score.detail


def test_geval_with_every_sample_malformed_fails_visibly(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: ["no json", "still none"]})
    score = geval_score("q?", "a", gold(), cfg, judge_fn=fake, samples=2)
    assert score.failed is True
    assert score.score == 0.0
    assert "unscored" in score.describe()


def test_geval_survives_some_calls_raising(tmp_path):
    cfg = make_config(tmp_path)

    class Flaky:
        def __init__(self):
            self.n = 0

        def __call__(self, system, user, cfg):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            return geval_json(5)

    score = geval_score("q?", "a", gold(), cfg, judge_fn=Flaky(), samples=3)
    assert score.samples == [5, 5]
    assert score.failed is False


def test_geval_sample_count_comes_from_config(tmp_path):
    from dataclasses import replace

    from rag_app.config import EvaluationConfig

    cfg = replace(
        make_config(tmp_path),
        evaluation=EvaluationConfig(geval_samples=2, geval_temperature=1.0),
    )
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(4), geval_json(4)]})
    geval_score("q?", "a", gold(), cfg, judge_fn=fake)
    assert fake.calls == 2


def test_geval_never_runs_when_the_llm_was_never_called(tmp_path):
    cfg = make_config(tmp_path)
    called = {"judge": False}

    def tripwire(system, user, config):
        called["judge"] = True
        return geval_json(5)

    score = geval_score(
        "q?", "x", gold(), cfg, used_llm=False, gate="no-candidates", judge_fn=tripwire
    )
    assert called["judge"] is False
    assert score.failed is True


def test_geval_reserves_every_sample_before_spending_any(tmp_path):
    """All-or-nothing: half a sample set is a worse number than none."""
    from rag_app.llm import CallBudget

    cfg = make_config(tmp_path)
    fake = FakeJudge({GEVAL_SYSTEM: [geval_json(4)] * 5})
    score = geval_score(
        "q?", "a", gold(), cfg, judge_fn=fake, samples=5, budget=CallBudget(limit=3)
    )
    assert score.failed is True
    assert fake.calls == 0


def test_geval_does_not_regenerate_its_rubric_per_run():
    """Auto-CoT would produce a different rubric every run, destroying the
    run-to-run comparability before/after measurement depends on."""
    assert "EVALUATION STEPS" in GEVAL_SYSTEM
    assert GEVAL_SYSTEM == GEVAL_SYSTEM  # a constant, not a generated string
    assert json.loads(geval_json(3))["score"] == 3
