"""The four RAGAS metrics.

Each metric's arithmetic is pinned against a hand-computed value, and each
metric's degenerate case is pinned too — because the degenerate cases are where
a metric quietly starts rewarding the wrong behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import FakeEmbedder, FakeJudge, make_config

from rag_app.ragas_metrics import (
    CONTEXT_PRECISION_SYSTEM,
    CONTEXT_RECALL_SYSTEM,
    FAITHFULNESS_STATEMENTS_SYSTEM,
    FAITHFULNESS_VERDICTS_SYSTEM,
    RELEVANCY_QUESTIONS_SYSTEM,
    RagasReport,
    answer_relevancy,
    average_precision,
    context_precision,
    context_recall,
    faithfulness,
    ragas_report_to_json,
    score_question,
    split_sentences,
)
from rag_app.store import Chunk, ScoredChunk


def contexts(*texts) -> list[ScoredChunk]:
    return [
        ScoredChunk(Chunk(f"c::{i}", "doc.pdf", t, {}), 0.9)
        for i, t in enumerate(texts)
    ]


def statements_reply(*sts) -> str:
    return json.dumps({"statements": list(sts)})


def verdicts_reply(*flags) -> str:
    return json.dumps([{"index": i, "verdict": f, "reason": "r"}
                       for i, f in enumerate(flags, start=1)])


def useful_reply(*flags) -> str:
    return json.dumps([{"position": i, "useful": f} for i, f in enumerate(flags, start=1)])


def attributed_reply(*flags) -> str:
    return json.dumps([{"index": i, "attributed": f} for i, f in enumerate(flags, start=1)])


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_is_supported_statements_over_total(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply("a", "b", "c"),
        FAITHFULNESS_VERDICTS_SYSTEM: verdicts_reply(1, 1, 0),
    })
    f = faithfulness("an answer", contexts("ctx"), cfg, judge_fn=fake)
    assert f.score == pytest.approx(2 / 3)
    assert f.supported == [1, 1, 0]
    assert fake.calls == 2  # decompose once, verdict all at once


def test_an_answer_with_no_statements_is_unscored_not_perfect(tmp_path):
    """1.0 would make a contentless answer maximally faithful."""
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply(),
        FAITHFULNESS_VERDICTS_SYSTEM: verdicts_reply(),
    })
    f = faithfulness("mm", contexts("ctx"), cfg, judge_fn=fake)
    assert f.failed is True
    assert f.score is None
    assert "unscored" in f.describe()


def test_a_refusal_is_excluded_before_any_call_is_made(tmp_path):
    """Otherwise the metric rewards the gate for refusing everything."""
    from rag_app.generate import DONT_KNOW

    cfg = make_config(tmp_path)
    called = {"n": 0}

    def tripwire(system, user, config):
        called["n"] += 1
        return statements_reply("x")

    f = faithfulness(DONT_KNOW, contexts("ctx"), cfg, judge_fn=tripwire)
    assert called["n"] == 0
    assert f.failed is True
    assert "refusal" in f.detail


def test_faithfulness_keeps_the_statements_so_the_number_is_inspectable(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply("the link lasts 60 minutes"),
        FAITHFULNESS_VERDICTS_SYSTEM: verdicts_reply(0),
    })
    f = faithfulness("a", contexts("ctx"), cfg, judge_fn=fake)
    assert f.statements == ["the link lasts 60 minutes"]
    assert f.reasons == ["r"]


def test_a_short_verdict_list_fails_visibly_rather_than_scoring_fewer(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply("a", "b", "c"),
        FAITHFULNESS_VERDICTS_SYSTEM: verdicts_reply(1, 1),
    })
    f = faithfulness("a", contexts("ctx"), cfg, judge_fn=fake)
    assert f.failed is True
    assert "2 verdicts for 3 statements" in f.detail


def test_the_prompt_says_true_but_absent_still_scores_zero():
    assert "TRUE IN THE WORLD" in FAITHFULNESS_VERDICTS_SYSTEM


def test_the_statement_prompt_demands_standalone_statements():
    assert "stand alone" in FAITHFULNESS_STATEMENTS_SYSTEM


# ---------------------------------------------------------------------------
# 2. Answer relevancy
# ---------------------------------------------------------------------------


class QueryOnlyEmbedder(FakeEmbedder):
    """Raises if the passage-side encoder is used on a question."""

    def encode_documents(self, texts):
        raise AssertionError(
            "answer relevancy compared a question encoded as a passage; bge is "
            "asymmetric, so this shifts every similarity with no error"
        )

    def encode_queries(self, texts):
        return FakeEmbedder.encode_documents(self, texts)


def test_answer_relevancy_uses_encode_queries_on_both_sides(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        RELEVANCY_QUESTIONS_SYSTEM: json.dumps(
            {"questions": ["q1", "q2", "q3"], "noncommittal": False}
        )
    })
    r = answer_relevancy("q?", "an answer", contexts("c"), cfg,
                         embedder=QueryOnlyEmbedder(), judge_fn=fake)
    assert r.failed is False
    assert len(r.similarities) == 3


def test_answer_relevancy_is_the_mean_cosine_to_the_generated_questions(tmp_path):
    cfg = make_config(tmp_path)

    class Fixed(FakeEmbedder):
        """Original at 1.0 similarity to the first, 0.0 to the second."""

        def encode_queries(self, texts):
            v = np.zeros((len(texts), 2), dtype=np.float32)
            v[0] = [1.0, 0.0]   # the original question
            v[1] = [1.0, 0.0]   # identical -> cos 1
            v[2] = [0.0, 1.0]   # orthogonal -> cos 0
            return v

    fake = FakeJudge({
        RELEVANCY_QUESTIONS_SYSTEM: json.dumps(
            {"questions": ["same", "other"], "noncommittal": False}
        )
    })
    r = answer_relevancy("q?", "a", [], cfg, embedder=Fixed(), judge_fn=fake)
    assert r.similarities == pytest.approx([1.0, 0.0])
    assert r.score == pytest.approx(0.5)


def test_a_noncommittal_answer_scores_zero_however_similar(tmp_path):
    """"I don't know" is topically close to the question and says nothing."""
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        RELEVANCY_QUESTIONS_SYSTEM: json.dumps(
            {"questions": ["q1"], "noncommittal": True}
        )
    })
    r = answer_relevancy("q?", "I don't know", [], cfg,
                         embedder=QueryOnlyEmbedder(), judge_fn=fake)
    assert r.score == 0.0
    assert r.noncommittal is True
    assert "noncommittal" in r.describe()


def test_relevancy_costs_exactly_one_llm_call(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({
        RELEVANCY_QUESTIONS_SYSTEM: json.dumps({"questions": ["a"], "noncommittal": False})
    })
    answer_relevancy("q?", "a", [], cfg, embedder=QueryOnlyEmbedder(), judge_fn=fake)
    assert fake.calls == 1


# ---------------------------------------------------------------------------
# 3. Context precision
# ---------------------------------------------------------------------------


def test_average_precision_is_rank_aware():
    """The whole reason this is not "fraction useful"."""
    assert average_precision([1, 0, 0]) == pytest.approx(1.0)
    assert average_precision([0, 0, 1]) == pytest.approx(1 / 3)
    assert average_precision([1, 1, 0]) == pytest.approx(1.0)     # (1/1 + 2/2) / 2
    assert average_precision([0, 1, 1]) == pytest.approx(0.5833, abs=1e-3)  # (1/2 + 2/3)/2
    assert average_precision([0, 0, 0]) == 0.0


def test_context_precision_scores_a_perfect_ranking_higher_than_a_reversed_one(tmp_path):
    cfg = make_config(tmp_path)
    ctx = contexts("a", "b", "c")
    first = context_precision(
        "q?", ctx, cfg, judge_fn=FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(1, 0, 0)})
    )
    last = context_precision(
        "q?", ctx, cfg, judge_fn=FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(0, 0, 1)})
    )
    assert first.score == pytest.approx(1.0)
    assert last.score == pytest.approx(1 / 3)


def test_context_precision_judges_every_context_in_one_call(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(1, 1, 0)})
    context_precision("q?", contexts("a", "b", "c"), cfg, judge_fn=fake)
    assert fake.calls == 1


def test_a_short_verdict_list_fails_visibly_rather_than_silently_scoring_fewer(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(1, 0)})
    cp = context_precision("q?", contexts("a", "b", "c"), cfg, judge_fn=fake)
    assert cp.failed is True
    assert "2 verdicts for 3 contexts" in cp.detail


def test_nothing_useful_scores_zero_and_is_not_a_failure(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(0, 0)})
    cp = context_precision("q?", contexts("a", "b"), cfg, judge_fn=fake)
    assert cp.score == 0.0
    assert cp.failed is False   # well-defined: nothing useful was retrieved


def test_contexts_are_labelled_by_position_so_verdicts_can_be_ordered(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(1, 0)})
    context_precision("q?", contexts("alpha", "beta"), cfg, judge_fn=fake)
    user = fake.seen[0][1]
    assert "CONTEXT 1\nalpha" in user
    assert "CONTEXT 2\nbeta" in user


# ---------------------------------------------------------------------------
# 4. Context recall
# ---------------------------------------------------------------------------


def test_sentences_split_without_an_llm_call():
    out = split_sentences("The link lasts 60 minutes. Requesting a new one voids it.")
    assert out == ["The link lasts 60 minutes.", "Requesting a new one voids it."]


def test_sentence_splitting_does_not_break_on_an_empty_reference():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_context_recall_is_attributable_sentences_over_total(tmp_path):
    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_RECALL_SYSTEM: attributed_reply(1, 0)})
    cr = context_recall(
        contexts("ctx"), "First fact here. Second fact here.", cfg, judge_fn=fake
    )
    assert cr.score == pytest.approx(0.5)
    assert cr.attributed == [1, 0]


def test_context_recall_without_a_reference_is_none_not_zero(tmp_path):
    """Averaging a 0 for missing data reports a regression that never happened."""
    cfg = make_config(tmp_path)
    called = {"n": 0}

    def tripwire(system, user, config):
        called["n"] += 1
        return attributed_reply(1)

    cr = context_recall(contexts("ctx"), "", cfg, judge_fn=tripwire)
    assert cr.score is None
    assert cr.measured is False
    assert called["n"] == 0                     # and it costs nothing
    assert "not measured" in cr.describe()
    assert "add one" in cr.detail


def test_context_recall_never_derives_a_reference_from_must_contain(tmp_path):
    """expect_in_chunk is *defined* as text in a retrieved chunk, so a recall
    derived from it would be 1.0 by construction."""
    cfg = make_config(tmp_path)

    class Gold:
        must_contain = ["3 business days"]
        expect_in_chunk = ["authorization"]
        reference_answer = ""

    scores = score_question(
        "q?", "an answer", contexts("ctx"), cfg, gold=Gold(),
        embedder=QueryOnlyEmbedder(),
        judge_fn=FakeJudge({
            FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply("s"),
            FAITHFULNESS_VERDICTS_SYSTEM: verdicts_reply(1),
            RELEVANCY_QUESTIONS_SYSTEM: json.dumps({"questions": ["q"], "noncommittal": False}),
            CONTEXT_PRECISION_SYSTEM: useful_reply(1),
            CONTEXT_RECALL_SYSTEM: attributed_reply(1),
        }),
    )
    assert scores.context_recall.score is None


# ---------------------------------------------------------------------------
# Budget and aggregate
# ---------------------------------------------------------------------------


def test_an_exhausted_budget_degrades_rather_than_spending(tmp_path):
    from rag_app.llm import CallBudget

    cfg = make_config(tmp_path)
    fake = FakeJudge({CONTEXT_PRECISION_SYSTEM: useful_reply(1)})
    cp = context_precision(
        "q?", contexts("a"), cfg, judge_fn=fake, budget=CallBudget(limit=0 or 1, used=1)
    )
    assert cp.failed is True
    assert fake.calls == 0


def test_faithfulness_reserves_both_calls_before_spending_either(tmp_path):
    from rag_app.llm import CallBudget

    cfg = make_config(tmp_path)
    fake = FakeJudge({FAITHFULNESS_STATEMENTS_SYSTEM: statements_reply("a")})
    f = faithfulness("a", contexts("c"), cfg, judge_fn=fake, budget=CallBudget(limit=1))
    assert f.failed is True
    assert fake.calls == 0


def test_the_means_report_their_own_denominator(tmp_path):
    from rag_app.ragas_metrics import ContextRecall, Faithfulness, RagasScores

    report = RagasReport(rows=[
        ("q1", RagasScores(faithfulness=Faithfulness(score=1.0, statements=["a"], supported=[1]))),
        ("q2", RagasScores(faithfulness=Faithfulness(failed=True, detail="x"))),
    ])
    out = report.describe()
    assert "n=1 of 2" in out
    assert report.mean("faithfulness") == pytest.approx(1.0)   # the failure is excluded


def test_unmeasured_context_recall_is_called_out_not_averaged_in():
    from rag_app.ragas_metrics import ContextRecall, RagasScores

    report = RagasReport(rows=[("q1", RagasScores(context_recall=ContextRecall()))])
    out = report.describe()
    assert "no reference_answer" in out
    assert "excluded, not scored 0" in out
    assert report.mean("context_recall") is None


def test_the_json_summary_carries_all_four_metrics():
    payload = json.loads(ragas_report_to_json(RagasReport(rows=[])))
    for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert key in payload
        assert payload[key] is None
