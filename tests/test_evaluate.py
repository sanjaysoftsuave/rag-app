"""Metrics and failure labelling.

The numbers here decide whether a change to retrieval helped, so a metric that
is quietly wrong is worse than no metric — it launders a regression as an
improvement. Each one is checked against a hand-computed value.
"""

from __future__ import annotations

import pytest

from rag_app.chunking import Chunk
from rag_app.evaluate import (
    EvalReport,
    GoldQuestion,
    QuestionResult,
    evaluate,
    label_failures,
    load_gold,
    parse_gold,
    report_to_json,
    write_gold,
)
from rag_app.store import ScoredChunk

from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store


def chunk(source: str, text: str) -> Chunk:
    return Chunk(chunk_id=f"{source}::0", source=source, text=text, metadata={})


def result(gold, r_rank, n_rank, r_found=(), n_found=(), **kw) -> QuestionResult:
    base = dict(
        best_score=0.9, used_llm=True, refused=False, gate="answered", answer_text=""
    )
    base.update(kw)
    return QuestionResult(
        gold=gold, retrieved_rank=r_rank, reranked_rank=n_rank,
        retrieved_found=set(r_found), reranked_found=set(n_found), **base,
    )


# --- the gold set format ---------------------------------------------------


def test_expect_in_chunk_defaults_to_must_contain():
    g = parse_gold([{"question": "q", "must_contain": "60 minutes"}])[0]
    assert g.snippets == ["60 minutes"]
    assert g.must_contain == ["60 minutes"]


def test_expect_in_chunk_can_differ_from_must_contain():
    """The document may phrase a fact differently from how an answer states it."""
    g = parse_gold([{
        "question": "q", "must_contain": "5 retries", "expect_in_chunk": "retry",
    }])[0]
    assert g.snippets == ["retry"]
    assert g.must_contain == ["5 retries"]


def test_a_single_string_and_a_list_both_work():
    one = parse_gold([{"question": "q", "must_contain": "a"}])[0]
    many = parse_gold([{"question": "q", "must_contain": ["a", "b"]}])[0]
    assert one.must_contain == ["a"] and many.must_contain == ["a", "b"]


def test_matching_is_case_insensitive():
    g = parse_gold([{"question": "q", "must_contain": "Invalid Credentials"}])[0]
    assert g.found_in("cleared the invalid credentials cache") == ["Invalid Credentials"]


def test_an_answerable_question_with_nothing_to_look_for_is_rejected():
    """Otherwise it silently counts as a free pass in every metric."""
    with pytest.raises(ValueError, match="names nothing to look for"):
        parse_gold([{"question": "q"}])


def test_an_unanswerable_question_needs_no_snippets():
    g = parse_gold([{"question": "q", "unanswerable": True}])[0]
    assert not g.answerable and g.snippets == []


def test_malformed_entries_say_which_one():
    with pytest.raises(ValueError, match="entry 2"):
        parse_gold([{"question": "ok", "must_contain": "x"}, {"nope": 1}])
    with pytest.raises(ValueError, match="must be a list"):
        parse_gold({"question": "q"})


def test_gold_round_trips_through_a_file(tmp_path):
    original = parse_gold([
        {"question": "a", "must_contain": ["x", "y"], "note": "n"},
        {"question": "b", "unanswerable": True},
    ])
    path = tmp_path / "gold.yaml"
    write_gold(original, path)
    assert load_gold(make_config(tmp_path), path) == original


def test_a_missing_gold_file_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="write them for YOUR documents"):
        load_gold(make_config(tmp_path), tmp_path / "nope.yaml")


# --- the metrics -----------------------------------------------------------


def report_of(results, k=10, n=3, generated=False) -> EvalReport:
    return EvalReport(preset="C", k=k, n=n, results=results, generated=generated)


def test_hit_rate_counts_questions_recall_counts_snippets():
    """The distinction that matters: one question needing two facts, only one
    found, is a hit but only half a recall."""
    g = GoldQuestion("q", must_contain=["a", "b"])
    r = report_of([result(g, 1, 1, r_found={"a"}, n_found={"a"})])
    assert r.hit_rate == 1.0
    assert r.recall_at_k == pytest.approx(0.5)


def test_hit_rate_and_recall_coincide_with_one_snippet_each():
    rows = [
        result(GoldQuestion("a", must_contain=["x"]), 1, 1, {"x"}, {"x"}),
        result(GoldQuestion("b", must_contain=["y"]), None, None),
    ]
    r = report_of(rows)
    assert r.hit_rate == pytest.approx(0.5)
    assert r.recall_at_k == pytest.approx(0.5)


def test_mrr_is_the_mean_reciprocal_rank():
    rows = [
        result(GoldQuestion("a", must_contain=["x"]), 1, 1, {"x"}, {"x"}),
        result(GoldQuestion("b", must_contain=["y"]), 4, None, {"y"}),
        result(GoldQuestion("c", must_contain=["z"]), None, None),
    ]
    # (1/1 + 1/4 + 0) / 3
    assert report_of(rows).mrr == pytest.approx((1 + 0.25) / 3)


def test_rerank_lift_compares_like_with_like():
    """Both sides are 'did the right text occupy one of N slots' — one chosen
    by the bi-encoder, one by the cross-encoder."""
    rows = [
        # retrieved at 5 (outside top-3), reranked into context: the reranker won
        result(GoldQuestion("a", must_contain=["x"]), 5, 2, {"x"}, {"x"}),
        # already inside the retriever's top-3 and stayed: no credit
        result(GoldQuestion("b", must_contain=["y"]), 1, 1, {"y"}, {"y"}),
    ]
    r = report_of(rows, n=3)
    assert r.hit_rate_after_rerank == 1.0
    assert r.rerank_lift == pytest.approx(0.5)


def test_rerank_lift_goes_negative_when_the_reranker_loses_a_hit():
    rows = [result(GoldQuestion("a", must_contain=["x"]), 1, None, {"x"}, set())]
    assert report_of(rows, n=3).rerank_lift == pytest.approx(-1.0)


def test_refusal_accuracy_and_false_refusals_are_read_together():
    """A gate that refuses everything scores 100% on refusals and is useless."""
    rows = [
        result(GoldQuestion("real", must_contain=["x"]), 1, 1, {"x"}, {"x"}, refused=True),
        result(GoldQuestion("absent", unanswerable=True), None, None, refused=True),
    ]
    r = report_of(rows)
    assert r.refusal_accuracy == 1.0
    assert len(r.false_refusals) == 1


def test_answer_accuracy_requires_every_must_contain():
    g = GoldQuestion("q", must_contain=["alpha", "beta"])
    partial = result(g, 1, 1, {"alpha"}, {"alpha"}, answer_text="only alpha here")
    full = result(g, 1, 1, {"alpha"}, {"alpha"}, answer_text="alpha and beta")
    assert not partial.answer_ok
    assert full.answer_ok


def test_metrics_on_an_empty_set_are_zero_not_a_crash():
    r = report_of([])
    assert (r.hit_rate, r.recall_at_k, r.mrr, r.refusal_accuracy) == (0.0, 0.0, 0.0, 0.0)


def test_json_summary_carries_every_headline_number():
    import json

    rows = [result(GoldQuestion("a", must_contain=["x"]), 1, 1, {"x"}, {"x"})]
    payload = json.loads(report_to_json(report_of(rows)))
    for key in ("hit_rate_at_k", "recall_at_k", "mrr", "rerank_lift",
                "refusal_accuracy", "false_refusals"):
        assert key in payload
    assert payload["answer_accuracy"] is None, "None until --generate is passed"


# --- end to end through the real pipeline ----------------------------------


def build(tmp_path):
    embedder = FakeEmbedder()
    chunks = [
        chunk("a.md", "Password reset links expire after 60 minutes."),
        chunk("b.md", "Refunds are issued within five business days."),
        chunk("c.md", "The dashboard supports up to 8 widgets."),
    ]
    vectors = embedder.encode_documents([c.text for c in chunks])
    return embedder, make_qdrant_store(tmp_path, chunks, vectors)


def test_evaluate_runs_the_real_pipeline(tmp_path):
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0, retrieve_k=3, rerank_n=3)
    gold = parse_gold([
        {"question": "how long is a reset link valid", "must_contain": "60 minutes"},
        {"question": "what is the price", "unanswerable": True},
    ])
    r = evaluate(gold, preset="C", config=cfg, store=store,
                 embedder=embedder, reranker=FakeReranker(0.9))
    assert r.hit_rate == 1.0
    assert r.k == 3 and r.n == 3
    assert not r.generated
    store.close()


def test_label_failures_separates_retrieval_from_generation(tmp_path):
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0, retrieve_k=3, rerank_n=3)

    gold = parse_gold([
        {"question": "reset link", "must_contain": "60 minutes"},
        {"question": "something absent", "must_contain": "not in any chunk"},
    ])
    report = label_failures(
        gold, preset="C", config=cfg, store=store,
        embedder=embedder, reranker=FakeReranker(0.9),
        use_llm=True, generate_fn=lambda q, c, k: "Links expire after 60 minutes.",
    )
    buckets = {x.gold.question: x.bucket for x in report.labels}
    assert buckets["reset link"] == "pass"
    assert buckets["something absent"] == "retrieval"
    assert "never retrieved" in next(
        x.evidence for x in report.labels if x.bucket == "retrieval"
    )
    store.close()


def test_a_false_refusal_is_a_generation_failure_not_a_retrieval_one(tmp_path):
    """Retrieval did its job — the gate is what rejected it. Calling this a
    retrieval failure would send you tuning the wrong stage."""
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.9, retrieve_k=3, rerank_n=3)
    gold = parse_gold([{"question": "reset link", "must_contain": "60 minutes"}])

    report = label_failures(gold, preset="C", config=cfg, store=store,
                            embedder=embedder, reranker=FakeReranker(0.1))
    label = report.labels[0]
    assert label.bucket == "generation"
    assert "score gate refused" in label.evidence
    assert "false refusal" in label.evidence.lower()
    store.close()


def test_without_generate_a_reached_chunk_is_unconfirmed_not_a_pass(tmp_path):
    """`use_llm=False` can only prove a retrieval failure. Claiming a pass
    would assert something nothing checked."""
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0, retrieve_k=3, rerank_n=3)
    gold = parse_gold([{"question": "reset link", "must_contain": "60 minutes"}])

    report = label_failures(gold, preset="C", config=cfg, store=store,
                            embedder=embedder, reranker=FakeReranker(0.9))
    assert report.labels[0].bucket == "unconfirmed"
    assert not report.generated
    store.close()


def test_the_answer_omitting_a_required_string_is_a_generation_failure(tmp_path):
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0, retrieve_k=3, rerank_n=3)
    gold = parse_gold([{"question": "reset link", "must_contain": "60 minutes"}])

    report = label_failures(
        gold, preset="C", config=cfg, store=store,
        embedder=embedder, reranker=FakeReranker(0.9),
        use_llm=True, generate_fn=lambda q, c, k: "Links expire eventually.",
    )
    assert report.labels[0].bucket == "generation"
    assert "omitted" in report.labels[0].evidence


def test_unanswerable_questions_are_not_labelled(tmp_path):
    """They have no right chunk to find, so the buckets do not apply."""
    embedder, store = build(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    gold = parse_gold([{"question": "the price", "unanswerable": True}])
    report = label_failures(gold, preset="C", config=cfg, store=store,
                            embedder=embedder, reranker=FakeReranker(0.9))
    assert report.labels == []
    store.close()
