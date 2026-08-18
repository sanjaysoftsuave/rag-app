"""Failure-bucket labeling and the dense-vs-hybrid before/after comparison —
the two pieces of this week's "debug retrieval" task."""

from __future__ import annotations

import numpy as np

from rag_app.chunking import Chunk
from rag_app.evaluate import GoldQuestion, compare_retrieval_modes, label_failures
from rag_app.pipeline import open_store
from rag_app.store import StoreMeta, VectorStore, store_path_for_preset

from conftest import FakeReranker, FixedEmbedder, make_config


def _seed(cfg, preset, chunks, vectors):
    VectorStore(
        chunks, np.asarray(vectors, dtype=np.float32),
        StoreMeta("fake", 2, "ticket", 2000, 200, len(chunks)),
    ).save(store_path_for_preset(cfg.store_dir, preset))
    return open_store(cfg, preset)


# --- label_failures ---------------------------------------------------------


def test_retrieval_bucket_when_ticket_never_retrieved(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-OTHER", "something else", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("Where is TIC-MISSING?", "TIC-MISSING", "x")]

    report = label_failures(store, cfg, "C", embedder=FixedEmbedder(),
                             reranker=FakeReranker(0.9), gold=gold, use_llm=False)
    assert report.labels[0].bucket == "retrieval"
    assert "never entered" in report.labels[0].evidence


def test_unconfirmed_bucket_without_generate_flag(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-A", "content", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-A", "x")]

    report = label_failures(store, cfg, "C", embedder=FixedEmbedder(),
                             reranker=FakeReranker(0.9), gold=gold, use_llm=False)
    assert report.labels[0].bucket == "unconfirmed"
    assert report.generated is False


def test_generation_bucket_when_gate_refuses_despite_correct_doc(tmp_path):
    """The document reached the context; the SCORE GATE is what threw the
    answer away. This still counts as 'right document, wrong answer'."""
    cfg = make_config(tmp_path, score_threshold=0.5)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-A", "content", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-A", "60")]

    report = label_failures(
        store, cfg, "C", embedder=FixedEmbedder(), reranker=FakeReranker(0.1),
        gold=gold, use_llm=True, generate_fn=lambda q, c, k: "should not run",
    )
    assert report.labels[0].bucket == "generation"
    assert "gate" in report.labels[0].evidence


def test_generation_bucket_when_answer_misses_expected_content(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.2)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-A", "content", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-A", "60")]

    report = label_failures(
        store, cfg, "C", embedder=FixedEmbedder(), reranker=FakeReranker(0.9),
        gold=gold, use_llm=True,
        generate_fn=lambda q, c, k: "The answer is something unrelated [TIC-A].",
    )
    assert report.labels[0].bucket == "generation"
    assert "does not mention" in report.labels[0].evidence


def test_pass_when_answer_contains_expected_content(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.2)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-A", "content", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-A", "60")]

    report = label_failures(
        store, cfg, "C", embedder=FixedEmbedder(), reranker=FakeReranker(0.9),
        gold=gold, use_llm=True,
        generate_fn=lambda q, c, k: "The limit is 60 requests per minute [TIC-A].",
    )
    assert report.labels[0].bucket == "pass"


def test_report_counts_and_describe(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-A", "content", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-A", "x"), GoldQuestion("missing", "TIC-ZZZ", "x")]

    report = label_failures(store, cfg, "C", embedder=FixedEmbedder(),
                             reranker=FakeReranker(0.9), gold=gold, use_llm=False)
    counts = report.counts()
    assert counts.get("unconfirmed") == 1
    assert counts.get("retrieval") == 1
    assert "retrieval failures" in report.describe()


# --- compare_retrieval_modes -------------------------------------------------


def test_compare_retrieval_detects_a_fix_from_keyword_search(tmp_path):
    """A ticket findable only via exact-term overlap, invisible to dense
    search given these vectors, should be labeled 'fixed' by hybrid."""
    cfg = make_config(tmp_path)
    chunks = [
        Chunk("a::0", "TIC-DENSE", "irrelevant filler text", {}),
        Chunk("b::0", "TIC-KEYWORD", "this ticket literally says ZQXK-7777", {}),
    ]
    store = _seed(cfg, "C", chunks, [[1.0, 0.0], [0.0, 1.0]])
    gold = [GoldQuestion("ZQXK-7777", "TIC-KEYWORD", "x")]

    report = compare_retrieval_modes(store, cfg, "C", embedder=FixedEmbedder(), gold=gold, k=1)
    row = report.rows[0]
    assert row.outcome == "fixed"
    assert row.dense_hit is False
    assert row.hybrid_hit is True


def test_compare_retrieval_reports_still_broken_honestly(tmp_path):
    cfg = make_config(tmp_path)
    store = _seed(cfg, "C", [Chunk("a::0", "TIC-OTHER", "nothing relevant here", {})], [[1.0, 0.0]])
    gold = [GoldQuestion("q", "TIC-MISSING", "x")]

    report = compare_retrieval_modes(store, cfg, "C", embedder=FixedEmbedder(), gold=gold, k=1)
    assert report.rows[0].outcome == "still-broken"
    assert "NOT fixed" in report.describe()


def test_compare_retrieval_hit_rates(tmp_path):
    cfg = make_config(tmp_path)
    chunks = [
        Chunk("a::0", "TIC-A", "content about the topic", {}),
        Chunk("b::0", "TIC-B", "unrelated other content", {}),
    ]
    store = _seed(cfg, "C", chunks, [[1.0, 0.0], [0.0, 1.0]])
    gold = [GoldQuestion("topic content", "TIC-A", "x")]

    report = compare_retrieval_modes(store, cfg, "C", embedder=FixedEmbedder(), gold=gold, k=2)
    assert report.dense_hit_rate == 1.0
    assert report.hybrid_hit_rate == 1.0
    assert report.rows[0].outcome == "always-hit"
