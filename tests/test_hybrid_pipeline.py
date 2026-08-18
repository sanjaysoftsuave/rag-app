"""Proof that ask() actually swaps its retrieval strategy when configured —
not just that hybrid_retrieve() works in isolation."""

from __future__ import annotations

import numpy as np

from rag_app.chunking import Chunk
from rag_app.config import RetrievalConfig
from rag_app.pipeline import ask
from rag_app.store import StoreMeta, VectorStore, store_path_for_preset

from conftest import FakeReranker, FixedEmbedder, make_config


def _seed(cfg, preset):
    chunks = [
        Chunk("a::0", "TIC-DENSE", "irrelevant filler text about something else", {}),
        Chunk("b::0", "TIC-KEYWORD", "this ticket literally contains ZQXK-7777", {}),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    VectorStore(chunks, vectors, StoreMeta("fake", 2, "ticket", 100, 0, 2)).save(
        store_path_for_preset(cfg.store_dir, preset)
    )


def test_dense_mode_misses_the_keyword_only_match(tmp_path):
    """Baseline: with retrieval.mode left at its default ('dense'), the
    keyword-only ticket should NOT be the top hit — FixedEmbedder makes every
    query vector identical to TIC-DENSE's stored vector."""
    cfg = make_config(tmp_path, score_threshold=0.0)
    _seed(cfg, "C")

    answer = ask(
        "ZQXK-7777", preset="C", config=cfg,
        embedder=FixedEmbedder(), reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "answered [TIC-DENSE]",
    )
    assert answer.retrieved[0].chunk.source == "TIC-DENSE"


def test_hybrid_mode_surfaces_the_keyword_only_match(tmp_path):
    """Same store, same embedder, same query — the ONLY thing that changed is
    cfg.retrieval.mode. That isolation is the point of the experiment."""
    cfg = make_config(
        tmp_path, score_threshold=0.0,
        retrieval=RetrievalConfig(mode="hybrid", rrf_k=60, bm25_pool=10),
    )
    _seed(cfg, "C")

    answer = ask(
        "ZQXK-7777", preset="C", config=cfg,
        embedder=FixedEmbedder(), reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "answered [TIC-KEYWORD]",
    )
    sources = [item.chunk.source for item in answer.retrieved]
    assert "TIC-KEYWORD" in sources
    assert answer.meta["retrieval_mode"] == "hybrid"


def test_hybrid_mode_reuses_a_supplied_bm25_index(tmp_path):
    """A caller (cli.py's shared-model pattern) can build the BM25 index once
    and pass it in, instead of ask() rebuilding it every call."""
    from rag_app.bm25 import BM25Index
    from rag_app.pipeline import open_store

    cfg = make_config(tmp_path, score_threshold=0.0, retrieval=RetrievalConfig(mode="hybrid"))
    _seed(cfg, "C")
    store = open_store(cfg, "C")
    index = BM25Index.from_store(store)

    answer = ask(
        "ZQXK-7777", preset="C", config=cfg, store=store, bm25=index,
        embedder=FixedEmbedder(), reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "answered [TIC-KEYWORD]",
    )
    assert "TIC-KEYWORD" in [item.chunk.source for item in answer.retrieved]
