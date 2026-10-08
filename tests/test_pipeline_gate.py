"""The score gate — the invariant CLAUDE.md says must survive any refactor.

Broader grounding coverage (citations, refusal detection, filters) lives in
test_grounding.py; this file stays focused on the two original guarantees so
`pytest tests/test_pipeline_gate.py` remains a meaningful smoke test.
"""

import numpy as np

from rag_app.chunking import Chunk
from rag_app.generate import DONT_KNOW, is_refusal
from rag_app.pipeline import ask
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.store import StoreMeta

from conftest import FakeReranker, FixedEmbedder, make_config


def _seed(cfg, preset="C"):
    chunks = [Chunk("TIC-001::0", "TIC-001", "Cursor pagination, max 100 per page",
                    {"ticket_id": "TIC-001", "product": "API"})]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    meta = StoreMeta("fake", 2, "ticket", 2000, 200, 1)
    path = qdrant_path_for_preset(cfg.store_dir, preset)
    store = QdrantStore(meta_dir=path, path=path)
    store.build(chunks, vectors, meta)
    store.close()


def test_score_gate_skips_llm(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.9)
    _seed(cfg)
    called = {"llm": False}

    def fake_generate(question, contexts, config):
        called["llm"] = True
        return "should not run"

    answer = ask(
        "How do I paginate?",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.1),
        generate_fn=fake_generate,
    )
    assert answer.used_llm is False
    assert called["llm"] is False
    assert is_refusal(answer.text)
    assert answer.text == DONT_KNOW


def test_above_threshold_calls_llm(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.2)
    _seed(cfg)

    def fake_generate(question, contexts, config):
        return "Use cursor pagination [TIC-001]."

    answer = ask(
        "How do I paginate?",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.8),
        generate_fn=fake_generate,
    )
    assert answer.used_llm is True
    assert "TIC-001" in answer.sources
    assert "cursor" in answer.text.lower()


# ---------------------------------------------------------------------------
# Week 11: ask()'s usage= gap, closed the same way agent.run_agent's was
# ---------------------------------------------------------------------------


def test_ask_usage_meters_only_the_internal_default_generator(tmp_path, monkeypatch):
    from rag_app.llm import Usage

    cfg = make_config(tmp_path, score_threshold=0.2)
    _seed(cfg)
    u = Usage()

    def fake_generate_answer(question, contexts, cfg_, client=None, usage=None):
        if usage is not None:
            usage.add(cfg_.llm.model, 10, 5)
        return "Use cursor pagination [TIC-001]."

    monkeypatch.setattr("rag_app.pipeline.generate_answer", fake_generate_answer)

    ask("How do I paginate?", preset="C", config=cfg,
        embedder=FixedEmbedder(), reranker=FakeReranker(0.8), usage=u)
    assert u.total_tokens == 15


def test_ask_usage_is_bypassed_by_a_custom_generate_fn(tmp_path):
    """Same documented precedent as run_agent's custom llm_fn: a caller's own
    generate_fn is its own instrumentation and has no usage parameter to
    report through."""
    from rag_app.llm import Usage

    cfg = make_config(tmp_path, score_threshold=0.2)
    _seed(cfg)
    u = Usage()

    ask("How do I paginate?", preset="C", config=cfg,
        embedder=FixedEmbedder(), reranker=FakeReranker(0.8), usage=u,
        generate_fn=lambda q, c, cfg_: "Use cursor pagination [TIC-001].")
    assert u.calls == 0
