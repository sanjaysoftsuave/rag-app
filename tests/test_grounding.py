"""The grounding guarantees — the part that must never silently regress."""

from __future__ import annotations

import numpy as np
import pytest

from rag_app.chunking import Chunk
from rag_app.embed import MODEL_REGISTRY, spec_for
from rag_app.generate import (
    DONT_KNOW,
    build_prompt,
    cited_sources,
    is_refusal,
)
from rag_app.pipeline import ask
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.rerank import apply_scale, rerank, sigmoid
from rag_app.store import ScoredChunk, StoreMeta

from conftest import FakeReranker, FixedEmbedder, make_config


def _contexts():
    return [
        ScoredChunk(Chunk("a::0", "TIC-001", "Free plan is 60 requests per minute",
                          {"product": "API", "customer_tier": "free"}), 0.9),
        ScoredChunk(Chunk("b::0", "TIC-002", "Pro plan is 600 requests per minute",
                          {"product": "API", "customer_tier": "pro"}), 0.8),
    ]


def _seed_store(cfg, preset="C", texts=(("TIC-001", "Free plan is 60 rpm"),)):
    chunks = [Chunk(f"{tid}::0", tid, text, {"ticket_id": tid}) for tid, text in texts]
    vectors = np.zeros((len(chunks), 2), dtype=np.float32)
    vectors[:, 0] = 1.0
    meta = StoreMeta("fake", 2, "ticket", 2000, 200, len(chunks))
    path = qdrant_path_for_preset(cfg.store_dir, preset)
    store = QdrantStore(meta_dir=path, path=path)
    store.build(chunks, vectors, meta)
    store.close()  # ask() reopens fresh via open_store(), same as real usage


def make_config_with_store(tmp=None, **kw):
    import tempfile
    from pathlib import Path

    base = Path(tmp or tempfile.mkdtemp())
    cfg = make_config(base, **kw)
    _seed_store(cfg)
    return cfg


# --- prompt construction ---------------------------------------------------


def test_prompt_labels_blocks_with_the_id_it_asks_to_be_cited():
    """Labelling blocks [1] while demanding the real label teaches the wrong format."""
    messages = build_prompt("How many requests?", _contexts())
    user = messages[1]["content"]
    assert "[TIC-001]" in user
    assert "[TIC-002]" in user
    assert "[1]" not in user and "[2]" not in user


def test_prompt_does_not_leak_scores_into_context():
    user = build_prompt("q", _contexts())[1]["content"]
    assert "0.9" not in user and "score=" not in user


# --- refusal detection -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        DONT_KNOW,
        "I don't know - that information is not in the provided documents.",  # hyphen
        "I don't know — that information is not in the provided documents.",
        "i don't know, that isn't in the documents",
    ],
)
def test_refusal_survives_punctuation_rewriting(text):
    """An exact == check would misclassify these as real answers."""
    assert is_refusal(text)


def test_real_answer_is_not_a_refusal():
    assert not is_refusal("The Free plan allows 60 requests per minute [TIC-001].")


# --- citation verification -------------------------------------------------


def test_cited_sources_splits_grounded_from_invented():
    grounded, invented = cited_sources(
        "Free is 60 [TIC-001] and Pro is 600 [TIC-002], see also [TIC-999].", _contexts()
    )
    assert grounded == ["TIC-001", "TIC-002"]
    assert invented == ["TIC-999"]


def test_citation_of_a_ticket_never_shown_is_flagged():
    cfg = make_config_with_store()
    answer = ask(
        "How many requests?",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "Answer [TIC-404].",
    )
    assert answer.hallucinated_citations == ["TIC-404"]


# --- the three gates -------------------------------------------------------


def test_gate_below_threshold_never_calls_the_llm():
    cfg = make_config_with_store()
    called = {"llm": False}

    def fake_generate(q, c, k):
        called["llm"] = True
        return "should not run"

    answer = ask(
        "Something unrelated",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.1),
        generate_fn=fake_generate,
    )
    assert answer.used_llm is False
    assert called["llm"] is False
    assert answer.refused is True
    assert answer.gate == "below-threshold"
    assert answer.sources == []
    assert is_refusal(answer.text)


def test_gate_above_threshold_answers_and_cites():
    cfg = make_config_with_store()
    answer = ask(
        "How many requests on Free?",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.8),
        generate_fn=lambda q, c, k: "Free allows 60 requests per minute [TIC-001].",
    )
    assert answer.used_llm is True
    assert answer.refused is False
    assert answer.sources == ["TIC-001"]
    assert answer.hallucinated_citations == []
    assert answer.gate == "answered"


def test_model_refusal_carries_no_sources():
    """A refusal with sources attached credits documents for a non-answer."""
    cfg = make_config_with_store()
    answer = ask(
        "Anything",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: DONT_KNOW,
    )
    assert answer.used_llm is True
    assert answer.refused is True
    assert answer.gate == "model-refused"
    assert answer.sources == []


def test_filter_excluding_everything_refuses_without_the_llm():
    from rag_app.filters import MetaFilter

    cfg = make_config_with_store()
    called = {"llm": False}

    def fake_generate(q, c, k):
        called["llm"] = True
        return "nope"

    answer = ask(
        "How many requests?",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.9),
        generate_fn=fake_generate,
        flt=MetaFilter({"ticket_id": "TIC-DOES-NOT-EXIST"}),
    )
    assert answer.gate == "no-candidates"
    assert answer.used_llm is False
    assert called["llm"] is False


def test_uncited_answer_falls_back_and_records_it():
    cfg = make_config_with_store()
    answer = ask(
        "q",
        preset="C",
        config=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "An answer with no citation at all.",
    )
    assert answer.sources == ["TIC-001"]
    assert answer.meta["cited"] is False


# --- score scaling ---------------------------------------------------------


def test_sigmoid_maps_logits_onto_a_probability_scale():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(-10.0) < 0.001
    assert sigmoid(10.0) > 0.999
    # Must not overflow on an out-of-range model.
    assert 0.0 <= sigmoid(-5000.0) <= 1.0
    assert 0.0 <= sigmoid(5000.0) <= 1.0


def test_scaling_never_reorders_results():
    from rag_app.chunking import Chunk as C

    candidates = [
        ScoredChunk(C("1", "TIC-1", "apples"), 0.0),
        ScoredChunk(C("2", "TIC-2", "oranges"), 0.0),
        ScoredChunk(C("3", "TIC-3", "bananas"), 0.0),
    ]
    scorer = FakeReranker(per_text={"oranges": 5.0, "apples": -1.0, "bananas": 3.0})
    raw = rerank("q", candidates, n=3, scorer=scorer, scale="raw")
    squashed = rerank("q", candidates, n=3, scorer=scorer, scale="sigmoid")
    assert [c.chunk.chunk_id for c in raw] == [c.chunk.chunk_id for c in squashed]
    assert all(0.0 <= c.score <= 1.0 for c in squashed)


def test_apply_scale_rejects_unknown():
    with pytest.raises(ValueError):
        apply_scale([1.0], "logarithmic")


# --- embedding model prefixes ---------------------------------------------


def test_asymmetric_models_declare_their_prefixes():
    e5 = MODEL_REGISTRY["intfloat/e5-small-v2"]
    assert e5.query_prefix == "query: " and e5.passage_prefix == "passage: "

    bge = MODEL_REGISTRY["BAAI/bge-small-en-v1.5"]
    assert bge.query_prefix and not bge.passage_prefix  # query-side instruction only

    mini = MODEL_REGISTRY["sentence-transformers/all-MiniLM-L6-v2"]
    assert not mini.asymmetric


def test_unknown_checkpoint_infers_family_rather_than_assuming_symmetric():
    """Treating an E5 checkpoint as symmetric loses quality with no error."""
    assert spec_for("some-org/my-e5-finetune").query_prefix == "query: "
    assert spec_for("some-org/bge-large-custom").query_prefix
    assert not spec_for("some-org/mystery-model").asymmetric


def test_asymmetric_models_encode_queries_and_passages_differently():
    """The whole reason `encode_queries` and `encode_documents` are separate.

    Using the wrong one on BGE or E5 does not error — it just retrieves worse,
    silently. A fake embedder that ignores the distinction would let that
    regression through, so this asserts on the registry's own rules.
    """
    from rag_app.embed import spec_for

    bge = spec_for("BAAI/bge-small-en-v1.5")
    assert bge.asymmetric
    assert bge.query_prefix and not bge.passage_prefix

    e5 = spec_for("intfloat/e5-small-v2")
    assert e5.query_prefix == "query: " and e5.passage_prefix == "passage: "

    mini = spec_for("sentence-transformers/all-MiniLM-L6-v2")
    assert not mini.asymmetric
