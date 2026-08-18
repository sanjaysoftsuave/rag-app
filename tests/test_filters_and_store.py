from pathlib import Path

import numpy as np
import pytest

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter
from rag_app.store import StoreMeta, VectorStore


def _unit(row):
    v = np.asarray(row, dtype=np.float32)
    return v / np.linalg.norm(v)


def _store(tmp_path: Path) -> VectorStore:
    chunks = [
        Chunk("a::0", "TIC-001", "free plan limit", {"product": "API", "customer_tier": "free"}),
        Chunk("b::0", "TIC-002", "pro plan limit", {"product": "API", "customer_tier": "pro"}),
        Chunk("c::0", "TIC-003", "refund timing", {"product": "Billing", "customer_tier": "pro"}),
    ]
    vectors = np.vstack([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 0, 1])])
    return VectorStore(chunks, vectors)


def test_store_roundtrip_preserves_metadata(tmp_path: Path):
    store = _store(tmp_path)
    store.meta = StoreMeta("m", 3, "ticket", 500, 50, 3)
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.chunks[0].metadata["customer_tier"] == "free"
    assert loaded.meta is not None
    assert loaded.meta.embedding_model == "m"


def test_search_orders_by_cosine(tmp_path: Path):
    hits = _store(tmp_path).search(_unit([1, 0, 0]), k=2)
    assert hits[0].chunk.source == "TIC-001"
    assert len(hits) == 2


def test_filter_restricts_candidates_before_ranking(tmp_path: Path):
    store = _store(tmp_path)
    # TIC-001 is the nearest vector, but it is excluded by the filter, so the
    # next-nearest matching chunk must win rather than nothing being returned.
    hits = store.search(_unit([1, 0, 0]), k=3, flt=MetaFilter({"customer_tier": "pro"}))
    assert [h.chunk.source for h in hits] == ["TIC-002", "TIC-003"]


def test_filter_matching_nothing_returns_empty(tmp_path: Path):
    hits = _store(tmp_path).search(_unit([1, 0, 0]), k=3, flt=MetaFilter({"product": "Mobile"}))
    assert hits == []


def test_filter_match_any(tmp_path: Path):
    hits = _store(tmp_path).search(
        _unit([1, 0, 0]), k=3, flt=MetaFilter({"product": ["API", "Billing"]})
    )
    assert len(hits) == 3


def test_filter_matches_list_valued_metadata():
    flt = MetaFilter({"tags": "billing"})
    assert flt.matches({"tags": ["billing", "refund"]})
    assert not flt.matches({"tags": ["api"]})


def test_filter_parse_repeated_key_becomes_match_any():
    flt = MetaFilter.parse(["product=API", "product=Billing", "status=resolved"])
    assert flt.must["product"] == ["API", "Billing"]
    assert flt.must["status"] == "resolved"


def test_filter_parse_rejects_malformed():
    with pytest.raises(ValueError):
        MetaFilter.parse(["productAPI"])
    with pytest.raises(ValueError):
        MetaFilter.parse(["product="])


def test_dimension_mismatch_names_the_real_cause(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="different embedding model"):
        store.search(np.array([1.0, 0.0], dtype=np.float32), k=1)


def test_store_meta_rejects_model_swap():
    meta = StoreMeta("model-a", 384, "ticket", 500, 50, 10)
    meta.assert_compatible_with("model-a")
    with pytest.raises(ValueError, match="not comparable"):
        meta.assert_compatible_with("model-b")


def test_empty_store_and_zero_k(tmp_path: Path):
    empty = VectorStore([], np.zeros((0, 3), dtype=np.float32))
    assert empty.search(_unit([1, 0, 0]), k=5) == []
    assert _store(tmp_path).search(_unit([1, 0, 0]), k=0) == []
