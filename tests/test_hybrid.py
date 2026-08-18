import numpy as np

from rag_app.bm25 import BM25Index
from rag_app.chunking import Chunk
from rag_app.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from rag_app.store import ScoredChunk, VectorStore


def _sc(chunk_id: str, source: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id, source, "text"), score=score)


def test_rrf_agreement_beats_a_single_strong_ranking():
    """A doc ranked #2 in BOTH lists should beat a doc ranked #1 in only one —
    that is the entire point of fusing two independent signals."""
    dense = [_sc("x", "X", 0.9), _sc("agree", "AGREE", 0.8)]
    bm25 = [_sc("agree", "AGREE", 10.0), _sc("y", "Y", 8.0)]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert fused[0].chunk.source == "AGREE"


def test_rrf_surfaces_a_document_found_by_only_one_ranking():
    dense = [_sc("a", "A", 0.9)]
    bm25 = [_sc("keyword-only", "KW", 5.0)]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert {r.chunk.source for r in fused} == {"A", "KW"}


def test_rrf_dedupes_by_chunk_id_not_by_object_identity():
    dense = [_sc("dup", "DUP", 0.9)]
    bm25 = [_sc("dup", "DUP", 7.0)]
    fused = reciprocal_rank_fusion([dense, bm25])
    assert len(fused) == 1
    # rank 1 in both -> 1/61 + 1/61
    assert fused[0].score > (1.0 / 61)


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_k_changes_how_much_low_ranks_matter():
    dense = [_sc("a", "A", 0.9), _sc("b", "B", 0.8)]
    tight = reciprocal_rank_fusion([dense], k_rrf=1)
    loose = reciprocal_rank_fusion([dense], k_rrf=1000)
    # small k_rrf spreads rank-1 vs rank-2 further apart than large k_rrf
    tight_gap = tight[0].score - tight[1].score
    loose_gap = loose[0].score - loose[1].score
    assert tight_gap > loose_gap


def _unit(row):
    v = np.asarray(row, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_hybrid_retrieve_surfaces_a_keyword_only_match():
    """A ticket that shares almost no embedding-space similarity with the
    query but contains its exact id should still surface via BM25."""
    chunks = [
        Chunk("a::0", "TIC-1001", "unrelated filler text about something else entirely"),
        Chunk("b::0", "TIC-9999", "this chunk literally contains the string ZQXK-7777"),
    ]
    # Vectors deliberately favor TIC-1001 in embedding space.
    vectors = np.vstack([_unit([1, 0]), _unit([0, 1])])
    store = VectorStore(chunks, vectors)
    bm25 = BM25Index(chunks)

    query_vec = _unit([1, 0])  # dense search alone would prefer TIC-1001
    results = hybrid_retrieve(store, bm25, query_vec, "ZQXK-7777", k=2)
    sources = [r.chunk.source for r in results]
    assert "TIC-9999" in sources, "the exact-id match must surface via the keyword side of the fusion"


def test_hybrid_retrieve_respects_filter():
    from rag_app.filters import MetaFilter

    chunks = [
        Chunk("a::0", "TIC-1001", "rate limit free plan", {"customer_tier": "free"}),
        Chunk("b::0", "TIC-1002", "rate limit pro plan", {"customer_tier": "pro"}),
    ]
    vectors = np.vstack([_unit([1, 0]), _unit([1, 0.01])])
    store = VectorStore(chunks, vectors)
    bm25 = BM25Index(chunks)

    results = hybrid_retrieve(
        store, bm25, _unit([1, 0]), "rate limit", k=5,
        flt=MetaFilter({"customer_tier": "pro"}),
    )
    assert all(r.chunk.metadata["customer_tier"] == "pro" for r in results)


def test_hybrid_retrieve_truncates_to_k():
    chunks = [Chunk(f"{i}::0", f"TIC-{i}", f"rate limit ticket number {i}") for i in range(10)]
    vectors = np.vstack([_unit([1, i * 0.01]) for i in range(10)])
    store = VectorStore(chunks, vectors)
    bm25 = BM25Index(chunks)
    results = hybrid_retrieve(store, bm25, _unit([1, 0]), "rate limit ticket", k=3)
    assert len(results) == 3
