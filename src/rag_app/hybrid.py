"""Hybrid search — fuse a dense ranking and a BM25 ranking into one.

Keyword vs semantic search are not competitors; they fail in complementary
ways. Dense retrieval finds "why do I keep getting cut off" when the ticket
says "rate limit exceeded" — no shared words, pure meaning match. BM25 finds
"TIC-1001" or "ERR-4032" — an exact token dense retrieval treats as just
another point in embedding space, with no special pull toward literal matches.
Neither ranking alone sees what the other sees; fusing them is strictly more
information than either.

RECIPROCAL RANK FUSION (RRF)
-----------------------------
The naive way to combine two rankings is to normalize and add their scores —
but a cosine similarity and a BM25 score live on unrelated, incomparable
scales (roughly 0-1 vs unbounded and corpus-dependent), so any such blend is
really an arbitrary weighting in disguise.

RRF sidesteps this by discarding the scores and fusing on RANK POSITION only:

    RRF(d) = sum over rankings r that contain d of:  1 / (k_rrf + rank_r(d))

rank_r(d) is d's 1-based position in ranking r (best = 1). A document ranked
#1 in one list and absent from the other still scores well; a document ranked
#2 in BOTH lists usually beats a document ranked #1 in only one, because it
had two independent signals agree on it. `k_rrf` (60 is the standard choice
from the original TREC paper) discounts how much low ranks matter — it flattens
the curve so #1 vs #2 matters more than #50 vs #51.

This is the ONE change this module makes to retrieval. It fuses two rankings;
it does not touch chunking, embedding, the cross-encoder, or the score gate —
so a before/after hit-rate comparison isolates exactly this fusion's effect.
"""

from __future__ import annotations

import numpy as np

from rag_app.bm25 import BM25Index
from rag_app.filters import MetaFilter
from rag_app.store import ScoredChunk, SearchBackend

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], k_rrf: int = DEFAULT_RRF_K
) -> list[ScoredChunk]:
    """Merge N already-sorted rankings by rank position, not by score.

    A chunk's identity across rankings is its `chunk_id` — the same chunk
    retrieved by both the dense and the keyword ranking must be recognized as
    one document, not counted as two unrelated hits.
    """
    fused: dict[str, float] = {}
    owner: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank)
            owner.setdefault(cid, item)

    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    return [
        ScoredChunk(chunk=owner[cid].chunk, score=score) for cid, score in ordered
    ]


def hybrid_retrieve(
    store: SearchBackend,
    bm25: BM25Index,
    query_vec: np.ndarray,
    query_text: str,
    k: int,
    flt: MetaFilter | None = None,
    dense_pool: int | None = None,
    bm25_pool: int | None = None,
    k_rrf: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """Dense search + BM25 search, fused by RRF, truncated to top-k.

    The pools feeding fusion are deliberately wider than `k`: a chunk ranked
    #8 in one list and unranked in the other can still fuse into the top-k
    once combined with a strong position in the other list, but only if it
    was in the pool to begin with.
    """
    dense_pool = dense_pool or max(k * 2, 10)
    bm25_pool = bm25_pool or max(k * 2, 10)

    dense_ranking = store.search(query_vec, k=dense_pool, flt=flt)
    bm25_ranking = bm25.search(query_text, k=bm25_pool, flt=flt)

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k_rrf=k_rrf)
    return fused[:k]
