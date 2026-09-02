"""Maximal Marginal Relevance — trade a little relevance for coverage.

THE PROBLEM
-----------
Rerank sorts by relevance and the top N go to the LLM. If the three most
relevant chunks all say the same thing — easy to arrange in a corpus with
repeating structure, boilerplate headers, or overlapping windows — the model
gets one fact stated three times and the second-best *distinct* fact never
reaches it. Precision is fine; coverage is not.

THE FORMULA
-----------
Pick greedily. The first pick is simply the most relevant. Every pick after
that maximises:

    MMR(d) = λ · relevance(d)  −  (1 − λ) · max  similarity(d, s)
                                          s ∈ selected

so a candidate is penalised for resembling something already chosen.

    λ = 1.0   pure relevance — identical to no MMR at all
    λ = 0.0   pure novelty — actively avoids the query
    λ = 0.7   default: relevance-led, breaks ties toward new information

WHY THE FIRST PICK MATTERS
--------------------------
Standard MMR selects the highest-relevance candidate first, and that is
load-bearing here rather than incidental: `pipeline.ask()` reads
`reranked[0].score` to decide the score gate. Because position 0 is still the
best-scoring candidate, adding MMR cannot move the gate. Selection changes what
the LLM *reads*, never whether it is *called*.
"""

from __future__ import annotations

import numpy as np

from rag_app.store import ScoredChunk


DEFAULT_LAMBDA = 0.7


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, assuming L2-normalized rows.

    `Embedder` normalizes everything it produces, so this is a plain dot
    product — the same assumption `QdrantStore` relies on for Distance.COSINE.
    Normalizes defensively anyway: a caller passing raw vectors would otherwise
    get similarities above 1.0 and a silently wrong penalty term.
    """
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def mmr_select(
    candidates: list[ScoredChunk],
    similarity: np.ndarray,
    n: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[ScoredChunk]:
    """Greedily pick `n` of `candidates`, balancing relevance against novelty.

    `candidates` must already be sorted best-first (rerank output), and
    `similarity[i][j]` is the similarity between candidates i and j.

    Relevance comes from each candidate's position, not its score: rerank
    scores on the sigmoid scale saturate near 1.0 (0.9874 and 0.9991 are
    "equally relevant" to any practical purpose), so subtracting a similarity
    penalty from them would be comparing a flattened axis against a linear one
    and diversity would almost always win. Rank is the honest signal — it is
    what the reranker actually asserted.
    """
    if not candidates or n <= 0:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be between 0 and 1, got {lambda_}")

    count = len(candidates)
    n = min(n, count)
    # Rank 1 -> 1.0, rank `count` -> ~0.0. Linear, so the penalty is comparable.
    relevance = [1.0 - (i / count) for i in range(count)]

    # The first pick is the most relevant, which keeps `selected[0]` identical
    # to `candidates[0]` and therefore leaves the score gate untouched.
    selected = [0]
    remaining = list(range(1, count))

    while len(selected) < n and remaining:
        best_index, best_score = remaining[0], float("-inf")
        for i in remaining:
            redundancy = max(float(similarity[i][s]) for s in selected)
            score = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        remaining.remove(best_index)

    return [candidates[i] for i in selected]


def apply_mmr(
    candidates: list[ScoredChunk],
    vectors: np.ndarray,
    n: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[ScoredChunk]:
    """`mmr_select` with the similarity matrix computed from chunk vectors."""
    if not candidates:
        return []
    return mmr_select(candidates, cosine_matrix(vectors), n, lambda_)
