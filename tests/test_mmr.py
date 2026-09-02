"""Maximal Marginal Relevance.

The invariant that matters most is the boring one: MMR must not move the score
gate. It changes what the LLM reads, never whether it is called.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_app.chunking import Chunk
from rag_app.mmr import apply_mmr, cosine_matrix, mmr_select
from rag_app.store import ScoredChunk


def sc(source: str, score: float, text: str = "text") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"{source}::0", source=source, text=text, metadata={}),
        score=score,
    )


def sim_from(pairs: dict[tuple[int, int], float], n: int) -> np.ndarray:
    m = np.eye(n, dtype=np.float32)
    for (i, j), v in pairs.items():
        m[i][j] = m[j][i] = v
    return m


# --- the gate invariant ----------------------------------------------------


def test_the_first_pick_is_always_the_most_relevant():
    """`pipeline.ask()` reads reranked[0].score for the gate. If MMR could
    demote the best candidate, enabling it would silently change which
    questions get refused."""
    cands = [sc("A", 0.99), sc("B", 0.98), sc("C", 0.97)]
    # A and B are identical; naive diversity would want to drop one of them.
    sim = sim_from({(0, 1): 1.0, (0, 2): 0.0, (1, 2): 0.0}, 3)
    for lam in (0.0, 0.3, 0.7, 1.0):
        picked = mmr_select(cands, sim, n=2, lambda_=lam)
        assert picked[0].chunk.source == "A", f"lambda={lam} moved the top candidate"


# --- the actual behaviour --------------------------------------------------


def test_a_near_duplicate_is_dropped_for_something_new():
    cands = [sc("A", 0.99), sc("A2", 0.98), sc("B", 0.50)]
    # A2 is a near-copy of A; B is unrelated but less relevant.
    sim = sim_from({(0, 1): 0.99, (0, 2): 0.05, (1, 2): 0.05}, 3)

    without = [c.chunk.source for c in mmr_select(cands, sim, n=2, lambda_=1.0)]
    with_mmr = [c.chunk.source for c in mmr_select(cands, sim, n=2, lambda_=0.5)]

    assert without == ["A", "A2"], "lambda=1.0 must behave exactly like no MMR"
    assert with_mmr == ["A", "B"], "diversity should prefer the new information"


def test_lambda_one_is_identical_to_plain_truncation():
    cands = [sc(chr(65 + i), 1.0 - i / 10) for i in range(6)]
    sim = np.ones((6, 6), dtype=np.float32)  # everything identical
    picked = mmr_select(cands, sim, n=3, lambda_=1.0)
    assert [c.chunk.source for c in picked] == ["A", "B", "C"]


def test_lambda_zero_maximises_novelty_after_the_first_pick():
    cands = [sc("A", 0.99), sc("A2", 0.98), sc("B", 0.10)]
    sim = sim_from({(0, 1): 0.95, (0, 2): 0.01, (1, 2): 0.01}, 3)
    picked = mmr_select(cands, sim, n=2, lambda_=0.0)
    assert [c.chunk.source for c in picked] == ["A", "B"]


# --- edges -----------------------------------------------------------------


def test_asking_for_more_than_exists_returns_everything():
    cands = [sc("A", 0.9), sc("B", 0.8)]
    picked = mmr_select(cands, np.eye(2, dtype=np.float32), n=10)
    assert len(picked) == 2


def test_no_candidates_or_no_slots_is_empty():
    assert mmr_select([], np.zeros((0, 0)), n=3) == []
    assert mmr_select([sc("A", 0.9)], np.eye(1), n=0) == []
    assert apply_mmr([], np.zeros((0, 8)), n=3) == []


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_lambda_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="lambda_"):
        mmr_select([sc("A", 0.9)], np.eye(1), n=1, lambda_=bad)


def test_selection_never_repeats_a_candidate():
    cands = [sc(chr(65 + i), 1.0 - i / 10) for i in range(5)]
    rng = np.random.default_rng(0)
    v = rng.normal(size=(5, 8)).astype(np.float32)
    picked = apply_mmr(cands, v, n=4, lambda_=0.5)
    ids = [c.chunk.chunk_id for c in picked]
    assert len(ids) == len(set(ids)) == 4


# --- similarity ------------------------------------------------------------


def test_cosine_matrix_is_symmetric_with_a_unit_diagonal():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(4, 16)).astype(np.float32)
    m = cosine_matrix(v)
    assert np.allclose(np.diag(m), 1.0, atol=1e-5)
    assert np.allclose(m, m.T, atol=1e-6)


def test_cosine_matrix_normalizes_unnormalized_input():
    """A caller passing raw vectors would otherwise get similarities above 1.0
    and a silently over-weighted redundancy penalty."""
    v = np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    m = cosine_matrix(v)
    assert m.max() <= 1.0 + 1e-6
    assert m[0][1] == pytest.approx(0.0, abs=1e-6)


def test_a_zero_vector_does_not_divide_by_zero():
    v = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    m = cosine_matrix(v)
    assert np.isfinite(m).all()
