import numpy as np

from rag_app.chunking import Chunk
from rag_app.rerank import rerank, stub_scorer
from rag_app.retrieve import retrieve
from rag_app.store import ScoredChunk, VectorStore


def test_retrieve_orders_by_cosine():
    chunks = [
        Chunk("1", "a.md", "one"),
        Chunk("2", "b.md", "two"),
        Chunk("3", "c.md", "three"),
    ]
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        dtype=np.float32,
    )
    # normalize last row for fair cosine
    vectors[2] = vectors[2] / np.linalg.norm(vectors[2])
    store = VectorStore(chunks, vectors)
    hits = retrieve(store, np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk.chunk_id == "1"
    assert len(hits) == 2


def test_rerank_reorders_with_stub():
    candidates = [
        ScoredChunk(Chunk("1", "a.md", "mentions apples here"), 0.9),
        ScoredChunk(Chunk("2", "b.md", "mentions oranges here"), 0.8),
        ScoredChunk(Chunk("3", "c.md", "mentions bananas here"), 0.7),
    ]
    scorer = stub_scorer({"oranges": 5.0, "apples": 1.0, "bananas": 3.0})
    top = rerank("fruit?", candidates, n=2, scorer=scorer)
    assert [c.chunk.chunk_id for c in top] == ["2", "3"]
    assert top[0].score == 5.0
