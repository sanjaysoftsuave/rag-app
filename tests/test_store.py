from pathlib import Path

import numpy as np

from rag_app.chunking import Chunk
from rag_app.store import StoreMeta

from conftest import make_qdrant_store


def test_store_build_reopen_search(tmp_path: Path):
    """The real regression this guards: build, close, reopen a FRESH
    QdrantStore on the same path, then search it. Caught a genuine bug during
    development — embedded Qdrant keeps its own `meta.json` collection
    registry in the same directory, and a naively-named sidecar file
    silently clobbered it, breaking every reopen after the first."""
    chunks = [
        Chunk(chunk_id="a::0", source="a.md", text="alpha"),
        Chunk(chunk_id="b::0", source="b.md", text="beta"),
        Chunk(chunk_id="c::0", source="c.md", text="gamma"),
    ]
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    store = make_qdrant_store(tmp_path, chunks, vectors, name="s1")
    store.close()

    from rag_app.qdrant_store import QdrantStore

    path = tmp_path / "s1"
    reopened = QdrantStore(meta_dir=path, path=path)
    hits = reopened.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk.source == "a.md"
    assert len(hits) == 2
    reopened.close()


def test_provenance_round_trips_through_reopen(tmp_path: Path):
    chunks = [Chunk("a::0", "a.md", "alpha")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    meta = StoreMeta("m", 2, "ticket", 500, 50, 1)
    store = make_qdrant_store(tmp_path, chunks, vectors, meta=meta, name="s2")
    store.close()

    from rag_app.qdrant_store import QdrantStore

    path = tmp_path / "s2"
    reopened = QdrantStore(meta_dir=path, path=path)
    loaded = reopened.load_meta()
    assert loaded is not None
    assert loaded.embedding_model == "m"
    reopened.close()
