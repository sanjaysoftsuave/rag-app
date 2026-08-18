from pathlib import Path

import numpy as np

from rag_app.chunking import Chunk
from rag_app.store import VectorStore


def test_store_save_load_search(tmp_path: Path):
    chunks = [
        Chunk(chunk_id="a::0", source="a.md", text="alpha"),
        Chunk(chunk_id="b::0", source="b.md", text="beta"),
        Chunk(chunk_id="c::0", source="c.md", text="gamma"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store = VectorStore(chunks=chunks, vectors=vectors)
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    hits = loaded.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk.source == "a.md"
    assert len(hits) == 2
