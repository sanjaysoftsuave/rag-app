from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter

STORE_FORMAT = 3


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class SearchBackend(Protocol):
    """What the pipeline needs from a vector store.

    Qdrant is the only implementation ([qdrant_store.py](qdrant_store.py)) —
    this Protocol exists so every caller (pipeline.py, bm25.py, ui.py...)
    depends on a small, explicit shape instead of importing
    QdrantStore directly, and so a future second backend has a contract to
    implement rather than one to reverse-engineer from QdrantStore's internals.
    """

    def search(
        self, query_vec: np.ndarray, k: int, flt: MetaFilter | None = ...
    ) -> list[ScoredChunk]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class StoreMeta:
    """Provenance for a built index.

    Without this, changing `bi_encoder_model` in config.yaml and forgetting to
    re-ingest either fails with a dimension-mismatch error from the store
    itself, or — when the old and new models happen to share a dimension,
    which MiniLM-class models very often do — silently returns garbage
    rankings with no error at all. `QdrantStore` persists this as a
    `provenance.json` sidecar next to the collection; `open_store()` reads it back
    and refuses to proceed on a mismatch, whether the collection is local
    (embedded) or on a remote server.
    """

    embedding_model: str
    dim: int
    chunk_size: int
    overlap: int
    n_chunks: int
    format: int = STORE_FORMAT

    def assert_compatible_with(self, embedding_model: str) -> None:
        if self.embedding_model != embedding_model:
            raise ValueError(
                f"Store was built with embedding model {self.embedding_model!r} but config "
                f"asks for {embedding_model!r}. Vectors from different models are not "
                f"comparable. Re-ingest this preset."
            )
