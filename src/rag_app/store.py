from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter

STORE_FORMAT = 2


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class SearchBackend(Protocol):
    """What the pipeline needs from a vector store, regardless of vendor."""

    def search(
        self, query_vec: np.ndarray, k: int, flt: MetaFilter | None = ...
    ) -> list[ScoredChunk]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class StoreMeta:
    """Provenance for a built index.

    Without this, changing `bi_encoder_model` in config.yaml and forgetting to
    re-ingest either explodes with a numpy shape error or — when the old and
    new models happen to share a dimension, which MiniLM-class models very
    often do — silently returns garbage rankings with no error at all.
    """

    embedding_model: str
    dim: int
    strategy: str
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


class VectorStore:
    """Brute-force cosine search over a numpy matrix.

    Exact by construction: it scores every vector, so recall is 100% and
    filtering is applied *before* ranking with no approximation. That makes it
    the honest baseline to measure an ANN index (see `qdrant_store`) against.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        meta: StoreMeta | None = None,
    ):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.meta = meta

    def __len__(self) -> int:
        return len(self.chunks)

    def all_chunks(self) -> list[Chunk]:
        """Full corpus dump — what BM25 needs to compute document frequencies."""
        return self.chunks

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = [asdict(c) for c in self.chunks]
        (directory / "chunks.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if self.meta is not None:
            (directory / "meta.json").write_text(
                json.dumps(asdict(self.meta), indent=2), encoding="utf-8"
            )

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        vectors = np.load(directory / "vectors.npy")
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
        meta = None
        meta_path = directory / "meta.json"
        if meta_path.exists():
            payload: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
            payload.pop("format", None)
            meta = StoreMeta(**payload)
        return cls(chunks=chunks, vectors=vectors, meta=meta)

    def search(
        self, query_vec: np.ndarray, k: int, flt: MetaFilter | None = None
    ) -> list[ScoredChunk]:
        if len(self.chunks) == 0 or k <= 0:
            return []
        q = query_vec.astype(np.float32).reshape(-1)
        if q.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Query vector has dimension {q.shape[0]} but the store holds "
                f"{self.vectors.shape[1]}-dimensional vectors. The store was almost "
                f"certainly built with a different embedding model — re-ingest."
            )

        # Pre-filter: restrict the candidate set, THEN rank. Exact search can do
        # this for free. An HNSW index cannot — see qdrant_store for why that
        # distinction matters once the corpus is large.
        if flt:
            keep = [i for i, c in enumerate(self.chunks) if flt.matches(c.metadata)]
            if not keep:
                return []
            idx = np.asarray(keep, dtype=np.int64)
            scores = self.vectors[idx] @ q
        else:
            idx = None
            scores = self.vectors @ q

        # vectors are L2-normalized → cosine == dot product
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            ScoredChunk(
                chunk=self.chunks[int(idx[i]) if idx is not None else int(i)],
                score=float(scores[i]),
            )
            for i in top
        ]


def store_path_for_preset(store_dir: Path, preset: str) -> Path:
    return store_dir / f"preset_{preset}"
