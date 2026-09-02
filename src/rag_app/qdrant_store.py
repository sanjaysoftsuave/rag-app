"""Qdrant — the only vector store. No in-process numpy fallback.

READ THIS BEFORE QUOTING HNSW NUMBERS
-------------------------------------
`QdrantClient(path=...)` runs Qdrant *embedded*: a pure-Python implementation
that exists for prototyping and tests. It accepts `hnsw_config` and then
ignores it — search is exact brute force, the same algorithm the numpy store
uses. You get the real API, real payload filtering and real persistence, but
NOT a real approximate index.

To actually exercise HNSW you need the server:

    docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

then set `qdrant.url: http://localhost:6333` in config.yaml. The code below is
unchanged either way — only the constructor differs.

WHY HNSW EXISTS
---------------
Brute force is O(N) per query: 36 tickets is nothing, 36 million is a problem.
HNSW (Hierarchical Navigable Small World) builds a layered proximity graph and
greedily walks it, giving roughly O(log N) with ~95-99% recall. The knobs:

  m             edges per node. Higher = better recall, more memory, slower build.
  ef_construct  candidate list size while building. Higher = better graph, slower build.
  hnsw_ef       candidate list size at query time. Higher = better recall, slower query.

The trade you are making is recall for latency — an ANN index can miss a true
neighbour, and `m`/`ef` decide how often.

FILTERING + ANN IS THE HARD PART
--------------------------------
Filtering an exact search is free: shrink the candidate set, then rank.
Filtering a graph index is not. Post-filtering (search then discard) can return
fewer than k results, or none, when the filter is selective. Pre-filtering
breaks the graph's connectivity assumptions. Qdrant's answer is a *filterable*
HNSW that consults payload indexes during traversal, which is why
`create_payload_index` below is not optional bookkeeping — without it, filtered
search degrades badly at scale.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter
from rag_app.store import ScoredChunk, StoreMeta

COLLECTION = "documents"

# Fields we filter on. Qdrant needs an explicit payload index per field.
# What a document corpus actually carries: a chunk's kind, and the file it came
# from. (Ignored in embedded mode, which warns as much; they matter once
# `qdrant.url` points at a real server.)
INDEXED_FIELDS = (
    "source_type",
    "source",
)


def _to_qdrant_filter(flt: MetaFilter | None):
    from qdrant_client import models

    if not flt:
        return None
    conditions = []
    for key, expected in flt.must.items():
        if isinstance(expected, (list, tuple, set)):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(expected)))
            )
        else:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=expected))
            )
    return models.Filter(must=conditions)


class QdrantStore:
    """Vector store backed by Qdrant, embedded or server.

    `meta_dir` is always a local directory, regardless of whether the vectors
    themselves live embedded on disk or on a remote server — the machine
    running this code always has *somewhere* local to keep provenance, so
    `provenance.json` lives there rather than needing a second storage mechanism
    inside Qdrant itself (a reserved point with a dummy vector, a special
    payload key excluded from every real search) for what is fundamentally
    a small local sidecar file, the same role it played for the store this
    replaced.
    """

    def __init__(
        self,
        meta_dir: Path,
        path: Path | None = None,
        url: str | None = None,
        collection: str = COLLECTION,
        m: int = 16,
        ef_construct: int = 100,
        hnsw_ef: int = 128,
    ):
        from qdrant_client import QdrantClient

        self.meta_dir = meta_dir
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        if url:
            self.client = QdrantClient(url=url)
            self.mode = "server"
        else:
            if path is None:
                raise ValueError("QdrantStore needs either a path (embedded) or a url (server)")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.mode = "embedded"
        self.collection = collection
        self.m = m
        self.ef_construct = ef_construct
        self.hnsw_ef = hnsw_ef
        self._meta: StoreMeta | None = None

    # -- build ---------------------------------------------------------------

    def build(self, chunks: list[Chunk], vectors: np.ndarray, meta: StoreMeta) -> None:
        from qdrant_client import models

        dim = int(vectors.shape[1])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
            # Honoured by the server; silently ignored in embedded mode.
            hnsw_config=models.HnswConfigDiff(m=self.m, ef_construct=self.ef_construct),
        )

        for field_name in INDEXED_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True)):
            payload: dict[str, Any] = dict(chunk.metadata)
            payload["chunk_id"] = chunk.chunk_id
            payload["source"] = chunk.source
            payload["text"] = chunk.text
            points.append(
                models.PointStruct(id=i, vector=vec.tolist(), payload=payload)
            )

        # Batched so a large drop does not build one giant request.
        for start in range(0, len(points), 256):
            self.client.upsert(
                collection_name=self.collection, points=points[start : start + 256]
            )
        self._meta = meta
        (self.meta_dir / "provenance.json").write_text(
            json.dumps(asdict(meta), indent=2), encoding="utf-8"
        )

    def load_meta(self) -> StoreMeta | None:
        """Read back the provenance `build()` wrote, or None if never built."""
        return read_provenance(self.meta_dir)

    # -- query ---------------------------------------------------------------

    def search(
        self, query_vec: np.ndarray, k: int, flt: MetaFilter | None = None
    ) -> list[ScoredChunk]:
        from qdrant_client import models

        if k <= 0:
            return []
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.astype(np.float32).reshape(-1).tolist(),
            limit=k,
            query_filter=_to_qdrant_filter(flt),
            search_params=models.SearchParams(hnsw_ef=self.hnsw_ef),
            with_payload=True,
        )
        out: list[ScoredChunk] = []
        for point in result.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.pop("chunk_id", str(point.id))
            source = payload.pop("source", "unknown")
            out.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=chunk_id, source=source, text=text, metadata=payload
                    ),
                    score=float(point.score),
                )
            )
        return out

    def __len__(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def all_chunks(self) -> list[Chunk]:
        """Full corpus dump via `scroll`, paged — what BM25 needs to build a
        hybrid index. Vectors are not fetched; BM25 only needs text+metadata."""
        if not self.client.collection_exists(self.collection):
            return []
        out: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                text = payload.pop("text", "")
                chunk_id = payload.pop("chunk_id", str(point.id))
                source = payload.pop("source", "unknown")
                out.append(Chunk(chunk_id=chunk_id, source=source, text=text, metadata=payload))
            if offset is None:
                break
        return out

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection) and len(self) > 0

    def close(self) -> None:
        # Embedded mode holds a file lock on the storage directory; without
        # closing, a second QdrantStore on the same path raises.
        self.client.close()


def read_provenance(meta_dir: Path) -> StoreMeta | None:
    """Read a store's provenance sidecar WITHOUT opening the database.

    Embedded Qdrant takes a real file lock on open, so anything that only wants
    to know "what settings built this index?" — the UI showing whether the
    current chunking matches what is actually indexed — must not have to
    instantiate a client to find out.

    Deliberately NOT named `meta.json`: in embedded mode `meta_dir` is the
    same directory Qdrant's own local storage uses, and Qdrant already writes
    its OWN `meta.json` there (its collection registry — a
    `{"collections": {...}}` file). A same-named sidecar silently clobbers it,
    and the next `QdrantClient(path=...)` open fails with a
    `KeyError: 'collections'` reading its own corrupted registry. Found by
    actually reopening a store after close(), not by inspection.

    Returns None when the store was never built.
    """
    meta_path = meta_dir / "provenance.json"
    if not meta_path.exists():
        return None
    payload: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    payload.pop("format", None)
    # Drop anything this version of StoreMeta no longer carries (a `strategy`
    # written by an older format, say) rather than dying on an unexpected key.
    # A sidecar is provenance, not a contract: failing to read it turns a
    # cosmetic mismatch into "your whole index is unopenable".
    known = {f.name for f in fields(StoreMeta)}
    return StoreMeta(**{k: v for k, v in payload.items() if k in known})


def qdrant_path_for_preset(store_dir: Path, preset: str) -> Path:
    return store_dir / f"qdrant_{preset}"
