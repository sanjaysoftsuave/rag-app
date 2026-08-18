from __future__ import annotations

from dataclasses import dataclass

from rag_app.chunking import Chunk, build_chunks
from rag_app.config import AppConfig, load_config
from rag_app.embed import Embedder
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.store import StoreMeta, VectorStore, store_path_for_preset
from rag_app.tickets import load_all


@dataclass
class IngestReport:
    preset: str
    strategy: str
    chunk_size: int
    overlap: int
    n_tickets: int
    n_chunks: int
    bleeding_chunks: int
    avg_chars: float
    backend: str

    def describe(self) -> str:
        bleed = ""
        if self.strategy == "flat":
            pct = 100.0 * self.bleeding_chunks / self.n_chunks if self.n_chunks else 0.0
            bleed = (
                f"\n  boundary bleed: {self.bleeding_chunks}/{self.n_chunks} chunks "
                f"({pct:.0f}%) span more than one ticket"
            )
        return (
            f"Preset {self.preset} [{self.backend}]: {self.n_chunks} chunks from "
            f"{self.n_tickets} tickets\n"
            f"  strategy={self.strategy} chunk_size={self.chunk_size} "
            f"overlap={self.overlap} avg_chars={self.avg_chars:.0f}{bleed}"
        )


def run_ingest(
    preset: str | None = None,
    config: AppConfig | None = None,
    *,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Load tickets → chunk → embed → persist.

    `embedder` is injectable for the same reason `ask()` takes one: without it
    this whole path is untestable offline.
    """
    cfg = config or load_config()
    name = preset or cfg.default_preset
    if name not in cfg.chunk_presets:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(cfg.chunk_presets)}")

    preset_cfg = cfg.chunk_presets[name]
    tickets = load_all(cfg.tickets_dir)
    chunks: list[Chunk] = build_chunks(
        tickets, preset_cfg.strategy, preset_cfg.chunk_size, preset_cfg.overlap
    )
    if not chunks:
        raise ValueError(f"Preset {name!r} produced no chunks")

    emb = embedder or Embedder(cfg.bi_encoder_model)
    vectors = emb.encode_documents([c.text for c in chunks])

    meta = StoreMeta(
        embedding_model=cfg.bi_encoder_model,
        dim=int(vectors.shape[1]),
        strategy=preset_cfg.strategy,
        chunk_size=preset_cfg.chunk_size,
        overlap=preset_cfg.overlap,
        n_chunks=len(chunks),
    )

    if cfg.backend == "qdrant":
        path = qdrant_path_for_preset(cfg.store_dir, name)
        store = QdrantStore(
            path=None if cfg.qdrant.url else path,
            url=cfg.qdrant.url,
            m=cfg.qdrant.m,
            ef_construct=cfg.qdrant.ef_construct,
            hnsw_ef=cfg.qdrant.hnsw_ef,
        )
        try:
            store.build(chunks, vectors, meta)
        finally:
            store.close()
    else:
        VectorStore(chunks=chunks, vectors=vectors, meta=meta).save(
            store_path_for_preset(cfg.store_dir, name)
        )

    return IngestReport(
        preset=name,
        strategy=preset_cfg.strategy,
        chunk_size=preset_cfg.chunk_size,
        overlap=preset_cfg.overlap,
        n_tickets=len(tickets),
        n_chunks=len(chunks),
        bleeding_chunks=sum(1 for c in chunks if c.metadata.get("bleed")),
        avg_chars=sum(len(c.text) for c in chunks) / len(chunks),
        backend=cfg.backend,
    )
