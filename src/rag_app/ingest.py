from __future__ import annotations

from dataclasses import dataclass, field

from rag_app.chunking import Chunk, chunk_docs
from rag_app.config import AppConfig, load_config
from rag_app.docs import load_docs
from rag_app.embed import Embedder
from rag_app.pdfs import PdfLoadReport, load_pdfs
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset
from rag_app.store import StoreMeta


@dataclass
class IngestReport:
    preset: str
    chunk_size: int
    overlap: int
    n_docs: int
    n_chunks: int
    doc_chunks: int
    avg_chars: float
    mode: str  # "embedded" or "server" — see QdrantStore.mode
    n_pdfs: int = 0
    n_pdf_pages: int = 0
    pdf_chunks: int = 0
    pdf_reports: list[PdfLoadReport] = field(default_factory=list)

    @property
    def unreadable_pdfs(self) -> list[PdfLoadReport]:
        """PDFs that contributed nothing — scanned or encrypted.

        Surfaced because "ingest succeeded, 0 chunks from your 40-page file"
        is otherwise indistinguishable from success.
        """
        return [r for r in self.pdf_reports if r.looks_scanned or r.encrypted]

    def describe(self) -> str:
        parts = []
        if self.n_docs:
            parts.append(f"{self.n_docs} documents ({self.doc_chunks} chunks)")
        if self.n_pdfs:
            parts.append(
                f"{self.n_pdfs} PDFs, {self.n_pdf_pages} pages of text "
                f"({self.pdf_chunks} chunks)"
            )
        warn = ""
        if self.unreadable_pdfs:
            warn = "\n  !! " + "\n  !! ".join(r.describe() for r in self.unreadable_pdfs)
        return (
            f"Preset {self.preset} [qdrant/{self.mode}]: {self.n_chunks} chunks from "
            f"{' + '.join(parts) or 'nothing'}\n"
            f"  chunk_size={self.chunk_size} overlap={self.overlap} "
            f"avg_chars={self.avg_chars:.0f}{warn}"
        )


def run_ingest(
    preset: str | None = None,
    config: AppConfig | None = None,
    *,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Load documents → chunk → embed → persist.

    `embedder` is injectable for the same reason `ask()` takes one: without it
    this whole path is untestable offline.

    Every source is windowed as one file, `.md`/`.txt` and `.pdf` alike, so no
    chunk can ever span two sources. See chunking.py.
    """
    cfg = config or load_config()
    name = preset or cfg.default_preset
    if name not in cfg.chunk_presets:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(cfg.chunk_presets)}")

    preset_cfg = cfg.chunk_presets[name]

    docs = load_docs(cfg.tickets_dir)
    doc_chunks_list = chunk_docs(docs, preset_cfg.chunk_size, preset_cfg.overlap)

    pdf_docs, pdf_reports = load_pdfs(cfg.tickets_dir)
    pdf_chunks_list = chunk_docs(
        pdf_docs, preset_cfg.chunk_size, preset_cfg.overlap, source_type="pdf"
    )

    chunks: list[Chunk] = doc_chunks_list + pdf_chunks_list
    if not chunks:
        unreadable = [r for r in pdf_reports if r.looks_scanned or r.encrypted]
        if unreadable:
            # A PDF-only corpus that extracted nothing is a different problem
            # from an empty directory, and saying "no files found" about a file
            # that is plainly there sends you looking in the wrong place.
            detail = "; ".join(r.describe() for r in unreadable)
            raise ValueError(
                f"Found PDFs in {cfg.tickets_dir} but none yielded any text: {detail}. "
                f"A scanned PDF has no text layer to extract; OCR it first, or add a "
                f"*.md / *.txt source."
            )
        raise ValueError(
            f"No documents (*.md, *.txt, *.pdf) found in {cfg.tickets_dir}"
        )

    emb = embedder or Embedder(cfg.bi_encoder_model)
    vectors = emb.encode_documents([c.text for c in chunks])

    meta = StoreMeta(
        embedding_model=cfg.bi_encoder_model,
        dim=int(vectors.shape[1]),
        chunk_size=preset_cfg.chunk_size,
        overlap=preset_cfg.overlap,
        n_chunks=len(chunks),
    )

    meta_dir = qdrant_path_for_preset(cfg.store_dir, name)
    store = QdrantStore(
        meta_dir=meta_dir,
        path=None if cfg.qdrant.url else meta_dir,
        url=cfg.qdrant.url,
        m=cfg.qdrant.m,
        ef_construct=cfg.qdrant.ef_construct,
        hnsw_ef=cfg.qdrant.hnsw_ef,
    )
    try:
        store.build(chunks, vectors, meta)
        mode = store.mode
    finally:
        store.close()

    return IngestReport(
        preset=name,
        chunk_size=preset_cfg.chunk_size,
        overlap=preset_cfg.overlap,
        n_docs=len(docs),
        n_chunks=len(chunks),
        doc_chunks=len(doc_chunks_list),
        avg_chars=sum(len(c.text) for c in chunks) / len(chunks),
        mode=mode,
        n_pdfs=len(pdf_reports),
        n_pdf_pages=sum(r.n_text_pages for r in pdf_reports),
        pdf_chunks=len(pdf_chunks_list),
        pdf_reports=pdf_reports,
    )
