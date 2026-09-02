"""Documents and PDFs through the real ingest path.

Covers the corpus shapes that have to work now that `.jsonl` tickets are gone:
documents only, PDFs only, and both together in one index — plus the empty
case, which must explain itself rather than producing a store with no chunks.
"""

from __future__ import annotations

import pytest

from rag_app.docs import load_docs
from rag_app.ingest import run_ingest
from rag_app.pipeline import ask, open_store
from rag_app.qdrant_store import QdrantStore, qdrant_path_for_preset

from conftest import FakeEmbedder, FakeReranker, make_config


def _write_doc(cfg, name="policy.md", text="Refunds are issued within five business days."):
    cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tickets_dir / name).write_text(text, encoding="utf-8")


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def _fake_pdf(cfg, name="handbook.pdf", pages=("Page one text.", "Page two text.")):
    """Write a placeholder file and return a reader factory for its pages.

    `run_ingest` calls `load_pdfs` directly, so tests that need real PDF
    ingestion patch the factory; tests that only need the file to exist can
    ignore the returned factory.
    """
    cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tickets_dir / name).write_bytes(b"%PDF-1.4")

    class Reader:
        is_encrypted = False

        def __init__(self, _path):
            self.pages = [FakePage(t) for t in pages]

    return Reader


def test_documents_only_corpus_ingests(tmp_path):
    cfg = make_config(tmp_path)
    _write_doc(cfg)

    report = run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    assert report.n_docs == 1
    assert report.n_chunks == 1
    assert report.doc_chunks == 1
    assert report.n_pdfs == 0

    path = qdrant_path_for_preset(cfg.store_dir, "C")
    store = QdrantStore(meta_dir=path, path=path)
    chunks = store.all_chunks()
    assert chunks[0].source == "policy.md"
    assert chunks[0].metadata["source_type"] == "doc"
    store.close()


def test_pdfs_and_documents_combine_into_one_index(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    _write_doc(cfg)
    reader = _fake_pdf(cfg)
    monkeypatch.setattr("rag_app.pdfs._open_reader", lambda path: reader(path))

    report = run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    assert report.n_docs == 1
    assert report.n_pdfs == 1
    assert report.n_pdf_pages == 2
    assert report.n_chunks == report.doc_chunks + report.pdf_chunks

    path = qdrant_path_for_preset(cfg.store_dir, "C")
    store = QdrantStore(meta_dir=path, path=path)
    chunks = store.all_chunks()
    # The PDF is ONE source, not one per page — its pages were joined first.
    assert {c.source for c in chunks} == {"policy.md", "handbook.pdf"}
    assert {c.metadata["source_type"] for c in chunks} == {"doc", "pdf"}
    store.close()


def test_an_empty_corpus_names_the_formats_it_accepts(tmp_path):
    cfg = make_config(tmp_path)
    cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match=r"No documents") as exc_info:
        run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    message = str(exc_info.value)
    assert ".md" in message and ".txt" in message and ".pdf" in message


def test_a_pdf_with_no_text_layer_says_so_instead_of_failing_vaguely(tmp_path, monkeypatch):
    """'No documents found' about a file that is plainly there sends you
    looking in the wrong place."""
    cfg = make_config(tmp_path)
    reader = _fake_pdf(cfg, "scan.pdf", pages=("", "   "))
    monkeypatch.setattr("rag_app.pdfs._open_reader", lambda path: reader(path))

    with pytest.raises(ValueError, match="none yielded any text") as exc_info:
        run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    assert "scan.pdf" in str(exc_info.value)


def test_load_docs_ignores_pdfs(tmp_path):
    cfg = make_config(tmp_path)
    _write_doc(cfg)
    _fake_pdf(cfg)
    assert [name for name, _ in load_docs(cfg.tickets_dir)] == ["policy.md"]


def test_doc_chunk_is_citable_end_to_end(tmp_path):
    """A filename must work as a real citation — proves generate.py needs no
    special case for documents."""
    cfg = make_config(tmp_path, score_threshold=0.0)
    _write_doc(cfg, "policy.md", "Refunds are issued within five business days of approval.")
    run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    store = open_store(cfg, "C")

    answer = ask(
        "How long do refunds take?",
        preset="C", config=cfg, store=store,
        embedder=FakeEmbedder(), reranker=FakeReranker(0.9),
        generate_fn=lambda q, c, k: "Refunds take five business days [policy.md].",
    )
    assert answer.sources == ["policy.md"]
    assert answer.hallucinated_citations == []
