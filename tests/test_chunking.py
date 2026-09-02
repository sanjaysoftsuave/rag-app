"""Windowing — the one chunking method the app has.

The property worth protecting is structural: every source is windowed on its
own, so no chunk can ever contain text from two sources and therefore no
citation can ever name a source that supplied only part of it.
"""

from __future__ import annotations

import pytest

from rag_app.chunking import chunk_doc, chunk_docs, chunk_text


# --- chunk_text ------------------------------------------------------------


def test_short_text_is_one_chunk():
    chunks = chunk_text("Refunds take five days.", "policy.md", 500, 50)
    assert len(chunks) == 1
    assert chunks[0].text == "Refunds take five days."
    assert chunks[0].source == "policy.md"
    assert chunks[0].chunk_id == "policy.md::0"


def test_long_text_splits_with_overlap():
    chunks = chunk_text("x" * 1000, "big.txt", chunk_size=400, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    # step = size - overlap, so windows advance by 300 and share 100 characters.
    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_chunk_ids_are_sequential_and_scoped_to_the_source():
    chunks = chunk_text("y" * 900, "notes.txt", 300, 0)
    assert [c.chunk_id for c in chunks] == [f"notes.txt::{i}" for i in range(len(chunks))]


def test_empty_or_whitespace_text_produces_nothing():
    assert chunk_text("", "a.md", 500, 50) == []
    assert chunk_text("   \n\n ", "a.md", 500, 50) == []


@pytest.mark.parametrize(
    "size,overlap,match",
    [
        (0, 0, "chunk_size must be positive"),
        (-1, 0, "chunk_size must be positive"),
        (100, -1, "overlap must be >= 0"),
        (100, 100, "overlap must be smaller"),
        (100, 200, "overlap must be smaller"),
    ],
)
def test_invalid_sizes_are_rejected(size, overlap, match):
    """overlap >= chunk_size would never advance — an infinite loop, not a
    degraded result. The UI slider is capped for the same reason."""
    with pytest.raises(ValueError, match=match):
        chunk_text("some text", "a.md", size, overlap)


def test_metadata_is_copied_not_shared():
    """Chunks must not alias one dict; mutating one would rewrite the others."""
    chunks = chunk_text("z" * 900, "a.md", 300, 0, metadata={"source_type": "doc"})
    assert len(chunks) > 1
    chunks[0].metadata["source_type"] = "changed"
    assert chunks[1].metadata["source_type"] == "doc"


# --- chunk_doc / chunk_docs ------------------------------------------------


def test_chunk_doc_labels_the_source_type():
    chunks = chunk_doc("policy.md", "Refunds are issued within five business days.", 500, 50)
    assert chunks[0].source == "policy.md"
    assert chunks[0].metadata["source_type"] == "doc"


def test_documents_never_contaminate_each_other():
    """The structural guarantee: two files chunked together never share a window."""
    docs = [("a.md", "A" * 900), ("b.md", "B" * 900)]
    chunks = chunk_docs(docs, chunk_size=400, overlap=0)
    for chunk in chunks:
        # Every chunk's text came from exactly one file.
        assert len(set(chunk.text)) == 1
        assert chunk.source == ("a.md" if chunk.text[0] == "A" else "b.md")


def test_chunk_docs_of_nothing_is_nothing():
    assert chunk_docs([], 500, 50) == []


def test_extra_metadata_rides_along():
    chunks = chunk_doc("h.pdf", "text", 500, 50, source_type="pdf",
                       extra={"origin": "upload"})
    assert chunks[0].metadata == {"source_type": "pdf", "origin": "upload"}


# --- source_type -----------------------------------------------------------


def test_pdfs_go_through_the_same_path_with_only_a_different_label():
    """A PDF is not a special case — same function, one keyword apart."""
    chunks = chunk_docs([("handbook.pdf", "Refunds take five days.")], 500, 50,
                        source_type="pdf")
    assert chunks[0].source == "handbook.pdf"
    assert chunks[0].metadata["source_type"] == "pdf"


def test_a_long_file_splits_but_every_chunk_cites_the_same_file():
    chunks = chunk_docs([("long.pdf", "x" * 1200)], 500, 0, source_type="pdf")
    assert len(chunks) > 1
    assert {c.source for c in chunks} == {"long.pdf"}
