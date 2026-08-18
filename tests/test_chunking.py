import pytest

from rag_app.chunking import build_chunks, chunk_flat, chunk_per_section, chunk_per_ticket, chunk_text
from rag_app.evaluate import boundary_bleed

from conftest import make_ticket


def test_empty_text_returns_empty():
    assert chunk_text("", "a.md", 100, 10) == []
    assert chunk_text("   ", "a.md", 100, 10) == []


def test_metadata_and_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    chunks = chunk_text(text, "doc.md", chunk_size=10, overlap=3)
    assert chunks[0].source == "doc.md"
    assert chunks[0].chunk_id == "doc.md::0"
    assert chunks[0].text == "abcdefghij"
    # next window starts at 10-3=7 → "hijklmnopq"
    assert chunks[1].text.startswith("hij")
    assert len(chunks) >= 2


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", "x.md", chunk_size=5, overlap=5)


def test_flat_strategy_bleeds_across_ticket_boundaries(tickets):
    """The core defect: a character window does not respect ticket edges."""
    chunks = chunk_flat(tickets, chunk_size=200, overlap=20)
    bleeding, total = boundary_bleed(chunks)
    assert total > 0
    assert bleeding > 0, "small windows over a concatenated corpus must straddle tickets"
    # Every bleeding chunk is cited as ONE ticket while containing text from more.
    for chunk in chunks:
        if chunk.metadata["bleed"]:
            assert chunk.metadata["spans_tickets"] > 1


def test_ticket_strategy_never_bleeds(tickets):
    chunks = chunk_per_ticket(tickets, chunk_size=2000, overlap=200)
    bleeding, total = boundary_bleed(chunks)
    assert bleeding == 0
    assert total == len(tickets)
    assert {c.source for c in chunks} == {t.ticket_id for t in tickets}


def test_section_strategy_repeats_header_on_every_chunk(tickets):
    chunks = chunk_per_section(tickets, chunk_size=700, overlap=0)
    assert boundary_bleed(chunks)[0] == 0
    for chunk in chunks:
        # Each fragment must name its own ticket, or a retrieved turn is
        # unattributable once it reaches the prompt.
        assert chunk.source in chunk.text


def test_oversized_ticket_splits_within_its_own_boundary():
    big = make_ticket("TIC-BIG", "x", resolution="y " * 2000)
    chunks = chunk_per_ticket([big], chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(c.source == "TIC-BIG" for c in chunks)
    assert all(not c.metadata["bleed"] for c in chunks)


def test_section_strategy_rejects_chunk_size_smaller_than_header(tickets):
    with pytest.raises(ValueError, match="too small for the header"):
        chunk_per_section(tickets, chunk_size=40, overlap=0)


def test_metadata_travels_onto_every_chunk(tickets):
    for strategy in ("flat", "ticket", "section"):
        chunks = build_chunks(tickets, strategy, 700, 50)
        for chunk in chunks:
            assert chunk.metadata["strategy"] == strategy
            assert chunk.metadata["product"]
            assert chunk.metadata["ticket_id"]


def test_unknown_strategy_is_rejected(tickets):
    with pytest.raises(KeyError):
        build_chunks(tickets, "nope", 500, 50)
