from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_app.tickets import Ticket, render_corpus, render_ticket, ticket_metadata


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy 1 — fixed character windows (the naive baseline)
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split text into overlapping character windows."""
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = text.strip()
    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::{index}",
                    source=source,
                    text=piece,
                    metadata=dict(metadata or {}),
                )
            )
            index += 1
        if end >= len(cleaned):
            break
        start += step

    return chunks


def chunk_flat(tickets: list[Ticket], chunk_size: int, overlap: int) -> list[Chunk]:
    """Treat the whole ticket drop as ONE document and slice it by characters.

    This is the strategy a naive pipeline lands on, and it is wrong for tickets
    in a specific, measurable way: a window that crosses the boundary between
    two tickets gets attributed to whichever ticket it *starts* in, so its
    citation names a ticket that only supplied part of the text.

    Each chunk records `spans_tickets` (how many ticket boundaries it straddles)
    and `bleed` so the damage can be counted instead of argued about. See
    `rag_app.evaluate.boundary_bleed`.
    """
    if not tickets:
        return []

    separator = "\n\n---\n\n"
    corpus = render_corpus(tickets)

    # Character span of every ticket inside the concatenated corpus.
    spans: list[tuple[int, int, Ticket]] = []
    cursor = 0
    for ticket in tickets:
        body = render_ticket(ticket)
        spans.append((cursor, cursor + len(body), ticket))
        cursor += len(body) + len(separator)

    # NOTE: chunk_text strips the corpus before windowing. render_corpus never
    # produces leading whitespace, so offsets line up — but if that changes,
    # these spans silently drift. Assert rather than trust.
    assert corpus == corpus.strip(), "corpus must not have leading/trailing whitespace"

    raw = chunk_text(corpus, source="tickets.jsonl", chunk_size=chunk_size, overlap=overlap)

    chunks: list[Chunk] = []
    offset = 0
    step = chunk_size - overlap
    for i, piece in enumerate(raw):
        start = offset
        end = start + chunk_size
        offset += step

        touched = [t for (s, e, t) in spans if s < end and e > start]
        owner = touched[0] if touched else tickets[0]
        meta = ticket_metadata(owner)
        meta["spans_tickets"] = len(touched)
        meta["bleed"] = len(touched) > 1
        meta["strategy"] = "flat"
        chunks.append(
            Chunk(
                chunk_id=f"flat::{i}",
                source=owner.source,
                text=piece.text,
                metadata=meta,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — one chunk per ticket (structure-aware)
# ---------------------------------------------------------------------------


def chunk_per_ticket(
    tickets: list[Ticket], chunk_size: int, overlap: int
) -> list[Chunk]:
    """One chunk per ticket, splitting only tickets that exceed chunk_size.

    Boundaries can never bleed, so every citation names exactly the ticket the
    text came from. Oversized tickets fall back to character windows *within*
    the ticket, which keeps the guarantee intact.
    """
    chunks: list[Chunk] = []
    for ticket in tickets:
        body = render_ticket(ticket)
        meta = ticket_metadata(ticket)
        meta["strategy"] = "ticket"
        meta["spans_tickets"] = 1
        meta["bleed"] = False

        if len(body) <= chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=f"{ticket.ticket_id}::0",
                    source=ticket.source,
                    text=body,
                    metadata=meta,
                )
            )
            continue

        for part in chunk_text(
            body, source=ticket.source, chunk_size=chunk_size, overlap=overlap, metadata=meta
        ):
            chunks.append(part)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — per conversation turn, with a repeated context header
# ---------------------------------------------------------------------------


def chunk_per_section(
    tickets: list[Ticket], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Pack conversation turns into chunks, repeating the ticket header on each.

    The header costs tokens on every chunk but makes each one self-describing:
    a retrieved turn still knows its product, plan and ticket id, so the LLM is
    never handed a bare sentence with no idea which ticket it belongs to.
    """
    chunks: list[Chunk] = []
    for ticket in tickets:
        header = (
            f"Ticket {ticket.ticket_id}: {ticket.subject}\n"
            f"Product: {ticket.product} | Category: {ticket.category} | "
            f"Plan: {ticket.customer_tier} | Status: {ticket.status}"
        )
        units = [f"{t.role.capitalize()}: {t.text}" for t in ticket.conversation]
        if ticket.resolution:
            units.append(f"Resolution: {ticket.resolution}")

        meta = ticket_metadata(ticket)
        meta["strategy"] = "section"
        meta["spans_tickets"] = 1
        meta["bleed"] = False

        budget = chunk_size - len(header) - 2
        if budget <= 0:
            raise ValueError(
                f"chunk_size {chunk_size} is too small for the header on {ticket.ticket_id} "
                f"({len(header)} chars). Use a larger preset for the 'section' strategy."
            )

        buf: list[str] = []
        index = 0
        for unit in units:
            candidate = "\n".join(buf + [unit])
            if buf and len(candidate) > budget:
                chunks.append(
                    Chunk(
                        chunk_id=f"{ticket.ticket_id}::s{index}",
                        source=ticket.source,
                        text=f"{header}\n\n" + "\n".join(buf),
                        metadata=dict(meta),
                    )
                )
                index += 1
                buf = [unit]
            else:
                buf.append(unit)
        if buf:
            chunks.append(
                Chunk(
                    chunk_id=f"{ticket.ticket_id}::s{index}",
                    source=ticket.source,
                    text=f"{header}\n\n" + "\n".join(buf),
                    metadata=dict(meta),
                )
            )
    return chunks


STRATEGIES = {
    "flat": chunk_flat,
    "ticket": chunk_per_ticket,
    "section": chunk_per_section,
}


def build_chunks(
    tickets: list[Ticket], strategy: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise KeyError(f"Unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](tickets, chunk_size, overlap)
