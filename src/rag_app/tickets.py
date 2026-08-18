"""Load and render customer-support tickets.

A ticket is the natural retrieval unit for a help-centre corpus: it has a
boundary, an outcome, and metadata worth filtering on. This module turns the
raw JSONL drop into text a bi-encoder can embed, plus the metadata dict that
travels with every chunk downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Fields promoted to chunk metadata for filtering. Kept explicit rather than
# "everything in the JSON" so a schema change in the drop cannot silently
# widen the filterable surface.
METADATA_FIELDS = (
    "ticket_id",
    "product",
    "category",
    "status",
    "priority",
    "channel",
    "customer_tier",
    "created_at",
)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    subject: str
    product: str
    category: str
    status: str
    priority: str
    channel: str
    created_at: str
    customer_tier: str
    conversation: list[Turn]
    resolution: str
    tags: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        """Citation label. Stable, human-meaningful, and unique per ticket."""
        return self.ticket_id


def _parse(raw: dict[str, Any]) -> Ticket:
    return Ticket(
        ticket_id=str(raw["ticket_id"]),
        subject=str(raw["subject"]),
        product=str(raw["product"]),
        category=str(raw["category"]),
        status=str(raw["status"]),
        priority=str(raw["priority"]),
        channel=str(raw.get("channel", "unknown")),
        created_at=str(raw.get("created_at", "")),
        customer_tier=str(raw.get("customer_tier", "unknown")),
        conversation=[
            Turn(role=str(t["role"]), text=str(t["text"]))
            for t in raw.get("conversation", [])
        ],
        resolution=str(raw.get("resolution", "")),
        tags=[str(t) for t in raw.get("tags", [])],
    )


def load_tickets(path: Path) -> list[Ticket]:
    """Read a JSONL ticket drop. Blank lines are skipped; bad lines are loud."""
    tickets: list[Ticket] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            tickets.append(_parse(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path.name}:{lineno} is not a valid ticket: {exc}") from exc
    if not tickets:
        raise ValueError(f"No tickets found in {path}")
    return tickets


def iter_ticket_files(tickets_dir: Path) -> Iterator[Path]:
    yield from sorted(tickets_dir.glob("*.jsonl"))


def load_all(tickets_dir: Path) -> list[Ticket]:
    tickets: list[Ticket] = []
    for path in iter_ticket_files(tickets_dir):
        tickets.extend(load_tickets(path))
    if not tickets:
        raise FileNotFoundError(f"No .jsonl ticket files in {tickets_dir}")
    return tickets


def render_ticket(ticket: Ticket) -> str:
    """Full ticket as embeddable text.

    The header line is deliberately included: 'product' and 'category' are
    strong lexical signals, and a bi-encoder that never sees them has to infer
    the topic from the conversation alone.
    """
    lines = [
        f"Ticket {ticket.ticket_id}: {ticket.subject}",
        f"Product: {ticket.product} | Category: {ticket.category} | "
        f"Plan: {ticket.customer_tier} | Status: {ticket.status}",
        "",
    ]
    for turn in ticket.conversation:
        lines.append(f"{turn.role.capitalize()}: {turn.text}")
    if ticket.resolution:
        lines.extend(["", f"Resolution: {ticket.resolution}"])
    return "\n".join(lines)


def render_corpus(tickets: list[Ticket]) -> str:
    """Every ticket concatenated into one document.

    This is what a naive loader produces, and it is what the 'flat' chunking
    strategy consumes. Retained on purpose so the boundary-bleed failure is
    reproducible rather than theoretical.
    """
    return "\n\n---\n\n".join(render_ticket(t) for t in tickets)


def ticket_metadata(ticket: Ticket) -> dict[str, Any]:
    meta = {name: getattr(ticket, name) for name in METADATA_FIELDS}
    meta["tags"] = list(ticket.tags)
    return meta
