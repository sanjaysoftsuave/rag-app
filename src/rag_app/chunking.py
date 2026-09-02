"""Split source text into embeddable chunks.

ONE METHOD, DELIBERATELY
------------------------
Every source this app accepts — `.md`, `.txt` and `.pdf` — is windowed **as one
file**, never concatenated with anything else. That single rule is what makes a
citation trustworthy: because there is never a second document in a window, a
chunk can never be attributed to a source that only supplied part of its text.

A PDF is not a special case: `pdfs.load_pdfs` joins its pages into one block of
text before it ever reaches this module, so it is windowed exactly like a `.md`
file and cited the same way.

An earlier version carried three strategies for a structured `.jsonl` ticket
corpus (`ticket`, `section`, and a `flat` negative control that concatenated
the whole corpus and demonstrably bled across record boundaries). That corpus
shape is gone, and with it the reason for the choice. What remains is the
strategy those three were being compared against — the one that could not
bleed by construction.

The size/overlap trade is real and still yours to make; see `chunk_text`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split text into overlapping character windows.

    `overlap` exists so a fact split across a boundary still appears whole in
    one of the two chunks. It costs storage and can put two near-identical
    chunks into the same top-K, wasting a slot.

    NOTE the units: characters, not tokens. `chunk_size` reads like tokens and
    is not — and the embedding model has its own token ceiling (256 word-pieces
    for MiniLM, roughly 1000 characters), past which the tail of a chunk is
    silently dropped at embed time. A chunk_size above that limit is not a
    trade-off, it is invisible text.
    """
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


def chunk_doc(
    filename: str,
    text: str,
    chunk_size: int,
    overlap: int,
    *,
    source_type: str = "doc",
    extra: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Window a single document, scoped to that one file.

    `source_type` records whether this came from a `.md`/`.txt` file or a PDF,
    so `--filter source_type=pdf` can narrow to one kind. `extra` is a hook for
    any further per-source metadata worth filtering on.
    """
    metadata: dict[str, Any] = {"source_type": source_type}
    if extra:
        metadata.update(extra)
    return chunk_text(
        text, source=filename, chunk_size=chunk_size, overlap=overlap, metadata=metadata
    )


def chunk_docs(
    docs: list[tuple[str, str]],
    chunk_size: int,
    overlap: int,
    *,
    source_type: str = "doc",
) -> list[Chunk]:
    """Window a list of (citation label, text) pairs, each file on its own.

    `.md`/`.txt` and `.pdf` both arrive in this shape — see `docs.load_docs`
    and `pdfs.load_pdfs` — so a PDF is not a special case here. `source_type`
    is the only thing distinguishing them, and it exists so `--filter
    source_type=pdf` can separate them again downstream.
    """
    chunks: list[Chunk] = []
    for filename, text in docs:
        chunks.extend(chunk_doc(filename, text, chunk_size, overlap, source_type=source_type))
    return chunks
