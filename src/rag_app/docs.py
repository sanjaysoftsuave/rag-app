"""Load plain reference documents — .md and .txt — alongside the ticket corpus.

A ticket has real structure worth filtering and citing precisely: an id, a
product, a status. A markdown runbook or a plain-text policy note has none of
that, and forcing one into the Ticket schema would mean inventing fake
metadata just to satisfy a shape it doesn't have. This module keeps the two
kinds of source separate at load time — see chunking.chunk_doc() for the
matching chunking side — and lets them meet only where they already can: as
plain Chunk objects, competing on relevance in the same store.

Citation label for a document is its filename (`[refund-policy.md]`), the
same idea `data/docs/*.md` used before the corpus moved to structured
tickets — CITATION_RE in generate.py already accepts dots in a citation
token, so this needed no change there.
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_EXTENSIONS = (".md", ".markdown", ".txt")

# Everything CITATION_RE in generate.py will not accept inside a citation token.
_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def citation_label(filename: str) -> str:
    """Fold a filename into a token `generate.CITATION_RE` can actually match.

    CITATION_RE is `\\[([A-Za-z0-9][A-Za-z0-9._\\-]*)\\]`. A file called
    `Q3 report.md` would be labelled `[Q3 report.md]`, which contains a space,
    fails that pattern, and is therefore never recognised as a citation at
    all — `cited_sources()` sees no grounded citation, `ask()` silently falls
    back to the retrieved chunks, and a perfectly correct answer is reported
    as uncited. Nothing errors; the signal just disappears.

    Ordinary names (`refund-policy.md`, `handbook.pdf`) pass through unchanged,
    so this is invisible until a filename needs it.
    """
    safe = _LABEL_UNSAFE.sub("-", filename).strip("-")
    if not safe or not safe[0].isalnum():
        safe = "doc-" + safe.lstrip("-._")
    return safe or "doc"


def iter_doc_files(tickets_dir: Path) -> list[Path]:
    return sorted(
        p for p in tickets_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS
    )


def load_docs(tickets_dir: Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every .md/.txt file directly in tickets_dir.

    A file that's present but empty (or whitespace-only) is skipped rather
    than producing a zero-content chunk downstream.
    """
    if not tickets_dir.exists():
        return []
    docs: list[tuple[str, str]] = []
    for path in iter_doc_files(tickets_dir):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((citation_label(path.name), text))
    return docs
