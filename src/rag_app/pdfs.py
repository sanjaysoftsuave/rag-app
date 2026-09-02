"""Load PDF files as plain documents.

A PDF goes through exactly the same path as a `.md` or `.txt` file: the whole
file becomes one block of text, windowed as a unit, cited by its filename.
Pages are an artefact of how the bytes were laid out for printing, not a
boundary in the content, and treating them as one had a specific cost — a
sentence straddling a page break was split into two chunks with `overlap`
unable to bridge it, because each page was windowed independently. Joining the
pages first removes that.

WHAT THIS GIVES UP
------------------
`[handbook.pdf]` names a document, not a location. On a long PDF that is a
weaker citation than a page number would be — you have to search the file to
check a claim. Deliberate trade, taken so PDFs behave exactly like every other
document rather than being a special case with their own citation format.

If page-level locality is ever wanted back, the extraction below already knows
which page each piece of text came from; what changed is that the pieces are
joined before windowing rather than after.

SCANNED PDFs EXTRACT TO NOTHING
-------------------------------
A PDF of page images carries no text layer, so `extract_text()` returns empty
strings and the file contributes zero chunks. Ingesting a 40-page scan and
reporting "0 chunks" with no explanation is exactly the kind of invisible
failure this codebase exists to avoid, so `load_pdfs` returns a per-file report
and `PdfLoadReport.looks_scanned` says so out loud. OCR is out of scope; the
honest answer is "this file has no text layer", not silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rag_app.docs import citation_label

PDF_EXTENSIONS = (".pdf",)

# Pages are joined with a blank line — the same separator that already marks a
# paragraph break inside a page, so the join adds no structure the text did not
# already have.
PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PdfLoadReport:
    """What happened to one PDF, so ingest can say it out loud."""

    filename: str
    n_pages: int
    n_text_pages: int
    encrypted: bool = False

    @property
    def empty_pages(self) -> int:
        return self.n_pages - self.n_text_pages

    @property
    def looks_scanned(self) -> bool:
        """No page yielded text — usually page images with no text layer."""
        return self.n_pages > 0 and self.n_text_pages == 0

    def describe(self) -> str:
        if self.encrypted:
            return f"{self.filename}: encrypted, could not be read"
        if self.looks_scanned:
            return (
                f"{self.filename}: {self.n_pages} pages, NO extractable text "
                f"(scanned images? this file contributes nothing)"
            )
        empty = f", {self.empty_pages} blank" if self.empty_pages else ""
        return f"{self.filename}: {self.n_text_pages}/{self.n_pages} pages with text{empty}"


# Join a word hyphenated across a line break: "refund-\ning" -> "refunding".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_HORIZONTAL_WS = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def clean_page_text(text: str) -> str:
    """Repair the two artefacts PDF extraction reliably introduces.

    Deliberately conservative. Hyphenated line breaks and ragged spacing are
    unambiguous extraction damage, so they are repaired. Single newlines are
    NOT collapsed into spaces: in a PDF they are just as likely to separate
    table cells or list items as to continue a sentence, and guessing wrong
    welds unrelated facts into one line — which is worse for retrieval than
    the ragged text it replaces.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = "\n".join(_HORIZONTAL_WS.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def iter_pdf_files(tickets_dir: Path) -> list[Path]:
    if not tickets_dir.exists():
        return []
    return sorted(
        p for p in tickets_dir.iterdir()
        if p.is_file() and p.suffix.lower() in PDF_EXTENSIONS
    )


def _open_reader(path: Path):
    """Import pypdf lazily, and fail with an instruction rather than a traceback.

    Lazy for the same reason `Embedder` imports sentence_transformers inside
    __init__: importing `rag_app.ingest` must not drag in a dependency that a
    corpus of .md/.txt files never needs.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PDF support needs pypdf, which is not installed. Run:\n"
            '    pip install -e ".[pdf]"      (or: pip install pypdf)\n'
            f"Otherwise remove {path.name} from the corpus directory."
        ) from exc
    return PdfReader(str(path))


def read_pdf(path: Path, *, reader_factory=None) -> tuple[str, PdfLoadReport]:
    """Extract one PDF as a single block of text. A bad page is skipped, never fatal.

    `reader_factory` is the test seam, for the same reason `ask()` takes an
    `embedder`: it keeps this path exercisable without shipping binary PDF
    fixtures or depending on pypdf being installed in the test environment.
    Resolved at call time rather than as a default argument value, so patching
    `pdfs._open_reader` actually takes effect.
    """
    reader = (reader_factory or _open_reader)(path)

    if getattr(reader, "is_encrypted", False):
        # An empty user password is the common "protected but not secret" case
        # and is worth one attempt; a real password is not something to guess.
        try:
            opened = reader.decrypt("")
        except Exception:
            opened = 0
        if not opened:
            return "", PdfLoadReport(path.name, 0, 0, encrypted=True)

    pieces: list[str] = []
    total = 0
    for index, page in enumerate(reader.pages, start=1):
        total = index
        try:
            raw = page.extract_text() or ""
        except Exception:
            # One malformed page must not lose the other thirty-nine.
            raw = ""
        text = clean_page_text(raw)
        if text:
            pieces.append(text)
    return PAGE_SEPARATOR.join(pieces), PdfLoadReport(path.name, total, len(pieces))


def load_pdfs(
    tickets_dir: Path, *, reader_factory=None
) -> tuple[list[tuple[str, str]], list[PdfLoadReport]]:
    """Return (citation label, text) for every PDF, plus per-file reports.

    The same shape `docs.load_docs` returns, so `ingest` can window both with
    the same call and PDFs stop being a special case downstream.
    """
    docs: list[tuple[str, str]] = []
    reports: list[PdfLoadReport] = []
    for path in iter_pdf_files(tickets_dir):
        text, report = read_pdf(path, reader_factory=reader_factory)
        if text:
            docs.append((citation_label(path.name), text))
        reports.append(report)
    return docs, reports
