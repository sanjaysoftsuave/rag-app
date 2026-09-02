"""PDF loading — page labels, text repair, and the failure modes that are silent.

No binary PDF fixtures and no pypdf dependency: `read_pdf` takes a
`reader_factory`, the same dependency-injection seam `ask()` uses, so these
tests exercise the real extraction logic against a stand-in reader.

A PDF is loaded as ONE document — pages joined, cited by filename — exactly
like a `.md` file.
"""

from __future__ import annotations

import pytest

from rag_app.docs import citation_label
from rag_app.generate import CITATION_RE
from rag_app.pdfs import PdfLoadReport, clean_page_text, load_pdfs, read_pdf


class FakePage:
    def __init__(self, text: str | None, boom: bool = False):
        self._text = text
        self._boom = boom

    def extract_text(self):
        if self._boom:
            raise ValueError("malformed content stream")
        return self._text


class FakeReader:
    def __init__(self, pages, is_encrypted: bool = False, decrypts: bool = False):
        self.pages = pages
        self.is_encrypted = is_encrypted
        self._decrypts = decrypts

    def decrypt(self, password):
        return 1 if self._decrypts else 0


def reader_for(*pages, **kwargs):
    return lambda path: FakeReader(list(pages), **kwargs)


# --- the label format is the load-bearing part -----------------------------


def test_a_filename_survives_the_citation_regex():
    """If this fails, every correct PDF answer is reported as a hallucination.

    `cited_sources()` classifies any citation it cannot match as invented, so a
    label the regex rejects does not degrade gracefully — it inverts the
    grounding report.
    """
    label = citation_label("handbook.pdf")
    assert label == "handbook.pdf"
    assert CITATION_RE.findall(f"Refunds take 5 days [{label}].") == [label]


def test_filenames_with_spaces_are_folded_into_a_matchable_label():
    """An uploaded 'Q3 report.pdf' must not produce an unciteable label."""
    label = citation_label("Q3 report.pdf")
    assert " " not in label
    assert CITATION_RE.findall(f"[{label}]") == [label]


def test_citation_label_leaves_ordinary_names_alone():
    assert citation_label("refund-policy.md") == "refund-policy.md"
    assert citation_label("notes_2026.txt") == "notes_2026.txt"


def test_citation_label_fixes_a_leading_non_alphanumeric():
    """CITATION_RE requires the first character to be alphanumeric."""
    label = citation_label(".hidden.md")
    assert label[0].isalnum()
    assert CITATION_RE.findall(f"[{label}]") == [label]


# --- text repair -----------------------------------------------------------


def test_hyphenated_line_breaks_are_rejoined():
    assert clean_page_text("the refund-\ning window") == "the refunding window"


def test_single_newlines_are_preserved():
    """Collapsing them would weld table rows and list items into one line."""
    assert clean_page_text("Free: 60\nPro: 600") == "Free: 60\nPro: 600"


def test_ragged_spacing_and_blank_runs_are_normalized():
    assert clean_page_text("a    b\t\tc") == "a b c"
    assert clean_page_text("one\n\n\n\n\ntwo") == "one\n\ntwo"


def test_clean_page_text_handles_empty_input():
    assert clean_page_text("") == ""
    assert clean_page_text("   \n \n ") == ""


# --- extraction behaviour --------------------------------------------------


def test_pages_are_joined_into_one_document(tmp_path):
    text, report = read_pdf(
        tmp_path / "doc.pdf",
        reader_factory=reader_for(FakePage("first page"), FakePage("second page")),
    )
    assert text == "first page\n\nsecond page"
    assert report.n_pages == 2 and report.n_text_pages == 2


def test_a_fact_spanning_a_page_break_is_no_longer_split(tmp_path):
    """The reason pages are joined before windowing rather than after."""
    text, _ = read_pdf(
        tmp_path / "doc.pdf",
        reader_factory=reader_for(FakePage("Refunds take"), FakePage("five business days.")),
    )
    assert "Refunds take" in text and "five business days." in text


def test_blank_pages_are_skipped(tmp_path):
    text, report = read_pdf(
        tmp_path / "doc.pdf",
        reader_factory=reader_for(FakePage("one"), FakePage("   "), FakePage("three")),
    )
    assert text == "one\n\nthree"
    assert report.n_pages == 3
    assert report.n_text_pages == 2
    assert report.empty_pages == 1


def test_one_malformed_page_does_not_lose_the_others(tmp_path):
    text, report = read_pdf(
        tmp_path / "doc.pdf",
        reader_factory=reader_for(FakePage("fine"), FakePage(None, boom=True), FakePage("also fine")),
    )
    assert text == "fine\n\nalso fine"
    assert report.n_pages == 3


def test_a_pdf_with_no_text_layer_is_reported_not_silently_empty(tmp_path):
    text, report = read_pdf(
        tmp_path / "scan.pdf",
        reader_factory=reader_for(FakePage(""), FakePage(None)),
    )
    assert text == ""
    assert report.looks_scanned
    assert "NO extractable text" in report.describe()


def test_encrypted_pdf_is_reported_rather_than_raising(tmp_path):
    text, report = read_pdf(
        tmp_path / "locked.pdf",
        reader_factory=reader_for(FakePage("secret"), is_encrypted=True),
    )
    assert text == ""
    assert report.encrypted
    assert "encrypted" in report.describe()


def test_encrypted_with_an_empty_password_still_reads(tmp_path):
    """'Protected but not secret' is common enough to be worth one attempt."""
    text, _ = read_pdf(
        tmp_path / "locked.pdf",
        reader_factory=reader_for(FakePage("readable"), is_encrypted=True, decrypts=True),
    )
    assert text == "readable"


# --- directory scanning ----------------------------------------------------


def test_load_pdfs_is_deterministic_and_ignores_other_extensions(tmp_path):
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.md").write_text("not a pdf", encoding="utf-8")
    (tmp_path / "tickets.jsonl").write_text("{}", encoding="utf-8")

    docs, reports = load_pdfs(tmp_path, reader_factory=reader_for(FakePage("x")))
    assert [r.filename for r in reports] == ["a.pdf", "b.pdf"]
    assert [label for label, _ in docs] == ["a.pdf", "b.pdf"]


def test_missing_directory_returns_empty_not_an_error(tmp_path):
    assert load_pdfs(tmp_path / "nope") == ([], [])


def test_report_describe_counts_blank_pages():
    report = PdfLoadReport("doc.pdf", n_pages=10, n_text_pages=7)
    assert not report.looks_scanned
    assert "7/10" in report.describe()
    assert "3 blank" in report.describe()
