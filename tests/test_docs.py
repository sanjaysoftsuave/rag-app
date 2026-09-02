from rag_app.docs import load_docs


def test_loads_md_and_txt(tmp_path):
    (tmp_path / "policy.md").write_text("# Refund policy\nRefunds in 5 days.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Plain text notes.", encoding="utf-8")
    docs = load_docs(tmp_path)
    names = {name for name, _ in docs}
    assert names == {"policy.md", "notes.txt"}


def test_loads_markdown_alias(tmp_path):
    (tmp_path / "runbook.markdown").write_text("Runbook content.", encoding="utf-8")
    docs = load_docs(tmp_path)
    assert docs[0][0] == "runbook.markdown"


def test_ignores_other_extensions(tmp_path):
    (tmp_path / "tickets.jsonl").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
    (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4 fake")
    assert load_docs(tmp_path) == []


def test_skips_empty_or_whitespace_only_files(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "blank.txt").write_text("   \n\n  ", encoding="utf-8")
    (tmp_path / "real.md").write_text("actual content", encoding="utf-8")
    docs = load_docs(tmp_path)
    assert [name for name, _ in docs] == ["real.md"]


def test_sorted_order_is_deterministic(tmp_path):
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    docs = load_docs(tmp_path)
    assert [name for name, _ in docs] == ["a.md", "b.md"]


def test_missing_directory_returns_empty_not_an_error(tmp_path):
    assert load_docs(tmp_path / "does-not-exist") == []
