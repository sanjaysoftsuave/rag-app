"""The tool registry and the four tools."""

from __future__ import annotations

import pytest
from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store

from rag_app.chunking import Chunk
from rag_app.tools import FINAL_ANSWER, Tool, ToolRegistry, build_registry


def tool(name="alpha", run=None) -> Tool:
    return Tool(name, "does a thing", "some input", "Action Input: x", run or (lambda _: "ok"))


def seeded(tmp_path, texts):
    embedder = FakeEmbedder()
    chunks = [Chunk(f"{s}::{i}", s, t, {}) for i, (s, t) in enumerate(texts)]
    vectors = embedder.encode_documents([c.text for c in chunks])
    return embedder, make_qdrant_store(tmp_path, chunks, vectors)


CORPUS = [
    ("handbook.pdf", "Password reset links expire after 60 minutes."),
    ("handbook.pdf", "Error ERR-4032 means the upload exceeded the size limit."),
    ("policy.md", "Refunds are issued within five business days."),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_the_registry_rejects_a_duplicate_name():
    reg = ToolRegistry()
    reg.register(tool("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(tool("alpha"))


def test_the_registry_refuses_the_reserved_final_answer_name():
    """Registering it would shadow termination; the agent could never stop."""
    with pytest.raises(ValueError, match="terminal action"):
        ToolRegistry().register(tool(FINAL_ANSWER))


@pytest.mark.parametrize("bad", ["Search", "search documents", "9lives", "search-docs", ""])
def test_the_registry_rejects_unparseable_names(bad):
    with pytest.raises(ValueError, match="cannot be parsed"):
        ToolRegistry().register(tool(bad))


def test_descriptions_reach_the_prompt():
    reg = ToolRegistry()
    reg.register(tool("alpha"))
    out = reg.describe_for_prompt()
    assert "alpha" in out
    assert "does a thing" in out
    assert "some input" in out


def test_an_unknown_tool_name_is_rejected_at_build_time(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(ValueError, match="Unknown tool"):
        build_registry(cfg, store=None, names=["teleport"])


def test_the_config_tool_names_match_the_real_builders():
    """Anti-drift: BUILTIN_TOOL_NAMES lives in config.py to avoid an import
    cycle, so nothing else keeps the two lists in step."""
    from rag_app.config import BUILTIN_TOOL_NAMES
    from rag_app.tools import BUILDERS

    assert sorted(BUILTIN_TOOL_NAMES) == sorted(BUILDERS)


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


def test_search_documents_labels_excerpts_the_way_generate_does(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path, score_threshold=0.0)
    reg = build_registry(cfg, store=store, embedder=embedder,
                         reranker=FakeReranker(0.9), names=["search_documents"])
    out = reg.get("search_documents").run("password reset")
    store.close()
    assert "[handbook.pdf]" in out or "[policy.md]" in out


def test_search_documents_needs_a_query(tmp_path):
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=None, embedder=None, reranker=None,
                         names=["search_documents"])
    assert "needs a query" in reg.get("search_documents").run("  ")


def test_a_tool_failure_becomes_an_observation_not_an_exception(tmp_path):
    """A raising tool would take down the loop and lose the whole trajectory."""
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=None, embedder=None, reranker=None,
                         names=["search_documents"])
    out = reg.get("search_documents").run("anything")   # store is None -> boom
    assert out.startswith("search_documents failed:")


# ---------------------------------------------------------------------------
# keyword_search
# ---------------------------------------------------------------------------


def test_keyword_search_finds_an_exact_code(tmp_path):
    """The reason BM25 is a separate tool: dense search treats ERR-4032 as noise."""
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["keyword_search"])
    out = reg.get("keyword_search").run("ERR-4032")
    store.close()
    assert "ERR-4032" in out


def test_keyword_search_says_so_when_nothing_matches(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["keyword_search"])
    out = reg.get("keyword_search").run("zzzznotpresent")
    store.close()
    assert "No document contains" in out


# ---------------------------------------------------------------------------
# list_sources and read_source
# ---------------------------------------------------------------------------


def test_list_sources_returns_the_labels_the_model_must_cite(tmp_path):
    """Directly attacks the [CS-1001] failure: the legal labels are lookup-able."""
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["list_sources"])
    out = reg.get("list_sources").run("")
    store.close()
    assert "[handbook.pdf]" in out
    assert "[policy.md]" in out
    assert "only labels you may cite" in out


def test_read_source_returns_one_whole_document(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["read_source"])
    out = reg.get("read_source").run("handbook.pdf")
    store.close()
    assert "60 minutes" in out
    assert "ERR-4032" in out
    assert "Refunds" not in out          # the other document is not included


def test_read_source_tolerates_a_bracketed_label(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["read_source"])
    out = reg.get("read_source").run("[handbook.pdf]")
    store.close()
    assert "60 minutes" in out


def test_read_source_points_at_list_sources_when_the_label_is_wrong(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, names=["read_source"])
    out = reg.get("read_source").run("CS-1001")
    store.close()
    assert "No document is labelled" in out
    assert "list_sources" in out


def test_read_source_truncates_visibly(tmp_path):
    """A silently clipped observation is evidence the model cannot know is partial."""
    from dataclasses import replace

    embedder, store = seeded(tmp_path, [("big.md", "x" * 5000)])
    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, max_observation_chars=100))
    reg = build_registry(cfg, store=store, names=["read_source"])
    out = reg.get("read_source").run("big.md")
    store.close()
    assert "[truncated," in out
    assert "chars omitted]" in out


def test_every_observation_is_a_string(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path, score_threshold=0.0)
    reg = build_registry(cfg, store=store, embedder=embedder, reranker=FakeReranker(0.9))
    for name in reg.names():
        assert isinstance(reg.get(name).run("handbook.pdf"), str)
    store.close()
