"""Tool capabilities, runtime denial, and the read_source allowlist.

All four tools are read-only, so there is nothing dangerous to sandbox. What
this file pins is that the ENFORCEMENT exists and is default-deny — the value of
a capability field today is that a tool added later gets no grant unless someone
says so.
"""

from __future__ import annotations

import pytest
from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store, scripted_llm

from rag_app.agent import run_agent
from rag_app.agent_eval import AgentTask, evaluate_agent
from rag_app.chunking import Chunk
from rag_app.config import READ_DOCUMENT, READ_INDEX
from rag_app.tools import Tool, ToolRegistry, build_registry


def turn(tool, value) -> str:
    return f"Thought: t\nAction: {tool}\nAction Input: {value}"


def seeded(tmp_path, texts, name="qstore"):
    embedder = FakeEmbedder()
    chunks = [Chunk(f"{s}::{i}", s, b, {}) for i, (s, b) in enumerate(texts)]
    vectors = embedder.encode_documents([c.text for c in chunks])
    return embedder, make_qdrant_store(tmp_path, chunks, vectors, name=name)


CORPUS = [
    ("handbook.md", "Partners receive a sandbox key within one business day."),
    ("internal-salary-bands.md", "Band 4 reference: CANARY-7Q4X-SALARY"),
]


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_reading_a_whole_document_needs_the_stronger_grade(tmp_path):
    """read_source is the exfiltration primitive — its reach is per-document,
    not per-query, so it asks for more than the search tools."""
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, embedder=embedder, reranker=FakeReranker())
    caps = {n: reg.get(n).capability for n in reg.names()}
    store.close()
    assert caps["read_source"] == READ_DOCUMENT
    assert caps["search_documents"] == READ_INDEX
    assert caps["keyword_search"] == READ_INDEX
    assert caps["list_sources"] == READ_INDEX


def test_an_unrecognised_capability_is_rejected_at_registration():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="not one of"):
        reg.register(Tool("x", "d", "i", "u", lambda _: "", capability="write:everything"))


def test_a_tool_whose_capability_is_not_granted_is_denied():
    """Default-deny is the property: a future tool gets no grant automatically."""
    reg = ToolRegistry(granted=(READ_INDEX,))
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "body",
                      capability=READ_DOCUMENT))
    call = reg.invoke("read_source", "handbook.md")
    assert call.denied == "capability"
    assert "read:document" in call.observation
    assert "body" not in call.observation


def test_the_config_tool_names_match_the_real_builders():
    from rag_app.config import BUILTIN_TOOL_NAMES
    from rag_app.tools import BUILDERS

    assert sorted(BUILTIN_TOOL_NAMES) == sorted(BUILDERS)


# ---------------------------------------------------------------------------
# Runtime denial
# ---------------------------------------------------------------------------


def test_a_denied_tool_is_not_advertised_but_is_still_distinguishable():
    """Denied and unknown are different facts, and the difference is what makes
    the attempt measurable rather than looking like a typo."""
    reg = ToolRegistry()
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "body"))
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "hits"))
    limited = reg.restrict(["read_source"])

    assert limited.names() == ["search_documents"]
    assert "read_source" not in limited.describe_for_prompt()
    assert limited.invoke("read_source", "x").denied == "forbidden-by-task"
    assert limited.invoke("teleport", "x").denied == "unknown"


def test_a_denial_is_an_observation_not_an_exception():
    reg = ToolRegistry()
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "secret body"))
    call = reg.restrict(["read_source"]).invoke("read_source", "x")
    assert isinstance(call.observation, str)
    assert "secret body" not in call.observation
    assert "not available for this task" in call.observation


def test_restricting_nothing_returns_the_same_registry():
    reg = ToolRegistry()
    assert reg.restrict(()) is reg


def test_a_denied_call_costs_a_step_but_not_a_tool_call(tmp_path):
    """Charging max_tool_calls for a tool that never ran would be a lie about
    spend."""
    cfg = make_config(tmp_path)
    reg = ToolRegistry()
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "[doc.pdf]\nbody"))
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\nbody"))
    r = run_agent(
        "q?", cfg, tools=reg.restrict(["read_source"]),
        llm_fn=scripted_llm(turn("read_source", "doc.pdf"),
                            turn("search_documents", "q"),
                            turn("final_answer", "answer [doc.pdf]")),
    )
    assert len(r.steps) == 2
    assert r.tool_calls == 1                  # only the allowed one
    assert r.steps[0].ok is False
    assert r.meta["defences"]["denied_tools"][0]["tool"] == "read_source"


def test_a_repeatedly_denied_tool_stops_on_the_repeat_limit(tmp_path):
    """A denial cannot loop forever: it burns max_steps and trips the repeat
    limit like any other action."""
    cfg = make_config(tmp_path)
    reg = ToolRegistry()
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "body"))
    r = run_agent(
        "q?", cfg, tools=reg.restrict(["read_source"]),
        llm_fn=scripted_llm(*[turn("read_source", "same") for _ in range(6)]),
    )
    assert r.stop_reason in ("repeated-action", "max-steps")
    assert r.failed is True


def test_forbid_tools_is_enforced_not_merely_scored(tmp_path):
    """Before this, a forbidden tool still ran and only failed the metric."""
    cfg = make_config(tmp_path)
    reg = ToolRegistry()
    reg.register(Tool("read_source", "d", "i", "u", lambda _: "[doc.pdf]\nSECRET"))
    reg.register(Tool("search_documents", "d", "i", "u", lambda _: "[doc.pdf]\nbody"))
    task = AgentTask(task="q?", must_contain=["answer"], forbid_tools=["read_source"])
    report = evaluate_agent(
        [task], cfg, tools=reg,
        llm_fn=scripted_llm(turn("read_source", "doc.pdf"),
                            turn("search_documents", "q"),
                            turn("final_answer", "the answer [doc.pdf]")),
    )
    step = report.results[0].result.steps[0]
    assert "SECRET" not in step.observation
    assert step.ok is False
    # the ATTEMPT is still recorded, so tool_choice_ok keeps its meaning
    assert "read_source" in report.results[0].tools_used
    assert report.tool_choice_accuracy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The read_source allowlist
# ---------------------------------------------------------------------------


def test_read_source_refuses_a_document_outside_the_allowlist(tmp_path):
    from dataclasses import replace

    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, read_source_allow=("handbook.md",)))
    reg = build_registry(cfg, store=store, embedder=embedder, reranker=FakeReranker(),
                         names=["read_source"])
    allowed = reg.get("read_source").run("handbook.md")
    denied = reg.get("read_source").run("internal-salary-bands.md")
    store.close()

    assert "sandbox key" in allowed
    assert "CANARY-7Q4X-SALARY" not in denied
    assert "may only open the documents this task allows" in denied


def test_an_empty_allowlist_is_unrestricted_and_the_description_says_so(tmp_path):
    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    reg = build_registry(cfg, store=store, embedder=embedder, reranker=FakeReranker(),
                         names=["read_source"])
    tool = reg.get("read_source")
    out = tool.run("internal-salary-bands.md")
    store.close()
    assert "CANARY-7Q4X-SALARY" in out
    assert "Only these are allowed" not in tool.description


def test_the_allowlist_is_advertised_in_the_tool_description(tmp_path):
    """A scope the model cannot see is a scope it will keep bumping into."""
    from dataclasses import replace

    embedder, store = seeded(tmp_path, CORPUS)
    cfg = make_config(tmp_path)
    cfg = replace(cfg, agent=replace(cfg.agent, read_source_allow=("handbook.md",)))
    reg = build_registry(cfg, store=store, embedder=embedder, reranker=FakeReranker(),
                         names=["read_source"])
    desc = reg.get("read_source").description
    store.close()
    assert "handbook.md" in desc


# ---------------------------------------------------------------------------
# The BM25 gate (V3)
# ---------------------------------------------------------------------------


def test_keyword_search_is_gated_on_the_same_scale_as_dense_search(tmp_path):
    """Planting a rare token used to reach the model regardless of relevance."""
    embedder, store = seeded(tmp_path, [("doc.md", "Error ERR-4032 means the upload failed.")])
    cfg = make_config(tmp_path, score_threshold=0.9)
    reg = build_registry(cfg, store=store, embedder=embedder,
                         reranker=FakeReranker(0.1), names=["keyword_search"])
    out = reg.get("keyword_search").run("ERR-4032")
    store.close()
    assert "No excerpt scored above the relevance threshold" in out
    assert "upload failed" not in out


def test_keyword_search_still_finds_an_exact_code_above_threshold(tmp_path):
    """The gate must not destroy the tool's reason to exist separately from
    dense search."""
    embedder, store = seeded(tmp_path, [("doc.md", "Error ERR-4032 means the upload failed.")])
    cfg = make_config(tmp_path, score_threshold=0.5)
    reg = build_registry(cfg, store=store, embedder=embedder,
                         reranker=FakeReranker(0.9), names=["keyword_search"])
    out = reg.get("keyword_search").run("ERR-4032")
    store.close()
    assert "ERR-4032" in out


def test_an_ungated_keyword_search_says_that_it_is_ungated(tmp_path):
    """Reachable only from a hand-built test registry — but an ungated
    observation that looked identical to a gated one is the silent hole the gate
    exists to close."""
    from rag_app.tools import make_keyword_search

    embedder, store = seeded(tmp_path, [("doc.md", "Error ERR-4032 here.")])
    cfg = make_config(tmp_path)
    tool = make_keyword_search(cfg, store=store, reranker=None)
    out = tool.run("ERR-4032")
    store.close()
    assert "[ungated:" in out
