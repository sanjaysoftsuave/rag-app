"""Agent memory: buffer, session log, summarization, and vector recall."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import FakeEmbedder, make_config

from rag_app.config import AgentConfig, AgentMemoryConfig
from rag_app.memory import (
    ConversationBuffer,
    MemoryStore,
    SessionLog,
    Turn,
    memory_dir,
    summarize_history,
)


def cfg_with_memory(tmp_path, **memory_kw):
    cfg = make_config(tmp_path)
    return replace(cfg, agent=replace(cfg.agent, memory=AgentMemoryConfig(**memory_kw)))


# ---------------------------------------------------------------------------
# Short-term
# ---------------------------------------------------------------------------


def test_the_buffer_keeps_only_the_last_n_turns():
    buf = ConversationBuffer(limit=2)
    for i in range(5):
        buf.remember(Turn("user", f"turn {i}"))
    assert [t.text for t in buf.history()] == ["turn 3", "turn 4"]


def test_the_buffer_renders_turns_for_the_prompt():
    buf = ConversationBuffer(limit=8)
    buf.remember(Turn("user", "how long?"))
    buf.remember(Turn("agent", "60 minutes"))
    assert buf.recall("") == "user: how long?\nagent: 60 minutes"


# ---------------------------------------------------------------------------
# Long-term
# ---------------------------------------------------------------------------


def test_long_term_memory_survives_close_and_reopen(tmp_path):
    path = tmp_path / "turns.jsonl"
    SessionLog(path).remember(Turn.now("user", "remembered across processes"))
    assert [t.text for t in SessionLog(path).history()] == ["remembered across processes"]


def test_sessions_are_isolated(tmp_path):
    path = tmp_path / "turns.jsonl"
    SessionLog(path, session="a").remember(Turn.now("user", "for a"))
    SessionLog(path, session="b").remember(Turn.now("user", "for b"))
    assert [t.text for t in SessionLog(path, session="a").history()] == ["for a"]


def test_a_truncated_final_line_costs_one_turn_not_the_log(tmp_path):
    """The argument for JSONL over one JSON document."""
    path = tmp_path / "turns.jsonl"
    log = SessionLog(path)
    log.remember(Turn.now("user", "good line"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"role": "user", "text": "trunc')
    assert [t.text for t in log.history()] == ["good line"]


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def test_a_short_history_is_not_summarized(tmp_path):
    cfg = cfg_with_memory(tmp_path, summary_trigger_chars=1000, summary_target_chars=100)
    called = {"n": 0}

    def tripwire(messages, config):
        called["n"] += 1
        return "summary"

    turns = [Turn("user", "short")]
    out = summarize_history(turns, cfg, llm_fn=tripwire)
    assert called["n"] == 0
    assert out.turns == turns


def test_summarization_replaces_the_old_turns_and_keeps_the_recent_ones(tmp_path):
    cfg = cfg_with_memory(tmp_path, summary_trigger_chars=50, summary_target_chars=20)
    turns = [Turn("user", "x" * 40), Turn("agent", "y" * 40), Turn("user", "recent")]
    out = summarize_history(turns, cfg, llm_fn=lambda m, c: "the gist", keep_recent=1)
    assert out.failed is False
    assert out.turns[0].role == "summary"
    assert out.turns[0].text == "the gist"
    assert out.turns[-1].text == "recent"


def test_summarization_goes_through_the_same_llm_seam_as_the_agent(tmp_path):
    cfg = cfg_with_memory(tmp_path, summary_trigger_chars=10, summary_target_chars=5)
    seen = []

    def llm(messages, config):
        seen.append(messages)
        return "gist"

    summarize_history(
        [Turn("user", "x" * 50), Turn("user", "y")], cfg, llm_fn=llm, keep_recent=1
    )
    assert seen[0][0]["role"] == "system"
    assert "Compress" in seen[0][0]["content"]


def test_a_failed_summarization_degrades_visibly(tmp_path):
    cfg = cfg_with_memory(tmp_path, summary_trigger_chars=10, summary_target_chars=5)

    def boom(messages, config):
        raise RuntimeError("down")

    out = summarize_history(
        [Turn("user", "x" * 50), Turn("user", "keep")], cfg, llm_fn=boom, keep_recent=1
    )
    assert out.failed is True
    assert "dropped" in out.describe()
    assert [t.text for t in out.turns] == ["keep"]


def test_an_empty_summary_is_treated_as_a_failure(tmp_path):
    cfg = cfg_with_memory(tmp_path, summary_trigger_chars=10, summary_target_chars=5)
    out = summarize_history(
        [Turn("user", "x" * 50), Turn("user", "keep")], cfg, llm_fn=lambda m, c: "  ",
        keep_recent=1,
    )
    assert out.failed is True


# ---------------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------------


def test_memory_lives_in_its_own_directory_not_the_corpus_preset(tmp_path):
    """Pins the lock decision: sharing the corpus directory would let ingest
    close memory mid-conversation, and sharing the collection would make a
    remembered guess citable as a document."""
    from rag_app.qdrant_store import qdrant_path_for_preset

    cfg = make_config(tmp_path)
    store = MemoryStore(cfg, embedder=FakeEmbedder())
    corpus = qdrant_path_for_preset(cfg.store_dir, cfg.default_preset)
    assert store.path != corpus
    assert cfg.store_dir not in store.path.parents
    store.close()


def test_memory_dir_is_derived_from_store_dir_so_tests_never_write_to_the_repo(tmp_path):
    cfg = make_config(tmp_path)
    assert tmp_path in memory_dir(cfg).parents or memory_dir(cfg).is_relative_to(tmp_path)


def test_an_explicit_memory_dir_is_honoured(tmp_path):
    cfg = cfg_with_memory(tmp_path, dir=tmp_path / "elsewhere")
    assert memory_dir(cfg) == tmp_path / "elsewhere"


def test_vector_recall_finds_a_semantically_related_earlier_turn(tmp_path):
    cfg = make_config(tmp_path)
    store = MemoryStore(cfg, embedder=FakeEmbedder())
    store.remember(Turn.now("user", "the refund window is five business days"))
    store.remember(Turn.now("user", "password reset links last 60 minutes"))
    out = store.recall("the refund window is five business days")
    store.close()
    assert "refund window" in out


def test_recall_on_an_empty_memory_is_blank_not_an_error(tmp_path):
    cfg = make_config(tmp_path)
    store = MemoryStore(cfg, embedder=FakeEmbedder())
    assert store.recall("anything") == ""
    store.close()


def test_memory_survives_a_close_and_reopen(tmp_path):
    cfg = make_config(tmp_path)
    first = MemoryStore(cfg, embedder=FakeEmbedder())
    first.remember(Turn.now("user", "durable fact about refunds"))
    first.close()

    second = MemoryStore(cfg, embedder=FakeEmbedder())
    assert [t.text for t in second.history()] == ["durable fact about refunds"]
    assert "durable fact" in second.recall("durable fact about refunds")
    second.close()


def test_the_jsonl_is_the_source_of_truth(tmp_path):
    """Qdrant is derived; the log is what survives."""
    cfg = make_config(tmp_path)
    store = MemoryStore(cfg, embedder=FakeEmbedder())
    store.remember(Turn.now("user", "a fact"))
    store.close()
    assert (memory_dir(cfg) / "turns.jsonl").exists()
