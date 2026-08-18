"""REPL command handling.

`Session` holds all the state so commands are testable without stdin; only
`run()` touches the terminal, and that stays untested by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter
from rag_app.repl import Quit, Session
from rag_app.store import StoreMeta, VectorStore, store_path_for_preset

from conftest import FakeReranker, FixedEmbedder, make_config


def _seed(cfg, preset):
    chunks = [
        Chunk(f"{preset}::0", "TIC-001", "Free plan is 60 rpm",
              {"ticket_id": "TIC-001", "product": "API", "customer_tier": "free"})
    ]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    VectorStore(chunks, vectors, StoreMeta("fake", 2, "ticket", 2000, 200, 1)).save(
        store_path_for_preset(cfg.store_dir, preset)
    )


@pytest.fixture
def session(tmp_path):
    cfg = make_config(tmp_path)
    _seed(cfg, "A")
    _seed(cfg, "C")
    return Session(
        cfg=cfg,
        embedder=FixedEmbedder(),
        reranker=FakeReranker(0.9),
        preset="C",
        generate_fn=lambda q, c, k: "Free plan is 60 rpm [TIC-001].",
    )


def test_exit_raises_quit(session):
    for cmd in ("/exit", "/quit", "/q"):
        with pytest.raises(Quit):
            session.command(cmd)


def test_help_and_config_render(session):
    assert "/filter" in session.command("/help")
    out = session.command("/config")
    assert "preset" in out and "threshold" in out


def test_filter_set_show_and_clear(session):
    assert "Filters ->" in session.command("/filter product=API customer_tier=free")
    assert session.flt.must == {"product": "API", "customer_tier": "free"}
    assert "product=API" in session.command("/filter")
    session.command("/nofilter")
    assert session.flt == MetaFilter()


def test_bad_filter_reports_without_crashing(session):
    out = session.command("/filter productAPI")
    assert "Bad filter" in out
    assert session.flt == MetaFilter()


def test_preset_switch_and_rejection(session):
    assert "Preset -> A" in session.command("/preset A")
    assert session.preset == "A"
    assert "Unknown preset" in session.command("/preset ZZZ")
    assert session.preset == "A", "a rejected preset must not change state"


def test_verbose_toggle(session):
    session.command("/verbose")
    assert session.verbose is True
    session.command("/quiet")
    assert session.verbose is False


def test_unknown_command_is_reported(session):
    assert "Unknown command" in session.command("/nope")


def test_last_before_asking(session):
    assert "Nothing asked yet" in session.command("/last")


def test_stores_are_opened_once_per_preset(session):
    first = session.store()
    assert session.store() is first, "store must be cached, not reopened per question"
    session.command("/preset A")
    assert session.store() is not first
    session.command("/preset C")
    assert session.store() is first, "switching back must reuse the cached store"


def test_answer_records_last(session):
    answer = session.answer("What is the Free plan rate limit?")
    assert answer.gate == "answered"
    assert answer.sources == ["TIC-001"]
    assert session.last is answer
    assert "TIC-001" in session.command("/last")


def test_gate_refusal_inside_a_session(tmp_path):
    """A refused question must not end the session or call the generator."""
    cfg = make_config(tmp_path, score_threshold=0.9)
    _seed(cfg, "C")
    called = {"llm": False}

    def gen(q, c, k):
        called["llm"] = True
        return "should not run"

    s = Session(cfg=cfg, embedder=FixedEmbedder(), reranker=FakeReranker(0.1),
                preset="C", generate_fn=gen)
    answer = s.answer("something not in the corpus")
    assert answer.used_llm is False
    assert called["llm"] is False
    assert s.last is answer


def test_show_prints_a_ticket(tmp_path):
    import json

    cfg = make_config(tmp_path)
    _seed(cfg, "C")
    cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
    (cfg.tickets_dir / "t.jsonl").write_text(
        json.dumps({
            "ticket_id": "TIC-001", "subject": "Rate limit", "product": "API",
            "category": "rate-limit", "status": "resolved", "priority": "high",
            "conversation": [{"role": "agent", "text": "60 requests per minute."}],
        }),
        encoding="utf-8",
    )
    s = Session(cfg=cfg, embedder=FixedEmbedder(), reranker=FakeReranker(0.9), preset="C")
    assert "60 requests per minute" in s.command("/show TIC-001")
    assert "No ticket" in s.command("/show TIC-999")
    assert "Usage:" in s.command("/show")
