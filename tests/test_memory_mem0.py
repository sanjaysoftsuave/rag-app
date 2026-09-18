"""The mem0 arm.

Deliberately thin, and the docstring in `memory_mem0.py` says why: mem0 cannot
be exercised offline, because even its local mode needs a configured LLM and
embedder. Testing protocol shape and the import guard is what can honestly be
tested here without a network, and pretending otherwise would be worse than
admitting the gap.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
from conftest import make_config

from rag_app.memory_mem0 import EXTRA_HINT, Mem0Memory, available


def test_the_module_imports_without_the_extra_installed():
    assert callable(Mem0Memory)
    assert "pip install" in EXTRA_HINT


def test_available_reports_whether_the_extra_is_present():
    assert available() == (find_spec("mem0") is not None)


def test_it_satisfies_the_same_memory_protocol_as_the_plain_store(tmp_path):
    """Interchangeable at run_agent(memory=...), so the agent cannot tell them
    apart and the comparison is about memory rather than two agents."""
    from rag_app.memory import MemoryStore

    for name in ("remember", "recall", "history", "describe", "close"):
        assert hasattr(Mem0Memory, name)
        assert hasattr(MemoryStore, name)


def test_recall_degrades_to_empty_rather_than_raising(tmp_path):
    """Memory is an optimisation; it must never take down a question."""
    class Broken:
        def search(self, *a, **kw):
            raise RuntimeError("no backend")

    mem = Mem0Memory(make_config(tmp_path), client=Broken())
    assert mem.recall("anything") == ""


def test_remember_degrades_rather_than_raising(tmp_path):
    from rag_app.memory import Turn

    class Broken:
        def add(self, *a, **kw):
            raise RuntimeError("no backend")

    mem = Mem0Memory(make_config(tmp_path), client=Broken())
    mem.remember(Turn.now("user", "a fact"))
    assert [t.text for t in mem.history()] == ["a fact"]
