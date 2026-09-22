"""mcp_server.py: the module stays importable without the extra, and — where
fastmcp is installed — the server exposes search_documents carrying the SAME
gate tools.make_search_documents already enforces, over a real in-process MCP
round trip.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.util import find_spec

import pytest
from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store

from rag_app.chunking import Chunk
from rag_app.mcp_server import EXTRA_HINT, available

pytestmark_reason = 'pip install -e ".[mcp]"'


def test_the_module_imports_without_the_extra_installed():
    import rag_app.mcp_server as mod

    assert callable(mod.build_server)
    assert "pip install" in EXTRA_HINT


def test_available_reports_whether_the_extra_is_present():
    assert available() == (find_spec("fastmcp") is not None)


def _round_trip(server):
    """Run one list_tools + call_tool round trip against `server` over
    in-memory streams and return (tool_names, call_tool_result)."""

    async def go():
        from mcp import ClientSession
        from mcp.shared.memory import create_client_server_memory_streams

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            client_read, client_write = client_streams
            task = asyncio.create_task(
                server._mcp_server.run(
                    server_read, server_write,
                    server._mcp_server.create_initialization_options(),
                )
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    await session.initialize()
                    names = [t.name for t in (await session.list_tools()).tools]
                    result = await session.call_tool(
                        "search_documents", {"query": "reset link"}
                    )
                    text = "".join(getattr(b, "text", "") for b in result.content)
                    return names, text
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return asyncio.run(go())


@pytest.mark.skipif(find_spec("mcp") is None or find_spec("fastmcp") is None, reason=pytestmark_reason)
def test_the_server_advertises_search_documents_and_answers_above_threshold(tmp_path):
    from rag_app.mcp_server import build_server

    embedder = FakeEmbedder()
    chunks = [Chunk("h.md::0", "handbook.md", "Password reset links last 60 minutes.", {})]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = make_qdrant_store(tmp_path, chunks, vectors)
    cfg = make_config(tmp_path, score_threshold=0.0)

    server = build_server(cfg, store=store, embedder=embedder, reranker=FakeReranker(0.9))
    try:
        names, text = _round_trip(server)
    finally:
        store.close()

    assert names == ["search_documents"]
    assert text.startswith("[handbook.md]")
    assert "60 minutes" in text


@pytest.mark.skipif(find_spec("mcp") is None or find_spec("fastmcp") is None, reason=pytestmark_reason)
def test_below_threshold_the_gate_still_fires_over_mcp(tmp_path):
    """The point of wrapping make_search_documents rather than rewriting
    retrieval: the score gate is not a second implementation to keep in sync."""
    from rag_app.mcp_server import build_server

    embedder = FakeEmbedder()
    chunks = [Chunk("h.md::0", "handbook.md", "Password reset links last 60 minutes.", {})]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = make_qdrant_store(tmp_path, chunks, vectors)
    cfg = make_config(tmp_path, score_threshold=0.99)

    server = build_server(cfg, store=store, embedder=embedder, reranker=FakeReranker(0.1))
    try:
        _, text = _round_trip(server)
    finally:
        store.close()

    assert "No excerpt scored above the relevance threshold" in text
