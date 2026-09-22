"""mcp_client.py: argument mapping (pure), the capability grant, and — where
the extra is installed — a real in-process MCP round trip.

Skipped unless `pip install -e ".[mcp]"` is present, same convention as
test_agent_langgraph.py.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.util import find_spec

import pytest
from conftest import make_config

from rag_app.config import READ_EXTERNAL
from rag_app.mcp_client import (
    EXTRA_HINT,
    MCPConnection,
    _build_arguments,
    available,
    build_mcp_registry,
    discover_mcp_tools,
)
from rag_app.tools import Tool, ToolRegistry


def test_the_module_imports_without_the_extra_installed():
    """The CLI must be able to offer --mcp-stdio and reject it politely."""
    import rag_app.mcp_client as mod

    assert callable(mod.MCPConnection)
    assert "pip install" in EXTRA_HINT


def test_available_reports_whether_the_extra_is_present():
    assert available() == (find_spec("mcp") is not None)


# ---------------------------------------------------------------------------
# _build_arguments — pure, no I/O
# ---------------------------------------------------------------------------


def test_a_json_object_is_passed_through_as_is():
    assert _build_arguments({"properties": {"a": {}, "b": {}}}, '{"a": 1, "b": "x"}') == {
        "a": 1, "b": "x",
    }


def test_a_single_string_property_gets_the_raw_text():
    schema = {"properties": {"query": {"type": "string"}}}
    assert _build_arguments(schema, "how long is a reset link valid") == {
        "query": "how long is a reset link valid"
    }


def test_a_single_integer_property_is_coerced():
    schema = {"properties": {"n": {"type": "integer"}}}
    assert _build_arguments(schema, "5") == {"n": 5}


def test_an_uncoercible_integer_degrades_to_the_raw_string():
    """Never raise out of argument mapping — a bad coercion is a bad tool
    call, not a crash."""
    schema = {"properties": {"n": {"type": "integer"}}}
    assert _build_arguments(schema, "five") == {"n": "five"}


def test_no_properties_maps_to_an_empty_call():
    assert _build_arguments({"properties": {}}, "-") == {}
    assert _build_arguments(None, "") == {}


def test_two_properties_with_no_json_is_ambiguous():
    schema = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    assert _build_arguments(schema, "just one string") is None


# ---------------------------------------------------------------------------
# discover_mcp_tools + build_mcp_registry — a fake connection, no network
# ---------------------------------------------------------------------------


class _FakeMcpTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _FakeConnection:
    """Duck-types the two MCPConnection methods discover_mcp_tools and the
    adapted Tool.run actually call."""

    def __init__(self, tools, replies):
        self._tools = tools
        self._replies = replies
        self.calls = []

    def list_tools(self):
        return self._tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._replies[name]


def test_discovered_tools_carry_the_external_capability():
    conn = _FakeConnection(
        [_FakeMcpTool("search_documents", "search", {"properties": {"query": {"type": "string"}}})],
        {"search_documents": "[doc.md]\nan excerpt"},
    )
    discovered = discover_mcp_tools(conn)
    assert len(discovered) == 1
    assert discovered[0].capability == READ_EXTERNAL
    assert discovered[0].name == "search_documents"


def test_a_discovered_tool_runs_through_to_the_connection():
    conn = _FakeConnection(
        [_FakeMcpTool("search_documents", "search", {"properties": {"query": {"type": "string"}}})],
        {"search_documents": "[doc.md]\nan excerpt"},
    )
    tool = discover_mcp_tools(conn)[0]
    assert tool.run("reset link") == "[doc.md]\nan excerpt"
    assert conn.calls == [("search_documents", {"query": "reset link"})]


def test_an_ambiguous_schema_is_a_visible_observation_not_a_crash():
    conn = _FakeConnection(
        [_FakeMcpTool("two_args", "d", {"properties": {"a": {}, "b": {}}})],
        {},
    )
    tool = discover_mcp_tools(conn)[0]
    observation = tool.run("not json")
    assert "structured input" in observation
    assert conn.calls == []  # never reached the connection


def test_a_connection_exception_becomes_an_observation():
    class _Boom(_FakeConnection):
        def call_tool(self, name, arguments):
            raise RuntimeError("server hung up")

    conn = _Boom(
        [_FakeMcpTool("t", "d", {"properties": {"q": {"type": "string"}}})], {}
    )
    tool = discover_mcp_tools(conn)[0]
    assert "t (MCP) failed" in tool.run("x")


def local_registry(tmp_path) -> ToolRegistry:
    cfg = make_config(tmp_path)
    reg = ToolRegistry(labels_fn=lambda: frozenset({"handbook.md"}))
    reg.register(Tool("search_documents", "d", "i", "u", lambda q: f"[handbook.md]\n{q}"))
    return reg


def test_without_allow_a_discovered_tool_is_visible_but_every_call_is_denied(tmp_path):
    local = local_registry(tmp_path)
    conn = _FakeConnection(
        [_FakeMcpTool("web_search", "search the web", {"properties": {"q": {"type": "string"}}})],
        {"web_search": "results"},
    )
    combined = build_mcp_registry(local, discover_mcp_tools(conn), allow=False)
    assert "web_search" in combined.names()  # advertised
    call = combined.invoke("web_search", "cats")
    assert call.denied == "capability"
    assert conn.calls == []  # the denial happened before the tool ever ran


def test_with_allow_the_discovered_tool_actually_runs(tmp_path):
    local = local_registry(tmp_path)
    conn = _FakeConnection(
        [_FakeMcpTool("web_search", "search the web", {"properties": {"q": {"type": "string"}}})],
        {"web_search": "results"},
    )
    combined = build_mcp_registry(local, discover_mcp_tools(conn), allow=True)
    call = combined.invoke("web_search", "cats")
    assert call.denied == ""
    assert call.observation == "results"


def test_the_local_tool_and_its_label_resolver_survive_the_merge(tmp_path):
    local = local_registry(tmp_path)
    combined = build_mcp_registry(local, [], allow=False)
    assert combined.citable_labels() == frozenset({"handbook.md"})
    call = combined.invoke("search_documents", "reset link")
    assert call.observation == "[handbook.md]\nreset link"


def test_registering_into_the_combined_registry_does_not_leak_back(tmp_path):
    """build_mcp_registry must not mutate the caller's registry — a shared
    _tools dict would leak the MCP tool into every OTHER run using `local`."""
    local = local_registry(tmp_path)
    conn = _FakeConnection(
        [_FakeMcpTool("web_search", "d", {"properties": {"q": {}}})], {"web_search": "r"}
    )
    build_mcp_registry(local, discover_mcp_tools(conn), allow=True)
    assert "web_search" not in local.names()


# ---------------------------------------------------------------------------
# End to end, in process, over the real mcp + fastmcp packages
# ---------------------------------------------------------------------------

pytestmark_reason = 'pip install -e ".[mcp]"'


def _inmemory_connect(server):
    """A connect() factory driving a REAL FastMCP server over in-memory
    streams — no subprocess, so this stays fast enough for the offline suite,
    while still exercising the real mcp ClientSession/JSON-RPC path rather
    than a stand-in."""

    @asynccontextmanager
    async def connect():
        from mcp.shared.memory import create_client_server_memory_streams

        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            task = asyncio.create_task(
                server._mcp_server.run(
                    server_read, server_write,
                    server._mcp_server.create_initialization_options(),
                )
            )
            try:
                yield client_streams
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return connect


@pytest.mark.skipif(find_spec("mcp") is None or find_spec("fastmcp") is None, reason=pytestmark_reason)
def test_a_real_mcp_round_trip_discovers_and_calls_a_tool():
    from fastmcp import FastMCP

    server = FastMCP(name="test-server")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    with MCPConnection(_inmemory_connect(server)) as conn:
        discovered = discover_mcp_tools(conn)
        assert [t.name for t in discovered] == ["add"]
        assert discovered[0].capability == READ_EXTERNAL
        result = discovered[0].run('{"a": 2, "b": 3}')
    assert result == "5"


@pytest.mark.skipif(find_spec("mcp") is None or find_spec("fastmcp") is None, reason=pytestmark_reason)
def test_a_real_connection_that_never_connects_raises_with_the_extra_hint_style_message():
    @asynccontextmanager
    async def connect():
        raise RuntimeError("no such server")
        yield  # pragma: no cover - unreachable, satisfies the generator protocol

    with pytest.raises(RuntimeError, match="Could not connect"):
        MCPConnection(connect).__enter__()
