"""The agent's MCP client — discovery instead of hard-coding, and the sync
bridge that lets the synchronous ReAct loop drive an inherently async
protocol.

Same framework-boundary shape as agent_langgraph.py: `mcp` is imported lazily
inside function bodies, never at module scope, and `available()`/`_require()`
mirror that module's pattern exactly.

WHY A BACKGROUND THREAD, NOT `asyncio.run()` PER CALL
------------------------------------------------------
`run_agent`'s loop is synchronous and calls `tools.invoke(name, argument)` —
a plain function, once per step. The MCP transport and `ClientSession` are
tied to the asyncio event loop that created them; a fresh `asyncio.run()` on
every tool call would need to reconnect (and re-run the JSON-RPC handshake)
every step, which is slow and, for a stdio server, means spawning a new
subprocess per call. So `MCPConnection` runs ONE event loop in a background
thread for the whole life of the connection, and `call_tool` is a blocking
function that schedules a coroutine onto that loop and waits for the result —
verified end to end against both an in-memory server (tests) and a real
stdio subprocess (manual smoke test) before being written here.

WHY DISCOVERED TOOLS GET A CAPABILITY NO OTHER TOOL CAN EARN BY ACCIDENT
--------------------------------------------------------------------------
Week 8's `Tool.capability` grades what an in-process tool is allowed to
reach. MCP gives us no such grading at all — a remote server can call itself
anything and describe itself however it likes. So every discovered tool is
tagged READ_EXTERNAL, a capability `build_mcp_registry` grants only when the
caller passes `allow=True` (the CLI's `--mcp-allow`). Without it, the tool is
still ADVERTISED to the model — hiding it would just make the model try it
and get "unknown tool" instead of the more honest "not available", and it is
still the thing you were asked to check before you trust it.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable

from rag_app.config import READ_EXTERNAL
from rag_app.tools import Tool, ToolRegistry

EXTRA_HINT = 'The mcp package is not installed. Run: pip install -e ".[mcp]"'


def available() -> bool:
    return find_spec("mcp") is not None


def _require():
    try:
        import mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(EXTRA_HINT) from exc


def stdio_connect(command: str, args: list[str]) -> Callable[[], object]:
    """A connect factory for `MCPConnection`: spawn `command args...` and
    speak MCP over its stdin/stdout."""
    _require()

    def connect():
        from mcp.client.stdio import StdioServerParameters, stdio_client

        return stdio_client(StdioServerParameters(command=command, args=list(args)))

    return connect


def http_connect(url: str) -> Callable[[], object]:
    """A connect factory for `MCPConnection`: an already-running server over
    streamable HTTP — what makes a server reachable by someone else's agent
    rather than only by a subprocess this process spawned itself."""
    _require()

    def connect():
        from mcp.client.streamable_http import streamable_http_client

        return streamable_http_client(url)

    return connect


class MCPConnection:
    """Owns one asyncio event loop, in a background thread, for the life of
    an MCP session. See the module docstring for why a per-call
    `asyncio.run()` does not work here.

    Used as a context manager:

        with MCPConnection(stdio_connect(sys.executable, [...])) as conn:
            tools = conn.list_tools()
            conn.call_tool(tools[0].name, {"query": "..."})
    """

    def __init__(self, connect: Callable[[], object]) -> None:
        self._connect = connect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def __enter__(self) -> "MCPConnection":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                f"Could not connect to the MCP server: {self._error}"
            ) from self._error
        return self

    def __exit__(self, *exc_info) -> None:
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # surfaced to __enter__, which is on another thread
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from mcp import ClientSession

        self._stop_event = asyncio.Event()
        async with self._connect() as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop_event.wait()

    def _call(self, coro):
        if self._loop is None or self._session is None:
            raise RuntimeError("MCPConnection is not connected. Use it as a context manager.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=60)

    def list_tools(self) -> list:
        """The remote `mcp.types.Tool` objects, as the server described them."""
        return self._call(self._session.list_tools()).tools

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call one remote tool and return its result as text.

        An MCP error result becomes a normal observation string, same policy
        as every local tool in tools.py: a raise here would take down the
        agent loop and lose the trajectory."""
        result = self._call(self._session.call_tool(name, arguments))
        text = "".join(getattr(block, "text", "") for block in result.content)
        if result.is_error:
            return f"{name} (MCP) returned an error: {text or '(no message)'}"
        return text or f"{name} (MCP) returned no content."


def _build_arguments(schema: dict | None, raw: str) -> dict | None:
    """Map one Action Input string onto an MCP tool's JSON-schema arguments.

    The ReAct loop's whole action format is ONE string per step (agent.py's
    `parse_action`) — that is what lets every local tool in tools.py take a
    single argument. An MCP tool can declare any schema, so this is where
    that mismatch actually surfaces:

      * valid JSON that decodes to an object is used as-is — the escape hatch
        for a tool that genuinely needs several fields.
      * a schema with exactly one property maps the raw string onto it,
        coercing to int/float/bool when the schema says so — this is the
        common case, and it is what makes search_documents (one `query`
        string) work with no special-casing on the agent side.
      * anything else is AMBIGUOUS and returns None; the caller turns that
        into a visible observation rather than guessing.
    """
    raw = (raw or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    props = (schema or {}).get("properties") or {}
    if not props:
        return {}
    if len(props) == 1:
        (prop_name, spec), = props.items()
        return {prop_name: _coerce_scalar(raw, (spec or {}).get("type"))}
    return None


def _coerce_scalar(raw: str, json_type: str | None):
    try:
        if json_type == "integer":
            return int(raw)
        if json_type == "number":
            return float(raw)
        if json_type == "boolean":
            return raw.strip().lower() in ("true", "1", "yes")
    except ValueError:
        pass
    return raw


def _make_run(conn: MCPConnection, mcp_tool) -> Callable[[str], str]:
    def run(argument: str) -> str:
        arguments = _build_arguments(mcp_tool.input_schema, argument)
        if arguments is None:
            props = ", ".join((mcp_tool.input_schema or {}).get("properties", {}))
            return (
                f"{mcp_tool.name} (MCP) takes structured input ({props}) that a single "
                f"Action Input string cannot express unambiguously. Pass a JSON object, "
                f'e.g. Action Input: {{"...": ...}}.'
            )
        try:
            return conn.call_tool(mcp_tool.name, arguments)
        except Exception as exc:  # a raise here would lose the trajectory, same as _safe()
            return f"{mcp_tool.name} (MCP) failed: {type(exc).__name__}: {exc}"

    return run


def discover_mcp_tools(conn: MCPConnection) -> list[Tool]:
    """Adapt every tool the server advertises into a local `Tool` — this is
    the discovery step: nothing here names a tool in advance."""
    discovered = []
    for mcp_tool in conn.list_tools():
        props = (mcp_tool.input_schema or {}).get("properties") or {}
        discovered.append(
            Tool(
                name=mcp_tool.name,
                description=(mcp_tool.description or "(no description given)")
                + " [discovered over MCP]",
                input_desc=(
                    ", ".join(props) if props else "(no arguments)"
                ) + " — or a JSON object if more than one field is needed",
                usage=f"Action Input: {{}}" if not props else f"Action Input: <{next(iter(props))}>",
                run=_make_run(conn, mcp_tool),
                capability=READ_EXTERNAL,
            )
        )
    return discovered


def build_mcp_registry(
    local_tools: ToolRegistry, discovered: list[Tool], *, allow: bool
) -> ToolRegistry:
    """A registry over the local tools PLUS the MCP-discovered ones.

    Not a mutation of `local_tools`: a fresh ToolRegistry is built and every
    local tool is re-registered into it, because `restrict()`'s sharing
    contract is for DENYING tools, not for adding new ones into the same
    dict — registering into a shared `_tools` would leak the discovered tools
    back into whatever registry the caller still holds a reference to.

    `allow=False` (no `--mcp-allow`) grants every known capability EXCEPT
    READ_EXTERNAL: the discovered tools are listed in the prompt — their
    existence is honest — but every call to one is denied, and the denial is
    what makes "checked a tool before trusting it" a measurable trace instead
    of a one-off claim.
    """
    from rag_app.config import CAPABILITIES

    granted = CAPABILITIES if allow else tuple(c for c in CAPABILITIES if c != READ_EXTERNAL)
    combined = ToolRegistry(labels_fn=local_tools.labels_fn, guard=local_tools.guard, granted=granted)
    for name in local_tools.names():
        combined.register(local_tools.get(name))
    for tool in discovered:
        combined.register(tool)
    return combined
