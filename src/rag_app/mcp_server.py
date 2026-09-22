"""This app's own MCP server — a framework boundary, same shape as
agent_langgraph.py and memory_mem0.py: the extra is imported lazily inside
function bodies, never at module scope, so this module stays importable (and
`python -m rag_app mcp-serve` can fail politely) on a machine where fastmcp
was never installed.

WHY search_documents, AND WHY IT CALLS make_search_documents INSTEAD OF
REBUILDING IT
--------------------------------------------------------------------------
`tools.make_search_documents` already carries the relocated score gate: below
`cfg.score_threshold` it returns a "nothing matched" string and contributes no
excerpt at all. Writing a second implementation here — even one that "just"
called retrieve/rerank again — would be a second place that gate has to stay
correct, and Week 8's whole `_evidence_from` story is that a divergence like
that is exactly how a vulnerability gets in. So the MCP tool is a thin
wrapper around the SAME `Tool.run`, not a new retrieval path.

read_source and list_sources are deliberately NOT exposed here yet.
search_documents is READ_INDEX — gated, no single document ever leaves this
server whole. read_source is READ_DOCUMENT, the exfiltration primitive Week 8
put an allowlist around; exposing it to an arbitrary remote caller is a
product decision this week's exercise does not make for you.
"""

from __future__ import annotations

from importlib.util import find_spec

from rag_app.config import AppConfig

EXTRA_HINT = 'fastmcp is not installed. Run: pip install -e ".[mcp]"'


def available() -> bool:
    return find_spec("fastmcp") is not None


def _require():
    try:
        import fastmcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(EXTRA_HINT) from exc


def build_server(cfg: AppConfig, *, store, embedder=None, reranker=None):
    """Build the FastMCP server. `store` is INJECTED, same reasoning as
    `tools.build_registry`: embedded Qdrant locks a directory for one process,
    so this function must never open it itself.
    """
    _require()
    from fastmcp import FastMCP

    from rag_app.embed import Embedder
    from rag_app.rerank import build_reranker
    from rag_app.tools import make_search_documents

    embedder = embedder if embedder is not None else Embedder(cfg.bi_encoder_model)
    reranker = reranker if reranker is not None else build_reranker(cfg.cross_encoder_model)

    server = FastMCP(
        name="rag-app-search",
        instructions=(
            "Search a document corpus by meaning and get back labelled excerpts, "
            "each preceded by its citation label in square brackets. An excerpt "
            "with no citation label was not retrieved from any document."
        ),
    )
    search_tool = make_search_documents(cfg, store=store, embedder=embedder, reranker=reranker)

    @server.tool(name=search_tool.name, description=search_tool.description)
    def search_documents(query: str) -> str:
        return search_tool.run(query)

    return server


def serve(
    cfg: AppConfig,
    *,
    preset: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Open the store and run the server. The only caller is `mcp-serve`."""
    _require()
    from rag_app.pipeline import open_store

    name = preset or cfg.default_preset
    store = open_store(cfg, name)
    try:
        server = build_server(cfg, store=store)
        if transport == "stdio":
            server.run(transport="stdio", show_banner=False)
        else:
            server.run(transport=transport, host=host, port=port, show_banner=False)
    finally:
        store.close()
