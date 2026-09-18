"""The tools an agent can call, and the registry that describes them to it.

WHY THE SEARCH TOOL DOES NOT WRAP ask()
----------------------------------------
Wrapping `pipeline.ask()` looks like the safe choice — it already has the
grounding gate — and it is not. The agent still writes its own final prose over
whatever the sub-answers said, and THAT text is a second, ungated generation.
The guarantees would protect the inner calls and leave the actual output
unprotected. It also removes the only reason to run an agent over this corpus at
all: synthesising across several documents.

So `search_documents` wraps retrieve -> rerank -> top-N and returns labelled
excerpts, and the gate moves INTO the tool. Below threshold, the tool returns a
"nothing matched" string and contributes zero evidence, so the model physically
cannot see low-scoring text. That is stricter than `ask()`, which hands the
excerpts to the model and then refuses on the score.

EVERY OBSERVATION IS A STRING, AND EVERY FAILURE IS AN OBSERVATION
-------------------------------------------------------------------
A tool that raises takes down the loop and loses the trajectory. A tool that
returns its error as text gives the model a chance to recover and leaves the
failure visible in the trace. Same policy as `transform_query`: degrade, stay
visible, never raise.

Truncation is always announced. A silently clipped observation is a model
reasoning over evidence it does not know is incomplete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from rag_app.config import AppConfig
from rag_app.rerank import rerank_all
from rag_app.retrieve import retrieve

Observation = str

# The loop's terminal action. It is not a tool: registering one under this name
# would shadow termination and the agent could never stop.
FINAL_ANSWER = "final_answer"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str      # one line; goes verbatim into the prompt
    input_desc: str       # what Action Input should contain
    usage: str            # one worked example line
    run: Callable[[str], Observation]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name == FINAL_ANSWER:
            raise ValueError(
                f"{FINAL_ANSWER!r} is the loop's terminal action, not a tool. Registering "
                f"it would shadow termination and the agent could never finish."
            )
        if not _NAME_RE.match(tool.name):
            raise ValueError(
                f"Tool name {tool.name!r} must be lowercase letters, digits and "
                f"underscores starting with a letter — anything else cannot be parsed "
                f"back out of an 'Action:' line."
            )
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. Two tools with one "
                f"name means the model's choice is decided by dict ordering."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def describe_for_prompt(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"  {tool.name}")
            lines.append(f"      {tool.description}")
            lines.append(f"      Action Input: {tool.input_desc}")
            lines.append(f"      e.g. {tool.usage}")
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    """Clip loudly. A silent clip is evidence the model cannot know is partial."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


def _blocks(items) -> str:
    """Excerpts labelled exactly as generate.build_prompt labels them."""
    return "\n\n".join(f"[{c.chunk.source}]\n{c.chunk.text}" for c in items)


def _safe(name: str, fn):
    """Wrap a tool body so any exception becomes an observation, not a crash."""

    def runner(argument: str) -> Observation:
        try:
            return fn(argument)
        except Exception as exc:
            return f"{name} failed: {type(exc).__name__}: {exc}"

    return runner


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def make_search_documents(cfg: AppConfig, *, store, embedder, reranker) -> Tool:
    """Dense retrieval, carrying the relocated score gate."""

    def run(query: str) -> Observation:
        query = (query or "").strip()
        if not query:
            return "search_documents needs a query string."
        vector = embedder.encode_queries([query])[0]
        retrieved = retrieve(store, vector, k=cfg.retrieve_k)
        if not retrieved:
            return "Nothing was retrieved for that query."
        ranked = rerank_all(query, retrieved, scorer=reranker, scale=cfg.rerank_score_scale)
        top = ranked[: cfg.rerank_n]
        # The gate, moved into the tool. Below threshold the model receives no
        # excerpt at all, so it cannot reason over text the gate rejected.
        if not top or top[0].score < cfg.score_threshold:
            best = top[0].score if top else 0.0
            return (
                f"No excerpt scored above the relevance threshold "
                f"(best {best:.4f} < {cfg.score_threshold}). Nothing in the documents "
                f"matches this query."
            )
        return _truncate(_blocks(top), cfg.agent.max_observation_chars)

    return Tool(
        name="search_documents",
        description="Search the documents by meaning. Returns labelled excerpts.",
        input_desc="a search query in plain words",
        usage="Action Input: how long is a password reset link valid",
        run=_safe("search_documents", run),
    )


def make_keyword_search(cfg: AppConfig, *, store, bm25=None) -> Tool:
    """Exact-token search. Finds error codes dense retrieval treats as noise."""

    def run(query: str) -> Observation:
        query = (query or "").strip()
        if not query:
            return "keyword_search needs a term to look for."
        from rag_app.bm25 import BM25Index

        index = bm25 or BM25Index.from_store(store)
        hits = index.search(query, k=cfg.rerank_n)
        if not hits:
            return f"No document contains {query!r}."
        return _truncate(_blocks(hits), cfg.agent.max_observation_chars)

    return Tool(
        name="keyword_search",
        description=(
            "Search for an exact word, code or identifier. Use this for error codes, "
            "ticket numbers and proper nouns, which meaning-based search misses."
        ),
        input_desc="the exact token to find",
        usage="Action Input: CS-1005",
        run=_safe("keyword_search", run),
    )


def make_list_sources(cfg: AppConfig, *, store) -> Tool:
    """The labels the model is allowed to cite.

    CLAUDE.md records the #1 citation failure on this corpus: a PDF containing
    "Ticket CS-1001" produced the citation [CS-1001], which names an identifier
    in the body text rather than the document. Letting the agent LOOK UP the
    legal labels attacks that failure directly.
    """

    def run(_: str) -> Observation:
        seen: list[str] = []
        for chunk in store.all_chunks():
            if chunk.source not in seen:
                seen.append(chunk.source)
        if not seen:
            return "The index is empty."
        return (
            "These are the only labels you may cite:\n"
            + "\n".join(f"  [{s}]" for s in seen)
        )

    return Tool(
        name="list_sources",
        description="List every document label in the index. These are the only legal citations.",
        input_desc="(nothing)",
        usage="Action Input: -",
        run=_safe("list_sources", run),
    )


def make_read_source(cfg: AppConfig, *, store) -> Tool:
    """Read one document end to end — the multi-hop enabler."""

    def run(label: str) -> Observation:
        label = (label or "").strip().strip("[]")
        if not label:
            return "read_source needs a document label. Use list_sources to see them."
        chunks = [c for c in store.all_chunks() if c.source == label]
        if not chunks:
            return (
                f"No document is labelled {label!r}. Use list_sources to see the legal labels."
            )
        body = "\n\n".join(c.text for c in chunks)
        return _truncate(f"[{label}]\n{body}", cfg.agent.max_observation_chars)

    return Tool(
        name="read_source",
        description="Read one whole document, given its label.",
        input_desc="a document label exactly as list_sources printed it",
        usage="Action Input: handbook.pdf",
        run=_safe("read_source", run),
    )


BUILDERS = {
    "search_documents": lambda cfg, ctx: make_search_documents(
        cfg, store=ctx["store"], embedder=ctx["embedder"], reranker=ctx["reranker"]
    ),
    "keyword_search": lambda cfg, ctx: make_keyword_search(
        cfg, store=ctx["store"], bm25=ctx.get("bm25")
    ),
    "list_sources": lambda cfg, ctx: make_list_sources(cfg, store=ctx["store"]),
    "read_source": lambda cfg, ctx: make_read_source(cfg, store=ctx["store"]),
}


def build_registry(
    cfg: AppConfig,
    *,
    store,
    embedder=None,
    reranker=None,
    bm25=None,
    names=None,
) -> ToolRegistry:
    """Build the tools named in `cfg.agent.tools`.

    `store` is INJECTED and never opened here. Embedded Qdrant locks a directory
    for one process; a tool calling `open_store()` while the UI holds that
    handle would fail on Windows with a message about a storage folder.
    """
    ctx = {"store": store, "embedder": embedder, "reranker": reranker, "bm25": bm25}
    registry = ToolRegistry()
    for name in names if names is not None else cfg.agent.tools:
        builder = BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown tool {name!r}. Known tools: {sorted(BUILDERS)}."
            )
        registry.register(builder(cfg, ctx))
    return registry
