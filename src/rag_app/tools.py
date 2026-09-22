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

from rag_app.config import (
    CAPABILITIES,
    READ_DOCUMENT,
    READ_INDEX,
    AppConfig,
)
from rag_app.injection import Guard, scan
from rag_app.rerank import rerank_all
from rag_app.retrieve import retrieve

Observation = str

# The loop's terminal action. It is not a tool: registering one under this name
# would shadow termination and the agent could never stop.
FINAL_ANSWER = "final_answer"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ToolCall:
    """What came back from asking the registry to run a tool.

    `denied` is "" when the tool ran. Otherwise it names WHY, and the
    observation explains it to the model in words — a denial is an observation,
    never an exception, because a raise takes down the loop and loses the whole
    trajectory.
    """

    observation: Observation
    denied: str = ""
    detections: tuple = ()


@dataclass(frozen=True)
class Tool:
    name: str
    description: str      # one line; goes verbatim into the prompt
    input_desc: str       # what Action Input should contain
    usage: str            # one worked example line
    run: Callable[[str], Observation]
    # Defaulted, so every existing five-argument construction keeps working.
    # READ_INDEX is the weaker grade; a tool that reads a whole document has to
    # ask for READ_DOCUMENT explicitly.
    capability: str = READ_INDEX


class ToolRegistry:
    def __init__(
        self,
        *,
        labels_fn=None,
        guard: Guard | None = None,
        denied: tuple[str, ...] = (),
        granted: tuple[str, ...] = CAPABILITIES,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._labels_fn = labels_fn
        self._labels: frozenset[str] | None = None
        self.guard = guard
        self._denied = tuple(denied)
        self._granted = tuple(granted)

    @property
    def labels_fn(self):
        """The label resolver this registry was built with, or None.

        Public so a module composing a NEW registry over the same store (Week
        9's `mcp_client.build_mcp_registry`) can carry the same verification
        forward instead of silently falling back to the citable_labels()
        fail-open.
        """
        return self._labels_fn

    def citable_labels(self) -> frozenset[str] | None:
        """The labels a tool in this registry could legitimately have printed.

        `None` means "this registry cannot vouch for any label" — a bare
        registry built in a test. `build_registry` ALWAYS installs a resolver
        and a test pins that, so the production path is never unverified; the
        fail-open exists only so ~40 scripted tests need not each be handed a
        store, and it is recorded in meta whenever it happens.

        Cached after the first call: one full scroll per registry, the same cost
        BM25 already pays.
        """
        if self._labels_fn is None:
            return None
        if self._labels is None:
            self._labels = frozenset(self._labels_fn())
        return self._labels

    def restrict(self, denied) -> "ToolRegistry":
        """A view of this registry with some tools denied.

        Shares the same Tool objects, guard and label resolver — a copy would
        drain detections into the wrong place and re-scroll the store.
        """
        denied = tuple(denied or ())
        if not denied:
            return self
        clone = ToolRegistry(
            labels_fn=self._labels_fn,
            guard=self.guard,
            denied=self._denied + denied,
            granted=self._granted,
        )
        clone._tools = self._tools
        clone._labels = self._labels
        return clone

    def invoke(self, name: str, argument: str) -> ToolCall:
        """Check, then run. Every denial is an observation the model can read.

        `forbid_tools` used to be a post-hoc metric: a forbidden tool still ran
        and merely failed the score afterwards. This is where it becomes
        enforcement.
        """
        tool = self._tools.get(name)
        allowed = ", ".join(self.names())
        if tool is None or name in self._denied:
            reason = "unknown" if tool is None else "forbidden-by-task"
            verb = (
                f"Unknown tool {name!r}."
                if tool is None
                else f"{name} is not available for this task."
            )
            return ToolCall(
                f"{verb} Available: {allowed}, {FINAL_ANSWER}. Nothing was run and no "
                f"excerpt was returned; answer from what the available tools give you, "
                f"or say you cannot.",
                denied=reason,
            )
        if tool.capability not in self._granted:
            return ToolCall(
                f"{name} needs the {tool.capability!r} capability, which this run does "
                f"not grant (granted: {', '.join(self._granted) or 'nothing'}). Nothing "
                f"was read.",
                denied="capability",
            )
        observation = tool.run(argument)
        detections = self.guard.drain() if self.guard is not None else ()
        return ToolCall(observation, detections=detections)

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
        if tool.capability not in CAPABILITIES:
            raise ValueError(
                f"Tool {tool.name!r} declares capability {tool.capability!r}, which is "
                f"not one of {list(CAPABILITIES)}. An unrecognised grade would be "
                f"denied at every run, so it is rejected here where the message is "
                f"readable."
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
        """Only the tools this run may call — a denied tool is not advertised.

        `invoke` still tells *denied* from *unknown*, which is what keeps the
        attempt measurable rather than indistinguishable from a typo.
        """
        return [n for n in self._tools if n not in self._denied]

    def __len__(self) -> int:
        return len(self._tools)

    def describe_for_prompt(self) -> str:
        lines = []
        for name, tool in self._tools.items():
            if name in self._denied:
                continue
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


def _blocks(items, guard: Guard | None = None) -> str:
    """Excerpts labelled exactly as generate.build_prompt labels them.

    The BODY is neutralized here and nowhere else. Once the header and the body
    are one string they are indistinguishable — `[handbook.md]` is our header
    and `[invoice.pdf]` planted in a body looks identical — so the trust
    boundary is only knowable at the moment the body enters the text.
    """
    return "\n\n".join(
        f"[{c.chunk.source}]\n{guard.clean(c.chunk.text) if guard else c.chunk.text}"
        for c in items
    )


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


def make_search_documents(cfg: AppConfig, *, store, embedder, reranker, guard=None) -> Tool:
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
        return _truncate(_blocks(top, guard), cfg.agent.max_observation_chars)

    return Tool(
        name="search_documents",
        description="Search the documents by meaning. Returns labelled excerpts.",
        input_desc="a search query in plain words",
        usage="Action Input: how long is a password reset link valid",
        run=_safe("search_documents", run),
    )


def make_keyword_search(cfg: AppConfig, *, store, bm25=None, reranker=None, guard=None) -> Tool:
    """Exact-token search. Finds error codes dense retrieval treats as noise.

    THE GATE, AND WHY IT IS NOT AN ABSOLUTE BM25 FLOOR
    ---------------------------------------------------
    This tool used to have no relevance gate at all, while `search_documents`
    carried the relocated one — so planting a rare token was enough to reach the
    model regardless of relevance.

    The obvious fix, comparing the BM25 score to `cfg.score_threshold`, is a
    category error: BM25 scores are unbounded and CORPUS-RELATIVE (the same
    document scores differently as the corpus grows), while `score_threshold` is
    a sigmoid-scaled cross-encoder probability. Any constant floor would be
    meaningless and the two tools would be gated on incomparable scales.

    So the BM25 hits are RERANKED and gated on the same scale as dense search.
    One number to calibrate, and the tool keeps its reason to exist: BM25 still
    chooses the candidates, the cross-encoder only judges them.
    """

    def run(query: str) -> Observation:
        query = (query or "").strip()
        if not query:
            return "keyword_search needs a term to look for."
        from rag_app.bm25 import BM25Index

        index = bm25 or BM25Index.from_store(store)
        hits = index.search(query, k=max(cfg.rerank_n, cfg.retrieval.bm25_pool))
        if not hits:
            return f"No document contains {query!r}."
        if reranker is None:
            # Degrade VISIBLY. `build_registry` always supplies a reranker, so
            # this path is reachable only from a hand-built test registry — but
            # an ungated observation that looked identical to a gated one is
            # precisely the silent hole this tool's gate exists to close.
            return _truncate(
                "[ungated: no reranker, so the relevance gate was skipped]\n"
                + _blocks(hits, guard),
                cfg.agent.max_observation_chars,
            )
        ranked = rerank_all(query, hits, scorer=reranker, scale=cfg.rerank_score_scale)
        top = ranked[: cfg.rerank_n]
        if not top or top[0].score < cfg.score_threshold:
            best = top[0].score if top else 0.0
            return (
                f"No excerpt scored above the relevance threshold "
                f"(best {best:.4f} < {cfg.score_threshold}). {query!r} appears in the "
                f"documents but not in anything relevant to this question."
            )
        return _truncate(_blocks(top, guard), cfg.agent.max_observation_chars)

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


def make_list_sources(cfg: AppConfig, *, store, guard=None) -> Tool:
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
        # NOT cleaned: this output is registry-generated and trusted. But the
        # LABELS came from documents, so an instruction-shaped filename is still
        # worth flagging.
        if guard is not None:
            for label in seen:
                guard.pending.extend(scan(label).detections)
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


def make_read_source(cfg: AppConfig, *, store, allow=(), guard=None) -> Tool:
    """Read one document end to end — the multi-hop enabler, and the
    exfiltration primitive.

    `allow` scopes which documents this run may open; `()` means unrestricted,
    and the tool's DESCRIPTION says so, because a scope that silently is not one
    is worse than having none.

    There is deliberately NO path-traversal check. This tool filters
    `store.all_chunks()` by an exact label and never touches the filesystem, so
    a `..` guard would be theatre that teaches the wrong lesson about where the
    risk actually is — which is reach across the index, not escape from a
    directory.
    """

    def run(label: str) -> Observation:
        label = (label or "").strip().strip("[]")
        if not label:
            return "read_source needs a document label. Use list_sources to see them."
        if allow and label not in allow:
            return (
                f"read_source may only open the documents this task allows: "
                f"{', '.join(allow)}. {label!r} is not one of them, so nothing was read."
            )
        chunks = [c for c in store.all_chunks() if c.source == label]
        if not chunks:
            return (
                f"No document is labelled {label!r}. Use list_sources to see the legal labels."
            )
        body = "\n\n".join(c.text for c in chunks)
        if guard is not None:
            body = guard.clean(body)
        return _truncate(f"[{label}]\n{body}", cfg.agent.max_observation_chars)

    return Tool(
        name="read_source",
        description=(
            "Read one whole document, given its label."
            + (f" Only these are allowed: {', '.join(allow)}." if allow else "")
        ),
        input_desc="a document label exactly as list_sources printed it",
        usage="Action Input: handbook.pdf",
        run=_safe("read_source", run),
        capability=READ_DOCUMENT,
    )


BUILDERS = {
    "search_documents": lambda cfg, ctx: make_search_documents(
        cfg, store=ctx["store"], embedder=ctx["embedder"], reranker=ctx["reranker"],
        guard=ctx.get("guard"),
    ),
    "keyword_search": lambda cfg, ctx: make_keyword_search(
        cfg, store=ctx["store"], bm25=ctx.get("bm25"), reranker=ctx["reranker"],
        guard=ctx.get("guard"),
    ),
    "list_sources": lambda cfg, ctx: make_list_sources(
        cfg, store=ctx["store"], guard=ctx.get("guard")
    ),
    "read_source": lambda cfg, ctx: make_read_source(
        cfg, store=ctx["store"], allow=cfg.agent.read_source_allow,
        guard=ctx.get("guard"),
    ),
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
    defences = cfg.agent.defences
    guard = Guard(enabled=defences.neutralize) if defences.neutralize else None
    ctx = {
        "store": store, "embedder": embedder, "reranker": reranker,
        "bm25": bm25, "guard": guard,
    }
    # The production path ALWAYS knows its citable labels — a test pins this, so
    # `citable_labels() is None` can only ever mean a hand-built test registry.
    labels_fn = None
    if defences.verify_evidence and store is not None:
        labels_fn = lambda: frozenset(c.source for c in store.all_chunks())  # noqa: E731
    registry = ToolRegistry(
        labels_fn=labels_fn,
        guard=guard,
        granted=CAPABILITIES if defences.enforce_capabilities else CAPABILITIES,
    )
    for name in names if names is not None else cfg.agent.tools:
        builder = BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"Unknown tool {name!r}. Known tools: {sorted(BUILDERS)}."
            )
        registry.register(builder(cfg, ctx))
    return registry
