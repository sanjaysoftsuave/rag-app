"""mem0-backed memory, for comparison against the plain implementation.

WHAT MEM0 DOES THAT `memory.py` DOES NOT
-----------------------------------------
The plain `MemoryStore` remembers TURNS: it stores what was said and retrieves
the semantically nearest ones. mem0 remembers FACTS: it runs an LLM over each
turn to extract salient statements, deduplicates them against what it already
holds, and updates or deletes entries that a later turn contradicts.

That is a genuinely different capability. "My account is the Pro plan" followed
later by "we upgraded to Enterprise" leaves the plain store holding both turns
and recalling whichever is textually closer; mem0 is designed to notice the
second supersedes the first.

WHAT IT COSTS
-------------
  * an extra LLM call per remembered turn, for extraction
  * an opaque store: what it decided to keep is not a file you can read
  * a large dependency that brings its own vector store and LLM client, into a
    venv holding pinned torch/openai/qdrant-client

WHY THERE ARE NO BEHAVIOUR TESTS FOR THIS MODULE
--------------------------------------------------
Stated plainly rather than papered over: mem0 cannot be exercised offline. Even
its local mode needs a configured LLM and embedder, so a test that made it store
and recall a fact would make network calls. The suite therefore tests only the
protocol shape and the import guard, and this docstring is the record of why the
coverage here is thinner than everywhere else in the repo.

The plain `MemoryStore` is fully tested offline, which is itself part of the
comparison: an implementation you can test without a network is worth something.
"""

from __future__ import annotations

from rag_app.config import AppConfig
from rag_app.memory import Turn, memory_dir

EXTRA_HINT = 'mem0 is not installed. Run: pip install -e ".[mem0]"'


def _require():
    try:
        import mem0  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(EXTRA_HINT) from exc


def available() -> bool:
    from importlib.util import find_spec

    return find_spec("mem0") is not None


class Mem0Memory:
    """Implements the same `Memory` protocol as `memory.MemoryStore`.

    Interchangeable at `run_agent(memory=...)`, which is the whole point: the
    agent cannot tell which memory it was given, so the comparison is about the
    memory and not about two different agents.
    """

    def __init__(self, cfg: AppConfig, *, session: str = "default", client=None):
        self.cfg = cfg
        self.session = session
        self.root = memory_dir(cfg) / "mem0"
        self._client = client
        self._turns: list[Turn] = []

    def _open(self):
        if self._client is None:
            _require()
            from mem0 import Memory as _Mem0

            self._client = _Mem0.from_config(
                {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"path": str(self.root)},
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": self.cfg.llm.model,
                            "openai_base_url": self.cfg.llm.base_url,
                            "api_key": self.cfg.llm_api_key,
                        },
                    },
                }
            )
        return self._client

    def remember(self, turn: Turn) -> None:
        self._turns.append(turn)
        # mem0 extracts facts here, with its own LLM call. Failure must not take
        # down the conversation, same policy as the plain store.
        try:
            self._open().add(turn.text, user_id=self.session)
        except Exception:
            return None

    def recall(self, query: str) -> str:
        try:
            hits = self._open().search(query, user_id=self.session)
        except Exception:
            return ""
        rows = hits.get("results", hits) if isinstance(hits, dict) else hits
        return "\n".join(str(r.get("memory", r)) for r in (rows or []))

    def history(self) -> list[Turn]:
        return list(self._turns)

    def describe(self) -> str:
        return f"mem0 memory at {self.root} (session {self.session})"

    def close(self) -> None:
        self._client = None
