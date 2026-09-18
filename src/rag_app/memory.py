"""What the agent remembers: short-term, long-term, summarized, and semantic.

FOUR FLAVOURS, ONE PROTOCOL
---------------------------
  ConversationBuffer  in-process, the last N turns verbatim (short-term)
  SessionLog          append-only JSONL, survives the process (long-term)
  summarize_history   compresses the oldest turns when the buffer grows
  MemoryStore         embedded Qdrant over past turns (semantic recall)

`Memory` is the protocol `run_agent(memory=...)` accepts, so the plain
implementation and the mem0 one are interchangeable at that seam and can be
compared without changing the agent.

WHY VECTOR MEMORY GETS ITS OWN QDRANT DIRECTORY
------------------------------------------------
Three options were available, and only one survives contact with this codebase.

  1. Same collection as the corpus. Rejected loudly: `search_documents` would
     then retrieve MEMORIES and hand them to the model inside a `[source]`
     header, so a remembered guess becomes a citable document. That destroys
     grounding outright.

  2. Same directory, different collection. Rejected: `QdrantClient(path=...)`
     locks the DIRECTORY, not the collection, so this shares the corpus lock
     anyway — and `ui.close_handles(preset)` would close memory mid-conversation
     every time someone re-ingests.

  3. Separate directory, separate collection. Chosen. Its own lock, its own
     lifetime, unaffected by ingest. Costs two embedded clients in one process,
     which is fine because they are different directories.

THE JSONL IS THE SOURCE OF TRUTH; QDRANT IS A DERIVED INDEX
-------------------------------------------------------------
`QdrantStore.build()` deletes and recreates a collection — it is a batch build,
not an upsert path. So appending a memory rebuilds the collection from the
JSONL. That is exactly the relationship the corpus already has (files are truth,
ingest derives the index), it is milliseconds at memory scale, and decisively it
means ZERO changes to `qdrant_store.py`, the most scarred file in this repo.

It is O(N) per write. At a few hundred turns that is invisible; at tens of
thousands it would need a real upsert path. Recorded rather than discovered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from rag_app.config import AppConfig


@dataclass(frozen=True)
class Turn:
    role: str          # "user" | "agent" | "summary"
    text: str
    ts: str = ""

    @staticmethod
    def now(role: str, text: str) -> "Turn":
        return Turn(role, text, datetime.now(timezone.utc).isoformat(timespec="seconds"))


class Memory(Protocol):
    def remember(self, turn: Turn) -> None: ...
    def recall(self, query: str) -> str: ...
    def history(self) -> list[Turn]: ...
    def describe(self) -> str: ...
    def close(self) -> None: ...


def memory_dir(cfg: AppConfig) -> Path:
    """Where memory lives. Derived from store_dir, never a hard-coded repo path.

    Mirrors `evaluate.gold_path`. Deriving it means a test using
    `make_config(tmp_path)` automatically keeps memory inside tmp_path — a
    hard-coded default would let the suite write into the real repo.
    """
    return cfg.agent.memory.dir or (cfg.store_dir.parent / "agent_memory")


# ---------------------------------------------------------------------------
# Short-term
# ---------------------------------------------------------------------------


@dataclass
class ConversationBuffer:
    """The last N turns, verbatim, in process."""

    limit: int = 8
    turns: list[Turn] = field(default_factory=list)

    def remember(self, turn: Turn) -> None:
        self.turns.append(turn)

    def recall(self, query: str = "") -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.history())

    def history(self) -> list[Turn]:
        return self.turns[-self.limit :] if self.limit else list(self.turns)

    def describe(self) -> str:
        return f"conversation buffer: {len(self.turns)} turns, last {self.limit} recalled"

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Long-term
# ---------------------------------------------------------------------------


@dataclass
class SessionLog:
    """Append-only JSONL, one object per line.

    JSONL rather than a single JSON document, matching `data/traces/*.jsonl`:
    appending is one write with no read-modify-write, and a truncated write
    costs one line instead of the whole file.
    """

    path: Path
    session: str = "default"

    def remember(self, turn: Turn) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": turn.ts, "session": self.session, "role": turn.role, "text": turn.text}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def history(self) -> list[Turn]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial final line must not lose the whole log
            if row.get("session", "default") == self.session:
                out.append(Turn(row.get("role", ""), row.get("text", ""), row.get("ts", "")))
        return out

    def recall(self, query: str = "") -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.history())

    def describe(self) -> str:
        return f"session log at {self.path} ({len(self.history())} turns)"

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


SUMMARY_SYSTEM = (
    "Compress this conversation history into a short factual summary. Keep every "
    "specific term, name, number and document label — they are what makes the "
    "summary useful later. Drop pleasantries and repetition. Reply with the summary "
    "and nothing else."
)


@dataclass(frozen=True)
class MemorySummary:
    turns: list[Turn] = field(default_factory=list)
    failed: bool = False
    detail: str = ""

    def describe(self) -> str:
        if self.failed:
            return f"summarization failed; {self.detail}"
        return f"summarized to {len(self.turns)} turns"


def summarize_history(
    turns: list[Turn],
    cfg: AppConfig,
    *,
    llm_fn=None,
    keep_recent: int = 2,
) -> MemorySummary:
    """Replace the oldest turns with one summary turn.

    Goes through the SAME `llm_fn` seam the agent uses, so it is exercised
    offline by a scripted fake rather than needing its own mock.

    Failure policy copied from `transform_query`: on any exception, hard-drop
    the oldest turns and say so. Degraded but visible beats an exception that
    loses the conversation.
    """
    total = sum(len(t.text) for t in turns)
    if total <= cfg.agent.memory.summary_trigger_chars:
        return MemorySummary(turns=list(turns))

    old, recent = turns[:-keep_recent], turns[-keep_recent:]
    if not old:
        return MemorySummary(turns=list(turns))

    body = "\n".join(f"{t.role}: {t.text}" for t in old)
    target = cfg.agent.memory.summary_target_chars
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM + f" Keep it under {target} characters."},
        {"role": "user", "content": body},
    ]

    caller = llm_fn
    if caller is None:
        from rag_app.llm import chat_messages

        def caller(msgs, config):
            return chat_messages(msgs, config.llm, config)

    try:
        text = (caller(messages, cfg) or "").strip()
    except Exception as exc:
        return MemorySummary(
            turns=recent,
            failed=True,
            detail=f"oldest {len(old)} turns dropped ({type(exc).__name__})",
        )
    if not text:
        return MemorySummary(
            turns=recent, failed=True, detail=f"oldest {len(old)} turns dropped (empty summary)"
        )
    return MemorySummary(turns=[Turn.now("summary", text), *recent])


# ---------------------------------------------------------------------------
# Vector memory
# ---------------------------------------------------------------------------


class MemoryStore:
    """Semantic recall over past turns, in its own embedded Qdrant directory."""

    COLLECTION_DIRNAME = "qdrant_memory"

    def __init__(self, cfg: AppConfig, *, embedder, session: str = "default"):
        self.cfg = cfg
        self.embedder = embedder
        self.session = session
        self.root = memory_dir(cfg)
        self.log = SessionLog(self.root / "turns.jsonl", session=session)
        self._store = None

    @property
    def path(self) -> Path:
        return self.root / self.COLLECTION_DIRNAME

    def _open(self):
        from rag_app.qdrant_store import QdrantStore

        if self._store is None:
            self._store = QdrantStore(meta_dir=self.path, path=self.path)
        return self._store

    def _rebuild(self) -> None:
        """Rebuild the index from the JSONL, which is the source of truth."""
        from rag_app.chunking import Chunk
        from rag_app.store import StoreMeta

        turns = [t for t in self.log.history() if t.text.strip()]
        if not turns:
            return
        chunks = [
            Chunk(f"mem::{i}", f"memory:{self.session}", t.text, {"role": t.role})
            for i, t in enumerate(turns)
        ]
        # Memories are passages, so encode_documents. Recall uses
        # encode_queries. The asymmetry matters here as much as for the corpus.
        vectors = self.embedder.encode_documents([c.text for c in chunks])
        meta = StoreMeta(
            embedding_model=self.cfg.bi_encoder_model,
            dim=int(vectors.shape[1]),
            chunk_size=0,
            overlap=0,
            n_chunks=len(chunks),
        )
        self._open().build(chunks, vectors, meta)

    def remember(self, turn: Turn) -> None:
        self.log.remember(turn)
        self._rebuild()

    def recall(self, query: str) -> str:
        from rag_app.retrieve import retrieve

        try:
            store = self._open()
            if not store.exists():
                return ""
            vector = self.embedder.encode_queries([query])[0]
            hits = retrieve(store, vector, k=self.cfg.agent.memory.vector_recall_k)
        except Exception:
            return ""  # memory is an optimisation; never take down a question for it
        return "\n".join(h.chunk.text for h in hits)

    def history(self) -> list[Turn]:
        return self.log.history()

    def describe(self) -> str:
        return f"vector memory at {self.path} ({len(self.history())} turns)"

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
