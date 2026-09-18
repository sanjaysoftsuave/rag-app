"""Shared fakes.

Every test here runs offline in the sense that matters most: nothing
downloads a model or calls a real LLM API. Vector storage is the one
exception — Qdrant is the only backend the app has, so tests build real
(embedded) Qdrant stores via `make_qdrant_store()` below rather than a
numpy-shaped stand-in. Embedded Qdrant writes real files to `tmp_path` and
holds a real file lock while open; nothing here talks to a network.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_app.config import AppConfig, ChunkPreset, LlmConfig, QdrantConfig
from rag_app.qdrant_store import QdrantStore
from rag_app.store import StoreMeta


class FakeEmbedder:
    """Deterministic hash-based vectors. Normalized, because the store's dot
    product is only cosine if the inputs are unit length."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.model_name = "fake"

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_documents(self, texts):
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    encode = encode_documents

    def encode_queries(self, texts):
        return self.encode_documents(texts)


class FixedEmbedder:
    """Returns the same vector for everything — makes ranking depend purely on
    the reranker, which is what several gate tests want to isolate."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def encode_documents(self, texts):
        v = np.zeros((len(texts), self.dim), dtype=np.float32)
        v[:, 0] = 1.0
        return v

    encode = encode_documents
    encode_queries = encode_documents


class FakeReranker:
    def __init__(self, score: float = 0.9, per_text: dict[str, float] | None = None):
        self.score = score
        self.per_text = per_text or {}

    def predict(self, pairs):
        out = []
        for _, text in pairs:
            value = self.score
            for key, score in self.per_text.items():
                if key in text:
                    value = score
                    break
            out.append(value)
        return out


def make_config(tmp_path, **overrides) -> AppConfig:
    base = dict(
        docs_dir=tmp_path / "docs",
        tickets_dir=tmp_path / "tickets",
        store_dir=tmp_path / "store",
        chunk_presets={
            "A": ChunkPreset(500, 50),
            "C": ChunkPreset(2000, 200),
        },
        default_preset="C",
        bi_encoder_model="fake",
        cross_encoder_model="fake-ce",
        retrieve_k=5,
        rerank_n=3,
        score_threshold=0.5,
        rerank_score_scale="raw",
        qdrant=QdrantConfig(),
        llm=LlmConfig(base_url="http://example", model="x", temperature=0.0),
        llm_api_key="test-key",
    )
    base.update(overrides)
    return AppConfig(**base)  # type: ignore[arg-type]


def make_qdrant_store(tmp_path, chunks, vectors, meta: StoreMeta | None = None, name: str = "qstore"):
    """Build a real, embedded QdrantStore for a test.

    `name` lets one test build more than one independent store under the same
    `tmp_path` without them colliding on the same collection directory.
    `meta` defaults to a placeholder — `QdrantStore.build()` requires one
    (unlike the old numpy store, which tolerated `meta=None`), so most tests
    that don't care about provenance just get a generic one for free.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    meta = meta or StoreMeta("fake", dim, 500, 50, len(chunks))
    path = tmp_path / name
    store = QdrantStore(meta_dir=path, path=path)
    store.build(chunks, vectors, meta)
    return store


class FakeJudge:
    """A `judge_fn(system, user, cfg) -> str`, recording everything it was sent.

    One seam serves six prompt types (the gold judge, the trace judge, G-Eval
    and the four RAGAS prompts), so this one fake covers all of them.

    `replies` is either a single string answered to everything, or a mapping of
    {system prompt constant: reply}. A list value is popped in order, which is
    what makes multi-sample G-Eval testable. An unmatched system prompt raises
    rather than returning a default: a scorer sending an unexpected prompt is a
    bug worth failing on, not one worth quietly scoring.
    """

    def __init__(self, replies, calls_raise: bool = False):
        self.replies = replies
        self.calls_raise = calls_raise
        self.seen: list[tuple[str, str]] = []

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def systems(self) -> list[str]:
        return [s for s, _ in self.seen]

    def __call__(self, system, user, cfg):
        self.seen.append((system, user))
        if self.calls_raise:
            raise RuntimeError("the judge API is down")
        if isinstance(self.replies, str):
            return self.replies
        try:
            reply = self.replies[system]
        except KeyError:
            raise AssertionError(
                f"FakeJudge got an unscripted system prompt "
                f"(starts {system[:60]!r})"
            ) from None
        if isinstance(reply, list):
            if not reply:
                raise AssertionError("FakeJudge ran out of scripted replies")
            return reply.pop(0)
        return reply


def verdict_json(verdict="correct", reasoning="because", confidence=0.9) -> str:
    import json

    return json.dumps(
        {"reasoning": reasoning, "verdict": verdict, "confidence": confidence}
    )


def geval_json(score=4, reasoning="because") -> str:
    import json

    return json.dumps({"reasoning": reasoning, "score": score})


def scripted_llm(*replies):
    """An `llm_fn(messages, cfg) -> str` that returns `replies` in order.

    Records every prompt in `.seen`, which is what lets a test prove the loop
    actually loops (the observation of step N must appear in the prompt of step
    N+1). Running past the script raises with a COUNT rather than an IndexError:
    a runaway loop should fail saying how many turns it took.
    """
    queue = list(replies)
    seen: list[list[dict]] = []

    def _call(messages, cfg):
        seen.append(messages)
        if not queue:
            raise AssertionError(
                f"the agent asked for reply #{len(seen)} but only "
                f"{len(replies)} were scripted"
            )
        return queue.pop(0)

    _call.seen = seen  # type: ignore[attr-defined]
    return _call


def boom_llm(messages, cfg):
    raise RuntimeError("the API is down")


class FakeClock:
    """Deterministic monotonic clock: every read advances by `step`."""

    def __init__(self, start: float = 0.0, step: float = 1.0):
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value
