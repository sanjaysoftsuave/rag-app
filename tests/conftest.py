"""Shared fakes.

Every test here runs offline. Nothing in this suite downloads a model or calls
an API — that is a hard constraint, not a convenience, because it is the only
reason the suite is fast enough to run on every change.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_app.config import AppConfig, ChunkPreset, LlmConfig, QdrantConfig
from rag_app.tickets import Ticket, Turn


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


def make_ticket(ticket_id: str, subject: str = "Subject", **kwargs) -> Ticket:
    defaults = dict(
        product="API",
        category="rate-limit",
        status="resolved",
        priority="high",
        channel="email",
        created_at="2026-01-01",
        customer_tier="free",
        conversation=[Turn("customer", "Question text"), Turn("agent", "Answer text")],
        resolution="Resolved.",
        tags=["tag"],
    )
    defaults.update(kwargs)
    return Ticket(ticket_id=ticket_id, subject=subject, **defaults)  # type: ignore[arg-type]


def make_config(tmp_path, **overrides) -> AppConfig:
    base = dict(
        docs_dir=tmp_path / "docs",
        tickets_dir=tmp_path / "tickets",
        store_dir=tmp_path / "store",
        chunk_presets={
            "A": ChunkPreset(500, 50, "flat"),
            "C": ChunkPreset(2000, 200, "ticket"),
        },
        default_preset="C",
        bi_encoder_model="fake",
        cross_encoder_model="fake-ce",
        retrieve_k=5,
        rerank_n=3,
        score_threshold=0.5,
        rerank_score_scale="raw",
        backend="numpy",
        qdrant=QdrantConfig(),
        llm=LlmConfig(base_url="http://example", model="x", temperature=0.0),
        llm_api_key="test-key",
    )
    base.update(overrides)
    return AppConfig(**base)  # type: ignore[arg-type]


@pytest.fixture
def tickets():
    return [
        make_ticket("TIC-001", "Free plan rate limit", customer_tier="free"),
        make_ticket("TIC-002", "Pro plan rate limit", customer_tier="pro"),
        make_ticket("TIC-003", "Refund timing", product="Billing", category="refund"),
    ]
