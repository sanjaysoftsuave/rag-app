"""Cross-encoder reranking.

BI-ENCODER vs CROSS-ENCODER — the distinction the whole pipeline turns on:

  Bi-encoder    encodes query and passage *separately* into vectors, then
                compares with cosine. Passages are embedded once at ingest, so
                querying is a matrix multiply. Fast, indexable, and it never
                sees the query and passage together.

  Cross-encoder feeds [query, passage] through the transformer as ONE input and
                emits a single relevance score. Every pair costs a full forward
                pass, so it cannot be pre-indexed — but it can model word-level
                interaction the bi-encoder structurally cannot.

Hence retrieve-then-rerank: the bi-encoder cheaply narrows millions to K, the
cross-encoder expensively sorts those K correctly. On the ticket corpus this is
exactly what separates the three rate-limit tickets — they are near-identical in
embedding space, and only a model that reads "Free plan" *against* the question
gets the ordering right.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

from rag_app.chunking import Chunk
from rag_app.store import ScoredChunk


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def sigmoid(x: float) -> float:
    # math.exp overflows around |x| > 709; the CE never gets close, but clamp
    # anyway so a swapped-in model with a wilder range cannot crash the gate.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 709.0)))
    e = math.exp(max(x, -709.0))
    return e / (1.0 + e)


def apply_scale(scores: list[float], scale: str) -> list[float]:
    if scale == "raw":
        return scores
    if scale == "sigmoid":
        return [sigmoid(s) for s in scores]
    raise ValueError(f"Unknown rerank_score_scale {scale!r}")


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    n: int,
    scorer: Callable[[list[tuple[str, str]]], list[float]] | CrossEncoderLike,
    scale: str = "raw",
) -> list[ScoredChunk]:
    """Re-score (query, chunk) pairs and keep top-n.

    `scale` only rescales; it never reorders, since sigmoid is monotonic. It
    exists so the score gate can compare against a calibrated 0-1 number.
    """
    if not candidates:
        return []
    pairs = [(query, item.chunk.text) for item in candidates]
    predict = scorer.predict if hasattr(scorer, "predict") else scorer
    scores = apply_scale([float(s) for s in predict(pairs)], scale)
    rescored = [
        ScoredChunk(chunk=item.chunk, score=score)
        for item, score in zip(candidates, scores, strict=True)
    ]
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:n]


def stub_scorer(weights: dict[str, float]) -> Callable[[list[tuple[str, str]]], list[float]]:
    """Test helper: score by substring lookup in the chunk text."""

    def _score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for _, text in pairs:
            score = 0.0
            for key, value in weights.items():
                if key in text:
                    score = value
                    break
            out.append(score)
        return out

    return _score


def identity_chunks(chunks: list[Chunk], scores: list[float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=s) for c, s in zip(chunks, scores, strict=True)]
