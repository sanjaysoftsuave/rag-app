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
cross-encoder expensively sorts those K correctly. It earns its cost on
near-duplicate passages — several sections describing variants of one thing.
Those sit almost on top of each other in embedding space, and only a model that
reads the query *against* each passage gets the ordering right.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Protocol

from rag_app.chunking import Chunk
from rag_app.store import ScoredChunk


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass(frozen=True)
class RerankerSpec:
    """What a reranker costs and what scale it scores on.

    `score_scale` is the field that matters and the one people forget.
    `score_threshold` is calibrated against a SPECIFIC model's output
    distribution: ms-marco emits unbounded logits (roughly -11..+11) that need
    a sigmoid to be interpretable, while Cohere returns a 0-1 relevance
    directly and must NOT be squashed again — sigmoid(0.9) is 0.71, which would
    silently drag every good score toward the middle and make the gate mean
    something else entirely.

    Swapping rerankers therefore requires re-checking `score_threshold`. It
    needs no re-ingest: the reranker never touches stored vectors.
    """

    name: str
    kind: str            # "cross-encoder" (local) | "cohere" (API)
    params: str
    score_scale: str     # the rerank_score_scale this model expects
    note: str = ""


RERANKER_REGISTRY: dict[str, RerankerSpec] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RerankerSpec(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        kind="cross-encoder", params="22M", score_scale="sigmoid",
        note="The ecosystem default. ~25 ms/pair on CPU. Unbounded logits.",
    ),
    "cross-encoder/ms-marco-MiniLM-L-12-v2": RerankerSpec(
        name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        kind="cross-encoder", params="33M", score_scale="sigmoid",
        note="Same family and training, twice the depth. ~2x slower, a little "
             "more accurate on close calls.",
    ),
    "BAAI/bge-reranker-base": RerankerSpec(
        name="BAAI/bge-reranker-base",
        kind="cross-encoder", params="278M", score_scale="sigmoid",
        note="Clearly stronger, ~10x the parameters. Painful on CPU; sensible "
             "with a GPU. Different logit spread — re-tune score_threshold.",
    ),
    "BAAI/bge-reranker-v2-m3": RerankerSpec(
        name="BAAI/bge-reranker-v2-m3",
        kind="cross-encoder", params="568M", score_scale="sigmoid",
        note="Stronger still and multilingual. GPU territory.",
    ),
    "cohere/rerank-v3.5": RerankerSpec(
        name="cohere/rerank-v3.5",
        kind="cohere", params="API", score_scale="raw",
        note="Hosted. Returns a 0-1 relevance score already — use "
             "rerank_score_scale: raw, NOT sigmoid. Needs COHERE_API_KEY.",
    ),
}


def reranker_spec(model_name: str) -> RerankerSpec:
    """Known spec, or a conservative guess for an unregistered checkpoint."""
    if model_name in RERANKER_REGISTRY:
        return RERANKER_REGISTRY[model_name]
    kind = "cohere" if model_name.startswith("cohere/") else "cross-encoder"
    return RerankerSpec(
        name=model_name, kind=kind, params="?",
        score_scale="raw" if kind == "cohere" else "sigmoid",
        note="Unregistered model — verify its score range before trusting "
             "score_threshold.",
    )


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
    """A local cross-encoder. One forward pass per (query, passage) pair."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


class CohereReranker:
    """Hosted reranking via Cohere's API.

    Returns a relevance score already in 0-1, so `rerank_score_scale` must be
    `raw` — squashing it through a sigmoid a second time would compress every
    score toward 0.5 and quietly change what the gate means.

    The API returns results sorted and keyed by index; this restores the input
    order so `rerank_all` can do its own sorting and the caller's zip stays
    aligned. Getting that wrong would scramble scores onto the wrong chunks —
    a failure that looks like "the reranker is bad" rather than a bug.
    """

    def __init__(self, model_name: str, api_key: str | None = None, client=None):
        self.model_name = model_name
        self.model = model_name.split("/", 1)[-1]
        self._client = client
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Add it to .env, or choose a local "
                "cross_encoder_model — see `python -m rag_app models`."
            )
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Cohere reranking needs the cohere package:\n"
                '    pip install -e ".[cohere]"      (or: pip install cohere)'
            ) from exc
        self._client = cohere.ClientV2(self._api_key)
        return self._client

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [text for _, text in pairs]
        response = self._get_client().rerank(
            model=self.model, query=query, documents=documents, top_n=len(documents)
        )
        scores = [0.0] * len(documents)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


def build_reranker(model_name: str, **kwargs):
    """Construct whichever reranker `model_name` names."""
    if reranker_spec(model_name).kind == "cohere":
        return CohereReranker(model_name, **kwargs)
    return CrossEncoderReranker(model_name)


def rerank_all(
    query: str,
    candidates: list[ScoredChunk],
    scorer: Callable[[list[tuple[str, str]]], list[float]] | CrossEncoderLike,
    scale: str = "raw",
) -> list[ScoredChunk]:
    """Re-score every (query, chunk) pair and return ALL of them, sorted.

    The cross-encoder has to score all K candidates to rank them, so the
    K-minus-N that don't make the cut are already computed — `rerank()` just
    discards them. Returning them costs nothing and is the only way to see
    *which* candidate the reranker demoted out of the LLM's context — the
    evidence that separates "the bi-encoder never retrieved it" from "the
    cross-encoder ranked it out", which have different fixes.
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
    return rescored


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
    return rerank_all(query, candidates, scorer, scale)[:n]


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
