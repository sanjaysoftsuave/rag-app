"""Bi-encoder embedding with per-family prompt handling.

The topic that bites people here is *asymmetry*. A question and the passage
that answers it are not the same kind of text, and the leading open models
encode that difference with a prefix baked in during training:

  all-MiniLM-L6-v2   symmetric   — no prefix, query and passage encoded alike
  e5-*               asymmetric  — "query: " / "passage: " (required)
  bge-*              asymmetric  — instruction on the QUERY only
  gte-*              symmetric   — no prefix

Drop the prefix on an E5 or BGE model and it still runs, still returns
plausible vectors, and quietly loses a chunk of its retrieval quality — there
is no error to notice. That silent-degradation failure is the reason this
registry exists instead of a bare model name.

MTEB (huggingface.co/spaces/mteb/leaderboard) is the standard benchmark for
choosing between these. Read the *Retrieval* column, not the overall average:
a model can top the average on clustering and classification while being
mediocre at the one job we need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    note: str = ""

    @property
    def asymmetric(self) -> bool:
        return bool(self.query_prefix or self.passage_prefix)


# Small, CPU-friendly models. All three sit within a few MTEB retrieval points
# of each other while differing ~5x in size — which is the actual trade.
MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        note="22M params. Symmetric, no prefix. Fast baseline, weakest on MTEB retrieval.",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-small-en-v1.5",
        dim=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        note="33M params. Instruction on the query only; passages stay bare.",
    ),
    "intfloat/e5-small-v2": EmbeddingModelSpec(
        name="intfloat/e5-small-v2",
        dim=384,
        query_prefix="query: ",
        passage_prefix="passage: ",
        note="33M params. BOTH sides need a prefix — omitting them measurably hurts.",
    ),
}


def spec_for(model_name: str) -> EmbeddingModelSpec:
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    lowered = model_name.lower()
    # Unknown checkpoint: infer from the family name rather than silently
    # treating an asymmetric model as symmetric.
    if "e5" in lowered:
        return EmbeddingModelSpec(model_name, dim=0, query_prefix="query: ",
                                  passage_prefix="passage: ", note="inferred e5 family")
    if "bge" in lowered:
        return EmbeddingModelSpec(
            model_name, dim=0,
            query_prefix="Represent this sentence for searching relevant passages: ",
            note="inferred bge family",
        )
    return EmbeddingModelSpec(model_name, dim=0, note="unknown model, assumed symmetric")


class Embedder:
    """Thin wrapper around a sentence-transformers bi-encoder.

    `encode_queries` and `encode_documents` are separate on purpose. Calling
    the wrong one on an asymmetric model is the single easiest way to lose
    retrieval quality without seeing an error.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.spec = spec_for(model_name)
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros(
                (0, self._model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Backwards-compatible alias for document encoding."""
        return self.encode_documents(texts)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
