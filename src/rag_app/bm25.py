"""BM25 — exact keyword search, the complement to dense retrieval.

A bi-encoder compares meaning: it can match "why am I getting cut off" to
"rate limit exceeded" with no shared words. It is correspondingly bad at the
opposite case — an exact token like `TIC-1001`, `ERR-4032`, or `sk_test` is
just one point in embedding space among many, and nothing about the vector
says "this token must match literally." BM25 is the other direction: it does
not know that "429" and "rate limit" are related, but if the query contains
"429" and a chunk contains "429", that chunk wins, deterministically.

Implemented from scratch rather than via `rank_bm25` for the same reason the
numpy vector store is hand-rolled: this corpus is small enough that the real
algorithm is more legible as ~80 lines of Python than as an opaque dependency,
and CLAUDE.md's "plain Python by design" applies here too.

BM25 Okapi, the standard formulation:

    score(D, Q) = sum over query terms t of:
        idf(t) * f(t, D) * (k1 + 1)
                  ------------------------------------
                  f(t, D) + k1 * (1 - b + b * |D| / avgdl)

  f(t, D)   term frequency of t in document D
  |D|       document length in tokens
  avgdl     average document length across the corpus
  idf(t)    log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
  k1        term-frequency saturation (default 1.5) — how quickly repeating a
            term stops adding score. Too high and a document that just repeats
            "rate limit" ten times outranks one that says it once and answers
            the question.
  b         length normalization (default 0.75) — how much a document is
            penalized for being long. b=0 ignores length entirely; b=1 fully
            normalizes by it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter
from rag_app.store import ScoredChunk

# Keeps hyphenated/underscored tokens whole — "TIC-1001", "sk_test", "X-RateLimit"
# are exactly the identifiers keyword search exists to catch, and splitting them
# on punctuation would defeat the purpose.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    """Built once from a store's chunks, then queried repeatedly.

    Construction is O(corpus size); a query is O(query terms * postings), so
    building once per session (like the embedder/reranker) rather than per
    question is the right cache boundary — see `cli.shared_models`.
    """

    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._doc_tokens: list[list[str]] = [tokenize(c.text) for c in self.chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._tf: list[Counter] = [Counter(toks) for toks in self._doc_tokens]

        df: Counter = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self.chunks)
        # +1 inside the log keeps idf non-negative even for terms in every
        # document — a raw Robertson-Sparck-Jones idf can go negative there,
        # which would make a common word actively suppress a document's score.
        self._idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        length = self._doc_len[doc_idx]
        tf = self._tf[doc_idx]
        score = 0.0
        for term in query_terms:
            f = tf.get(term)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * length / self._avgdl) if self._avgdl else f
            score += idf * f * (self.k1 + 1) / denom
        return score

    def search(
        self, query: str, k: int, flt: MetaFilter | None = None
    ) -> list[ScoredChunk]:
        """Top-k by BM25 score. Documents scoring 0 (no term overlap at all)
        are excluded rather than padding the result with arbitrary ties."""
        if not self.chunks or k <= 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidates = range(len(self.chunks))
        if flt:
            candidates = [i for i in candidates if flt.matches(self.chunks[i].metadata)]

        scored = []
        for i in candidates:
            s = self._score(query_terms, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [
            ScoredChunk(chunk=self.chunks[i], score=float(s)) for s, i in scored[:k]
        ]

    @classmethod
    def from_store(cls, store) -> "BM25Index":
        """Build from any backend exposing `.all_chunks()` — numpy or Qdrant.

        BM25 needs the raw text of the whole corpus up front (to compute
        document frequencies), which is a different access pattern than
        `search()`'s per-query top-k, so it is built once from a full chunk
        dump rather than wired into `SearchBackend` itself.
        """
        return cls(chunks=store.all_chunks())
