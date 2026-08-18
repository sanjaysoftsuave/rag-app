from __future__ import annotations

import numpy as np

from rag_app.filters import MetaFilter
from rag_app.store import ScoredChunk, SearchBackend


def retrieve(
    store: SearchBackend,
    query_vec: np.ndarray,
    k: int,
    flt: MetaFilter | None = None,
) -> list[ScoredChunk]:
    """Dense top-K retrieval, optionally restricted by metadata.

    Named as its own stage even though it delegates: it is the seam where the
    backend (numpy / Qdrant) becomes interchangeable, and where filtering
    enters the pipeline.
    """
    return store.search(query_vec, k=k, flt=flt)
