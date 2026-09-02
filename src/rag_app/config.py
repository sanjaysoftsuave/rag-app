from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Name given to the single configuration from a `chunking:` block. It becomes
# the store directory suffix (data/store/qdrant_default), so changing it
# orphans an existing store rather than breaking anything.
DEFAULT_PRESET = "default"


@dataclass(frozen=True)
class ChunkPreset:
    """How text is windowed. There is only one method — see chunking.py."""

    chunk_size: int
    overlap: int

    def describe(self) -> str:
        return f"{self.chunk_size}/{self.overlap}"


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    temperature: float
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None
    m: int = 16
    ef_construct: int = 100
    hnsw_ef: int = 128


@dataclass(frozen=True)
class RetrievalConfig:
    """Which candidate-generation strategy `ask()` uses.

    'dense'  — bi-encoder cosine search only.
    'hybrid' — dense + BM25 keyword search, fused by Reciprocal Rank Fusion.
               See hybrid.py for why RRF rather than a weighted score blend.

    `query_mode` transforms the question before retrieval (rewrite.py) and
    `mmr` re-selects the reranked candidates for coverage (mmr.py). Both are
    off by default: each is a real change to what the model reads, and neither
    is worth enabling without a gold set to show it helped.
    """

    mode: str = "dense"
    rrf_k: int = 60
    bm25_pool: int = 20
    # Query transform applied BEFORE retrieval: "off" | "rewrite" | "hyde".
    # Both non-off modes cost an LLM call per question — see rewrite.py.
    query_mode: str = "off"
    # Maximal Marginal Relevance over the reranked candidates. lambda 1.0 is
    # pure relevance (identical to off); lower trades relevance for coverage.
    mmr: bool = False
    mmr_lambda: float = 0.7


@dataclass(frozen=True)
class AppConfig:
    docs_dir: Path
    tickets_dir: Path
    store_dir: Path
    chunk_presets: dict[str, ChunkPreset]
    default_preset: str
    bi_encoder_model: str
    cross_encoder_model: str
    retrieve_k: int
    rerank_n: int
    score_threshold: float
    llm: LlmConfig
    llm_api_key: str | None
    rerank_score_scale: str = "sigmoid"
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")
    path = config_path or (ROOT / "config.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Two accepted shapes, and the rest of the app cannot tell them apart:
    #
    #   chunking: {strategy, chunk_size, overlap}   -> one config named "default"
    #   chunk_presets: {A: {...}, B: {...}}         -> several, plus default_preset
    #
    # Everything downstream iterates `sorted(cfg.chunk_presets)`, so the single
    # form is just a one-entry map. That is what makes going back to several a
    # YAML-only change — `compare`, `eval --all`, `chunks --all` and the UI's
    # preset selector all start working again with no code touched.
    raw_presets = raw.get("chunk_presets")
    if raw_presets:
        presets = {
            name: ChunkPreset(
                chunk_size=int(vals["chunk_size"]),
                overlap=int(vals["overlap"]),
            )
            for name, vals in raw_presets.items()
        }
        default_preset = str(raw["default_preset"])
        # Fail at load time with a useful message rather than later with a
        # missing store directory that looks like an ingest problem.
        if default_preset not in presets:
            raise ValueError(
                f"default_preset {default_preset!r} is not defined in chunk_presets "
                f"{sorted(presets)}"
            )
    else:
        chunking = raw.get("chunking")
        if not chunking:
            raise ValueError(
                "config.yaml needs either a 'chunking' block (one configuration) or a "
                "'chunk_presets' map plus 'default_preset' (several). Found neither."
            )
        presets = {
            DEFAULT_PRESET: ChunkPreset(
                chunk_size=int(chunking["chunk_size"]),
                overlap=int(chunking["overlap"]),
            )
        }
        default_preset = DEFAULT_PRESET

    retrieve_k = int(raw["retrieve_k"])
    rerank_n = int(raw["rerank_n"])
    if rerank_n > retrieve_k:
        raise ValueError(
            f"rerank_n ({rerank_n}) exceeds retrieve_k ({retrieve_k}); the reranker can "
            f"only ever see {retrieve_k} candidates, so the extra slots are unreachable."
        )

    scale = str(raw.get("rerank_score_scale", "sigmoid"))
    if scale not in {"sigmoid", "raw"}:
        raise ValueError(f"rerank_score_scale must be 'sigmoid' or 'raw', got {scale!r}")

    threshold = float(raw["score_threshold"])
    if scale == "sigmoid" and not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"score_threshold {threshold} is outside 0-1, but rerank_score_scale is "
            f"'sigmoid' so scores are probabilities. Did you mean 'raw'?"
        )

    r_raw = raw.get("retrieval") or {}
    retrieval_mode = str(r_raw.get("mode", "dense"))
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError(
            f"retrieval.mode must be 'dense' or 'hybrid', got {retrieval_mode!r}"
        )

    query_mode = str(r_raw.get("query_mode", "off"))
    if query_mode not in {"off", "rewrite", "hyde"}:
        raise ValueError(
            f"retrieval.query_mode must be 'off', 'rewrite' or 'hyde', got {query_mode!r}"
        )

    mmr_lambda = float(r_raw.get("mmr_lambda", 0.7))
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError(
            f"retrieval.mmr_lambda is a mix between relevance (1.0) and diversity "
            f"(0.0), so it must be between 0 and 1; got {mmr_lambda}"
        )

    q_raw = raw.get("qdrant") or {}
    llm_raw = raw["llm"]

    return AppConfig(
        # Vestigial: nothing reads docs_dir since the corpus moved to
        # tickets_dir, which holds .jsonl, .md/.txt AND .pdf together. It was
        # a required key purely because the loader asked for it unconditionally
        # — now optional, so config.yaml need not carry a dead setting.
        docs_dir=_resolve(raw.get("docs_dir", "data/docs")),
        tickets_dir=_resolve(raw.get("tickets_dir", "data/tickets")),
        store_dir=_resolve(raw.get("store_dir", "data/store")),
        chunk_presets=presets,
        default_preset=default_preset,
        bi_encoder_model=str(raw["bi_encoder_model"]),
        cross_encoder_model=str(raw["cross_encoder_model"]),
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        score_threshold=threshold,
        rerank_score_scale=scale,
        retrieval=RetrievalConfig(
            mode=retrieval_mode,
            rrf_k=int(r_raw.get("rrf_k", 60)),
            bm25_pool=int(r_raw.get("bm25_pool", 20)),
            query_mode=query_mode,
            mmr=bool(r_raw.get("mmr", False)),
            mmr_lambda=mmr_lambda,
        ),
        qdrant=QdrantConfig(
            url=q_raw.get("url") or None,
            m=int(q_raw.get("m", 16)),
            ef_construct=int(q_raw.get("ef_construct", 100)),
            hnsw_ef=int(q_raw.get("hnsw_ef", 128)),
        ),
        llm=LlmConfig(
            base_url=str(llm_raw["base_url"]),
            model=str(llm_raw["model"]),
            temperature=float(llm_raw["temperature"]),
            timeout_seconds=float(llm_raw.get("timeout_seconds", 30.0)),
        ),
        llm_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY"),
    )
