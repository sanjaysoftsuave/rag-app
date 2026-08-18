from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChunkPreset:
    chunk_size: int
    overlap: int
    strategy: str = "flat"

    def describe(self) -> str:
        return f"{self.strategy}/{self.chunk_size}/{self.overlap}"


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

    'dense'  — bi-encoder cosine search only (the week-1 baseline).
    'hybrid' — dense + BM25 keyword search, fused by Reciprocal Rank Fusion.
               See hybrid.py for why RRF rather than a weighted score blend.
    """

    mode: str = "dense"
    rrf_k: int = 60
    bm25_pool: int = 20


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
    backend: str = "numpy"
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")
    path = config_path or (ROOT / "config.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    presets = {
        name: ChunkPreset(
            chunk_size=int(vals["chunk_size"]),
            overlap=int(vals["overlap"]),
            strategy=str(vals.get("strategy", "flat")),
        )
        for name, vals in raw["chunk_presets"].items()
    }

    default_preset = str(raw["default_preset"])
    # Fail at load time with a useful message rather than later with a missing
    # store directory that looks like an ingest problem.
    if default_preset not in presets:
        raise ValueError(
            f"default_preset {default_preset!r} is not defined in chunk_presets "
            f"{sorted(presets)}"
        )

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

    backend = str(raw.get("backend", "numpy"))
    if backend not in {"numpy", "qdrant"}:
        raise ValueError(f"backend must be 'numpy' or 'qdrant', got {backend!r}")

    r_raw = raw.get("retrieval") or {}
    retrieval_mode = str(r_raw.get("mode", "dense"))
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError(
            f"retrieval.mode must be 'dense' or 'hybrid', got {retrieval_mode!r}"
        )

    q_raw = raw.get("qdrant") or {}
    llm_raw = raw["llm"]

    return AppConfig(
        docs_dir=_resolve(raw["docs_dir"]),
        tickets_dir=_resolve(raw.get("tickets_dir", "data/tickets")),
        store_dir=_resolve(raw["store_dir"]),
        chunk_presets=presets,
        default_preset=default_preset,
        bi_encoder_model=str(raw["bi_encoder_model"]),
        cross_encoder_model=str(raw["cross_encoder_model"]),
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        score_threshold=threshold,
        rerank_score_scale=scale,
        backend=backend,
        retrieval=RetrievalConfig(
            mode=retrieval_mode,
            rrf_k=int(r_raw.get("rrf_k", 60)),
            bm25_pool=int(r_raw.get("bm25_pool", 20)),
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
