import pytest
import yaml

from rag_app.config import load_config


def test_load_config_has_presets_and_strategies():
    cfg = load_config()
    assert {"A", "B", "C", "D"} <= set(cfg.chunk_presets)
    assert cfg.retrieve_k == 10
    assert cfg.rerank_n == 3
    assert cfg.chunk_presets["A"].chunk_size == 500
    assert cfg.chunk_presets["B"].chunk_size == 1000
    # A and B differ only in size, so the size effect is isolated.
    assert cfg.chunk_presets["A"].strategy == cfg.chunk_presets["B"].strategy == "flat"
    assert cfg.chunk_presets["C"].strategy == "ticket"
    assert cfg.chunk_presets["D"].strategy == "section"


def test_default_preset_exists():
    cfg = load_config()
    assert cfg.default_preset in cfg.chunk_presets


def _write(tmp_path, **overrides):
    raw = {
        "docs_dir": "data/docs",
        "tickets_dir": "data/tickets",
        "store_dir": "data/store",
        "chunk_presets": {"A": {"chunk_size": 500, "overlap": 50, "strategy": "flat"}},
        "default_preset": "A",
        "bi_encoder_model": "m",
        "cross_encoder_model": "ce",
        "retrieve_k": 10,
        "rerank_n": 3,
        "score_threshold": 0.5,
        "rerank_score_scale": "sigmoid",
        "backend": "numpy",
        "llm": {"base_url": "http://x", "model": "m", "temperature": 0.0},
    }
    raw.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_rejects_unknown_default_preset(tmp_path):
    with pytest.raises(ValueError, match="not defined in chunk_presets"):
        load_config(_write(tmp_path, default_preset="Z"))


def test_rejects_rerank_n_larger_than_retrieve_k(tmp_path):
    with pytest.raises(ValueError, match="unreachable"):
        load_config(_write(tmp_path, retrieve_k=3, rerank_n=10))


def test_rejects_threshold_outside_sigmoid_range(tmp_path):
    """A threshold of 3.0 with sigmoid scaling would refuse everything, silently."""
    with pytest.raises(ValueError, match="outside 0-1"):
        load_config(_write(tmp_path, score_threshold=3.0, rerank_score_scale="sigmoid"))


def test_raw_scale_allows_logit_thresholds(tmp_path):
    cfg = load_config(_write(tmp_path, score_threshold=3.0, rerank_score_scale="raw"))
    assert cfg.score_threshold == 3.0


def test_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="backend must be"):
        load_config(_write(tmp_path, backend="pinecone"))


def test_retrieval_mode_defaults_to_dense_when_omitted(tmp_path):
    """No `retrieval:` section at all in the yaml — the common case for every
    config predating this feature — must not break loading."""
    cfg = load_config(_write(tmp_path))
    assert cfg.retrieval.mode == "dense"
    assert cfg.retrieval.rrf_k == 60


def test_retrieval_hybrid_mode_and_params_parse(tmp_path):
    cfg = load_config(_write(
        tmp_path, retrieval={"mode": "hybrid", "rrf_k": 30, "bm25_pool": 15}
    ))
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.rrf_k == 30
    assert cfg.retrieval.bm25_pool == 15


def test_rejects_unknown_retrieval_mode(tmp_path):
    with pytest.raises(ValueError, match="retrieval.mode must be"):
        load_config(_write(tmp_path, retrieval={"mode": "reranked-only"}))


def test_default_config_yaml_loads_with_retrieval_section():
    cfg = load_config()
    assert cfg.retrieval.mode in {"dense", "hybrid"}
