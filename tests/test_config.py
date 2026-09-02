import pytest
import yaml

from rag_app.config import DEFAULT_PRESET, load_config


def test_shipped_config_loads_one_chunking_configuration():
    cfg = load_config()
    assert cfg.retrieve_k == 10
    assert cfg.rerank_n == 3
    assert len(cfg.chunk_presets) == 1
    assert cfg.default_preset == DEFAULT_PRESET
    assert cfg.chunk_presets[DEFAULT_PRESET].chunk_size == 500


def test_default_preset_exists():
    cfg = load_config()
    assert cfg.default_preset in cfg.chunk_presets


# --- the two accepted config shapes ----------------------------------------


def _write_single(tmp_path, **overrides):
    raw = {
        "tickets_dir": "data/tickets",
        "store_dir": "data/store",
        "chunking": {"chunk_size": 2000, "overlap": 200},
        "bi_encoder_model": "m",
        "cross_encoder_model": "ce",
        "retrieve_k": 10,
        "rerank_n": 3,
        "score_threshold": 0.5,
        "rerank_score_scale": "sigmoid",
        "llm": {"base_url": "http://x", "model": "m", "temperature": 0.0},
    }
    raw.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_chunking_block_becomes_a_one_entry_preset_map(tmp_path):
    """Downstream code only ever iterates chunk_presets, so the single form has
    to be indistinguishable from a one-preset map — that is what makes going
    back to several configurations a YAML-only change."""
    cfg = load_config(_write_single(tmp_path))
    assert list(cfg.chunk_presets) == [DEFAULT_PRESET]
    assert cfg.default_preset == DEFAULT_PRESET
    assert cfg.chunk_presets[DEFAULT_PRESET].chunk_size == 2000
    assert cfg.chunk_presets[DEFAULT_PRESET].overlap == 200


def test_chunk_presets_still_work_and_win_over_chunking(tmp_path):
    cfg = load_config(
        _write_single(
            tmp_path,
            chunk_presets={
                "A": {"chunk_size": 500, "overlap": 50},
                "C": {"chunk_size": 2000, "overlap": 200},
            },
            default_preset="C",
        )
    )
    assert sorted(cfg.chunk_presets) == ["A", "C"]
    assert cfg.default_preset == "C"
    assert cfg.chunk_presets["A"].chunk_size == 500


def test_a_config_with_neither_shape_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    raw = yaml.safe_load(_write_single(tmp_path).read_text(encoding="utf-8"))
    raw.pop("chunking")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="either a 'chunking' block"):
        load_config(path)


def test_docs_dir_is_optional(tmp_path):
    """It is vestigial; requiring it meant config.yaml had to carry a dead key."""
    cfg = load_config(_write_single(tmp_path))
    assert cfg.docs_dir.name == "docs"


def _write(tmp_path, **overrides):
    raw = {
        "docs_dir": "data/docs",
        "tickets_dir": "data/tickets",
        "store_dir": "data/store",
        "chunk_presets": {"A": {"chunk_size": 500, "overlap": 50}},
        "default_preset": "A",
        "bi_encoder_model": "m",
        "cross_encoder_model": "ce",
        "retrieve_k": 10,
        "rerank_n": 3,
        "score_threshold": 0.5,
        "rerank_score_scale": "sigmoid",
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


# --- the active embedder's declared limits ---------------------------------


def test_active_bi_encoder_is_in_the_registry_with_a_measured_limit():
    """An unregistered model gets `max_tokens=0`, so the UI silently loses its
    chunk_size ceiling warning. Keep the configured model registered."""
    from rag_app.embed import MODEL_REGISTRY, spec_for

    cfg = load_config()
    assert cfg.bi_encoder_model in MODEL_REGISTRY
    spec = spec_for(cfg.bi_encoder_model)
    assert spec.max_tokens > 0
    assert spec.max_chars == spec.max_tokens * 4


def test_chunk_size_stays_under_the_embedders_ceiling():
    """Above it, the tail of every chunk is dropped before embedding — text in
    the index that can never be retrieved."""
    from rag_app.embed import spec_for

    cfg = load_config()
    spec = spec_for(cfg.bi_encoder_model)
    for name, preset in cfg.chunk_presets.items():
        assert preset.chunk_size <= spec.max_chars, (
            f"preset {name}: chunk_size {preset.chunk_size} exceeds "
            f"{spec.name}'s ~{spec.max_chars}-character ceiling"
        )
