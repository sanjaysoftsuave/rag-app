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


# ---------------------------------------------------------------------------
# evaluation: — the judge model and G-Eval's sampling
# ---------------------------------------------------------------------------


def test_the_evaluation_section_is_omissible(tmp_path):
    """No `evaluation:` block at all — every config predating the feature."""
    cfg = load_config(_write_single(tmp_path))
    assert cfg.evaluation.judge_model == "openai/gpt-4o"
    assert cfg.evaluation.geval_samples == 5
    assert cfg.evaluation.max_llm_calls == 200


def test_the_evaluation_section_parses(tmp_path):
    cfg = load_config(
        _write_single(
            tmp_path,
            evaluation={
                "judge_model": "anthropic/claude-sonnet-4",
                "geval_samples": 3,
                "geval_temperature": 0.7,
                "max_llm_calls": 40,
            },
        )
    )
    assert cfg.evaluation.judge_model == "anthropic/claude-sonnet-4"
    assert cfg.evaluation.geval_samples == 3
    assert cfg.evaluation.max_llm_calls == 40


def test_judge_llm_inherits_the_base_url_and_overrides_the_model(tmp_path):
    from rag_app.config import judge_llm

    cfg = load_config(_write_single(tmp_path, evaluation={"judge_model": "big/model"}))
    jl = judge_llm(cfg)
    assert jl.base_url == cfg.llm.base_url      # same account, same endpoint
    assert jl.model == "big/model"              # different model
    assert jl.model != cfg.llm.model
    assert jl.timeout_seconds == 60.0           # judges get longer than generation


def test_judge_base_url_can_point_the_judge_elsewhere(tmp_path):
    from rag_app.config import judge_llm

    cfg = load_config(
        _write_single(tmp_path, evaluation={"judge_base_url": "http://elsewhere"})
    )
    assert judge_llm(cfg).base_url == "http://elsewhere"


def test_geval_samples_must_be_at_least_one(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        load_config(_write_single(tmp_path, evaluation={"geval_samples": 0}))


def test_multi_sampling_at_temperature_zero_is_rejected(tmp_path):
    """N identical scores cost Nx and estimate no variance at all."""
    with pytest.raises(ValueError, match="same score"):
        load_config(
            _write_single(
                tmp_path, evaluation={"geval_samples": 5, "geval_temperature": 0.0}
            )
        )


def test_a_single_sample_at_temperature_zero_is_fine(tmp_path):
    cfg = load_config(
        _write_single(tmp_path, evaluation={"geval_samples": 1, "geval_temperature": 0.0})
    )
    assert cfg.evaluation.geval_samples == 1


def test_max_llm_calls_cannot_be_negative(tmp_path):
    with pytest.raises(ValueError, match="cannot be negative"):
        load_config(_write_single(tmp_path, evaluation={"max_llm_calls": -1}))


def test_an_empty_judge_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="judge_model is empty"):
        load_config(_write_single(tmp_path, evaluation={"judge_model": "  "}))


def test_a_judge_model_equal_to_the_generator_is_allowed_not_rejected(tmp_path):
    """A legitimate bad configuration. The report warns; the loader does not
    refuse, because seeing the self-preference effect is the teaching point."""
    cfg = load_config(_write_single(tmp_path, evaluation={"judge_model": "m"}))
    assert cfg.evaluation.judge_model == cfg.llm.model
