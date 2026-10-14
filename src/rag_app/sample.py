"""The bundled sample corpus: a working first answer for someone who has no
documents of their own yet.

ISOLATED STRUCTURALLY, THE SAME WAY THE RED-TEAM CORPUS IS
-----------------------------------------------------------
`sample_config()` mirrors `redteam.redteam_config()`: it redirects BOTH
`tickets_dir` and `store_dir`. Redirecting only the corpus would build the
sample index into the SAME `qdrant_<preset>/` directory as your real one and
silently replace it. A test asserts the two paths cannot coincide.

Because `gold_path`, `reqlog.log_path`, `before_after.snapshot_path` and
`reqlog.cache_path` all anchor on `cfg.tickets_dir.parent`, pointing
`tickets_dir` at `data/sample/corpus` isolates the sample's gold set, request
log, snapshots and answer cache under `data/sample/` for free — no new path
logic anywhere.

`sample_config` is IDEMPOTENT on purpose. `redteam.clean_set_dir` documents the
trap: once `tickets_dir` is redirected, `tickets_dir.parent` points somewhere
new, and naively redirecting again would resolve `data/sample/sample/corpus` —
a directory that does not exist, read as "no index, go ingest".

WHY THIS IS THE SECOND CLI INGEST
----------------------------------
Building an index is otherwise a UI action, deliberately in one place. This
exception exists for the same two reasons `redteam --build-index` does: it is
structurally incapable of touching your real corpus, and a stranger following
the guide must be able to reach a cited answer without first learning the UI.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rag_app.config import AppConfig

SAMPLE_DIRNAME = "sample"
SAMPLE_QUESTION = "How long do I have to request a refund after cancelling?"


def is_sample(cfg: AppConfig) -> bool:
    return cfg.tickets_dir.name == "corpus" and cfg.tickets_dir.parent.name == SAMPLE_DIRNAME


def sample_root(cfg: AppConfig) -> Path:
    if is_sample(cfg):
        return cfg.tickets_dir.parent
    return cfg.tickets_dir.parent / SAMPLE_DIRNAME


def sample_config(cfg: AppConfig) -> AppConfig:
    """The same app, pointed at the bundled sample corpus and a store of its own."""
    if is_sample(cfg):
        return cfg
    return replace(
        cfg,
        tickets_dir=sample_root(cfg) / "corpus",
        store_dir=cfg.store_dir / SAMPLE_DIRNAME,
    )


def build_sample_index(cfg: AppConfig, *, embedder=None, preset: str | None = None):
    from rag_app.ingest import run_ingest

    return run_ingest(preset=preset, config=sample_config(cfg), embedder=embedder)


def sample_index_exists(cfg: AppConfig, preset: str | None = None) -> bool:
    from rag_app.qdrant_store import qdrant_path_for_preset

    scfg = sample_config(cfg)
    return qdrant_path_for_preset(scfg.store_dir, preset or scfg.default_preset).exists()
