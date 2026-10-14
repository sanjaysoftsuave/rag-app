"""`python -m rag_app doctor`: what a stranger needs to know before the first run.

Free, offline, and fast on purpose. It loads no model (packages are checked
with `find_spec`, which never imports torch), makes no network call, never
validates the API key against a server (that would spend money and leak the
key into a request a doctor has no business making) and never PRINTS the key.
It reads the index's provenance sidecar without opening the database, so it
takes no file lock and can run while the UI is up.

Every line is PASS / WARN / FAIL plus the exact command that fixes it. A FAIL
means nothing will work; a WARN means one path (the UI, real answers, your own
documents, measurement) is unavailable and the rest still is.

The logic is `run_checks(cfg) -> list[Check]`, pure and tested offline;
`cmd_doctor` only prints it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.util import find_spec

from rag_app.config import AppConfig

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

CORE_PACKAGES = (
    "sentence_transformers", "torch", "qdrant_client", "openai", "yaml", "dotenv", "pypdf",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _python(version=None) -> Check:
    v = version or sys.version_info
    if (v[0], v[1]) >= (3, 11):
        return Check("python", PASS, f"Python {v[0]}.{v[1]}")
    return Check("python", FAIL, f"Python {v[0]}.{v[1]} — 3.11 or newer is required",
                 "Install Python 3.11+ and recreate the virtual environment.")


def _packages(spec=find_spec) -> list[Check]:
    missing = [p for p in CORE_PACKAGES if spec(p) is None]
    out = [
        Check("packages", FAIL, f"missing: {', '.join(missing)}",
              'python -m pip install -e ".[ui]"')
        if missing else Check("packages", PASS, "core packages installed")
    ]
    if spec("streamlit") is None:
        out.append(Check("ui", WARN, "streamlit not installed — the browser UI is unavailable",
                         'python -m pip install -e ".[ui]"'))
    else:
        out.append(Check("ui", PASS, "streamlit installed"))
    return out


def _key(cfg: AppConfig) -> Check:
    if cfg.llm_api_key:
        # Presence only. Whether the key is VALID (and has credit) can only be
        # learned by spending a call, which is the first `quickstart`'s job.
        return Check("api key", PASS, "OPENROUTER_API_KEY is set (not validated — no call made)")
    return Check(
        "api key", WARN,
        "OPENROUTER_API_KEY is not set (or is still the .env.example placeholder) — "
        "retrieval works, answers will be skipped",
        "Copy .env.example to .env and set OPENROUTER_API_KEY (https://openrouter.ai/keys).",
    )


def _index(cfg: AppConfig, label: str, fix: str) -> Check:
    from rag_app.qdrant_store import qdrant_path_for_preset, read_provenance

    path = qdrant_path_for_preset(cfg.store_dir, cfg.default_preset)
    if not path.exists():
        return Check(label, WARN, f"no index at {path}", fix)
    meta = read_provenance(path)
    if meta is None:
        return Check(label, WARN, f"{path} exists but has no provenance — rebuild it", fix)
    if meta.embedding_model != cfg.bi_encoder_model:
        return Check(
            label, FAIL,
            f"built with {meta.embedding_model}, config asks for {cfg.bi_encoder_model} — "
            f"vectors from different models are not comparable",
            fix,
        )
    return Check(label, PASS, f"{meta.n_chunks} chunks, {meta.embedding_model}")


def _corpus(cfg: AppConfig, label: str, fix: str) -> Check:
    folder = cfg.tickets_dir
    docs = [p for p in folder.glob("*") if p.suffix.lower() in {".md", ".txt", ".pdf", ".markdown"}]
    if not docs:
        return Check(label, WARN, f"no documents in {folder}", fix)
    return Check(label, PASS, f"{len(docs)} document(s) in {folder.name}/")


def _gold(cfg: AppConfig, label: str, fix: str) -> Check:
    from rag_app.evaluate import gold_path, load_gold

    path = gold_path(cfg)
    if not path.exists():
        return Check(label, WARN, f"no gold set at {path} — eval/debug have nothing to score", fix)
    try:
        gold = load_gold(cfg, path)
    except ValueError as exc:
        return Check(label, FAIL, f"{path.name} does not parse: {exc}", fix)
    answerable = sum(1 for g in gold if g.answerable)
    return Check(label, PASS, f"{len(gold)} questions ({answerable} answerable)")


def run_checks(cfg: AppConfig) -> list[Check]:
    from rag_app.sample import sample_config

    scfg = sample_config(cfg)
    checks = [_python(), *_packages(), _key(cfg)]
    checks.append(_corpus(scfg, "sample corpus", "git checkout data/sample  (it ships with the repo)"))
    checks.append(_index(scfg, "sample index", "python -m rag_app quickstart"))
    checks.append(_corpus(cfg, "your corpus", "python -m rag_app ui  → Add documents"))
    checks.append(_index(cfg, "your index", "python -m rag_app ui  → Add documents → Run ingest"))
    checks.append(_gold(cfg, "your gold set",
                        "cp gold.example.yaml data/gold.yaml, then write questions for YOUR documents"))
    return checks


def next_step(checks: list[Check]) -> str:
    """The single most useful thing to do next, in order of what blocks most."""
    by_name = {c.name: c for c in checks}
    for c in checks:
        if c.status == FAIL:
            return c.fix
    if by_name["sample index"].status != PASS:
        return "python -m rag_app quickstart"
    if by_name["api key"].status != PASS:
        return by_name["api key"].fix
    if by_name["your index"].status != PASS:
        return "python -m rag_app ui   — add your own documents and build the index"
    return 'python -m rag_app ask "your question"'


def describe(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"  {c.status}  {c.name:<{width}}  {c.detail}")
        if c.fix and c.status != PASS:
            lines.append(f"  {'':4}  {'':<{width}}  → {c.fix}")
    lines.append("")
    lines.append(f"Next: {next_step(checks)}")
    return "\n".join(lines)
