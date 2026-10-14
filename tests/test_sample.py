"""The bundled sample corpus: isolation, idempotence, and that its own gold set
is actually answerable from the documents it ships with."""

from __future__ import annotations

from conftest import make_config

from rag_app.before_after import snapshot_dir
from rag_app.config import ROOT, load_config
from rag_app.evaluate import gold_path, load_gold
from rag_app.qdrant_store import qdrant_path_for_preset
from rag_app.reqlog import cache_path, log_path
from rag_app.sample import is_sample, sample_config, sample_root


def test_the_sample_store_never_coincides_with_the_real_one(tmp_path):
    """Redirecting only the corpus would build the sample INTO your real
    index directory and silently replace it — same guard as the red team's."""
    cfg = make_config(tmp_path)
    scfg = sample_config(cfg)
    assert scfg.store_dir != cfg.store_dir
    assert qdrant_path_for_preset(scfg.store_dir, "C") != qdrant_path_for_preset(cfg.store_dir, "C")
    assert scfg.tickets_dir != cfg.tickets_dir


def test_sample_config_is_idempotent(tmp_path):
    """The redteam.clean_set_dir trap: redirecting an already-redirected
    config must not resolve data/sample/sample/corpus."""
    scfg = sample_config(make_config(tmp_path))
    assert sample_config(scfg) is scfg
    assert is_sample(scfg)
    assert sample_root(scfg) == scfg.tickets_dir.parent


def test_every_derived_path_isolates_under_the_sample_root(tmp_path):
    """gold, log, snapshots and cache all anchor on tickets_dir.parent, so
    they follow the redirect for free — this pins that they actually do."""
    scfg = sample_config(make_config(tmp_path))
    root = sample_root(scfg)
    for path in (gold_path(scfg), log_path(scfg), snapshot_dir(scfg), cache_path(scfg)):
        assert root in path.parents or path.parent == root


def test_every_shipped_sample_document_says_it_is_synthetic():
    corpus = ROOT / "data" / "sample" / "corpus"
    docs = sorted(corpus.glob("*.md"))
    assert docs, "the sample corpus must ship with the repo"
    for doc in docs:
        assert "Synthetic sample document" in doc.read_text(encoding="utf-8"), doc.name


def test_the_shipped_sample_gold_set_is_answerable_from_the_shipped_corpus():
    """A gold answer the corpus does not contain would grade every correct
    run a failure. Checked against the files, not assumed."""
    scfg = sample_config(load_config())
    corpus_text = " ".join(
        p.read_text(encoding="utf-8").lower() for p in scfg.tickets_dir.glob("*.md")
    )
    gold = load_gold(scfg)
    assert len(gold) >= 8
    assert any(not g.answerable for g in gold), "needs refusal cases to test the gate"
    for g in gold:
        if g.answerable:
            for snippet in g.snippets:
                assert snippet.lower() in corpus_text, f"{g.question!r}: {snippet!r} not in corpus"
