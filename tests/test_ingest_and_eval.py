"""Ingest and evaluation, end to end, with no model download.

`run_ingest` takes an injectable embedder for exactly this reason — without it
the whole load → chunk → embed → persist path is untestable offline.
"""

from __future__ import annotations

import json

import pytest

from rag_app.evaluate import GOLD, GoldQuestion, evaluate
from rag_app.ingest import run_ingest
from rag_app.pipeline import open_store
from rag_app.store import VectorStore, store_path_for_preset
from rag_app.tickets import load_all, load_tickets, render_ticket, ticket_metadata

from conftest import FakeEmbedder, FakeReranker, make_config, make_ticket

TICKET_LINE = {
    "ticket_id": "TIC-001",
    "subject": "Free plan rate limit",
    "product": "API",
    "category": "rate-limit",
    "status": "resolved",
    "priority": "high",
    "channel": "email",
    "created_at": "2026-01-01",
    "customer_tier": "free",
    "tags": ["429"],
    "conversation": [
        {"role": "customer", "text": "Why am I getting 429?"},
        {"role": "agent", "text": "The Free plan allows 60 requests per minute."},
    ],
    "resolution": "Explained the 60 rpm Free limit.",
}


def _write_tickets(cfg, rows=None):
    cfg.tickets_dir.mkdir(parents=True, exist_ok=True)
    rows = rows or [TICKET_LINE]
    (cfg.tickets_dir / "tickets.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def test_load_tickets_rejects_bad_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ticket_id": "X"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid ticket"):
        load_tickets(path)


def test_load_tickets_skips_blank_lines(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(json.dumps(TICKET_LINE) + "\n\n", encoding="utf-8")
    assert len(load_tickets(path)) == 1


def test_rendered_ticket_carries_its_own_identity():
    ticket = make_ticket("TIC-042", "Billing question", product="Billing")
    text = render_ticket(ticket)
    assert "TIC-042" in text
    assert "Billing" in text
    assert ticket_metadata(ticket)["product"] == "Billing"


def test_ingest_persists_metadata_and_provenance(tmp_path):
    cfg = make_config(tmp_path)
    _write_tickets(cfg)

    report = run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    assert report.n_tickets == 1
    assert report.n_chunks == 1
    assert report.bleeding_chunks == 0

    store = VectorStore.load(store_path_for_preset(cfg.store_dir, "C"))
    assert store.meta is not None
    assert store.meta.embedding_model == "fake"
    assert store.meta.strategy == "ticket"
    assert store.chunks[0].metadata["customer_tier"] == "free"


def test_ingest_flat_preset_reports_boundary_bleed(tmp_path):
    cfg = make_config(tmp_path)
    _write_tickets(cfg, [dict(TICKET_LINE, ticket_id=f"TIC-{i:03d}") for i in range(1, 6)])
    report = run_ingest(preset="A", config=cfg, embedder=FakeEmbedder())
    assert report.strategy == "flat"
    assert report.bleeding_chunks > 0
    assert "boundary bleed" in report.describe()


def test_ingest_rejects_unknown_preset(tmp_path):
    cfg = make_config(tmp_path)
    _write_tickets(cfg)
    with pytest.raises(KeyError):
        run_ingest(preset="ZZZ", config=cfg, embedder=FakeEmbedder())


def test_open_store_refuses_a_model_swap(tmp_path):
    cfg = make_config(tmp_path)
    _write_tickets(cfg)
    run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())

    swapped = make_config(tmp_path, bi_encoder_model="a-different-model")
    with pytest.raises(ValueError, match="not comparable"):
        open_store(swapped, "C")


def test_open_store_names_the_ingest_command_when_missing(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="ingest --preset"):
        open_store(cfg, "C")


def test_eval_scores_hits_and_refusals(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    _write_tickets(cfg)
    run_ingest(preset="C", config=cfg, embedder=FakeEmbedder())
    store = open_store(cfg, "C")

    gold = [
        GoldQuestion("What is the Free plan rate limit?", "TIC-001", "60"),
        GoldQuestion("Are you HIPAA compliant?", None),
    ]
    # High score for the answerable question, low for the unanswerable one.
    reranker = FakeReranker(score=0.9, per_text={})
    report = evaluate(store, cfg, "C", embedder=FakeEmbedder(), reranker=reranker, gold=gold)

    assert report.hit_at_k == 1.0
    assert report.top1_rerank == 1.0
    # Every score is 0.9, so the unanswerable question wrongly passes the gate —
    # which is precisely what refusal_accuracy is meant to expose.
    assert report.refusal_accuracy == 0.0


def test_gold_set_is_well_formed():
    answerable = [g for g in GOLD if g.answerable]
    unanswerable = [g for g in GOLD if not g.answerable]
    assert len(answerable) >= 10
    assert len(unanswerable) >= 3, "need negatives or the gate is never tested"
    assert len({g.question for g in GOLD}) == len(GOLD), "duplicate gold questions"
    for g in unanswerable:
        assert g.note, "every negative should say why it is absent from the corpus"


def test_shipped_corpus_matches_the_gold_set():
    """Guards against a gold question pointing at a ticket that no longer exists."""
    from rag_app.config import load_config

    cfg = load_config()
    tickets = load_all(cfg.tickets_dir)
    known = {t.ticket_id for t in tickets}
    for g in GOLD:
        if g.ticket_id:
            assert g.ticket_id in known, f"gold question references missing {g.ticket_id}"
