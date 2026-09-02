"""The introspection layer behind the UI.

These matter because the browser view is a debugging surface: if it reports
that the reranker demoted something it did not, or that the LLM was called
when it was not, it is worse than having no view at all.
"""

from __future__ import annotations

import pytest

from rag_app.chunking import Chunk
from rag_app.explain import (
    audit_citations,
    build_candidate_table,
    config_rows,
    corpus_mix,
    explain_gate,
    logit,
    query_encoding,
    retrieval_score_label,
)
from rag_app.generate import Answer
from rag_app.pipeline import ask
from rag_app.rerank import rerank, rerank_all, sigmoid, stub_scorer
from rag_app.store import ScoredChunk, StoreMeta

from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store


def sc(source: str, score: float, text: str = "text", **meta) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=f"{source}::0", source=source, text=text, metadata=meta),
        score=score,
    )


def answer_with(retrieved, reranked_all, rerank_n, **kwargs) -> Answer:
    reranked = reranked_all[:rerank_n]
    base = dict(
        text="Some answer.",
        sources=[],
        best_score=reranked[0].score if reranked else float("-inf"),
        used_llm=True,
        retrieved=retrieved,
        reranked=reranked,
        reranked_all=reranked_all,
        gate="answered",
        meta={"threshold": 0.2, "retrieval_mode": "dense", "cited": True},
    )
    base.update(kwargs)
    return Answer(**base)


# --- logit -----------------------------------------------------------------


@pytest.mark.parametrize("x", [-8.0, -1.5, 0.0, 1.5, 8.0])
def test_logit_inverts_sigmoid(x):
    assert logit(sigmoid(x)) == pytest.approx(x, abs=1e-6)


def test_logit_returns_none_at_the_saturated_ends():
    """Any finite number here would be invented precision."""
    assert logit(0.0) is None
    assert logit(1.0) is None


# --- candidate table -------------------------------------------------------


def test_table_joins_both_rankings_and_reports_movement():
    retrieved = [sc("A", 0.9), sc("B", 0.8), sc("C", 0.7)]
    reranked_all = [sc("C", 0.99), sc("A", 0.5), sc("B", 0.1)]
    rows = build_candidate_table(answer_with(retrieved, reranked_all, rerank_n=2))

    by_source = {r.source: r for r in rows}
    assert by_source["C"].dense_rank == 3 and by_source["C"].rerank_rank == 1
    assert by_source["C"].rank_delta == 2  # promoted two places
    assert by_source["C"].movement == "▲ 2"
    assert by_source["A"].rank_delta == -1
    assert by_source["A"].movement == "▼ 1"
    assert by_source["B"].movement == "▼ 1"


def test_cut_candidates_are_present_and_marked_out_of_context():
    """The whole reason reranked_all exists — seeing what the LLM never got."""
    retrieved = [sc("A", 0.9), sc("B", 0.8), sc("C", 0.7)]
    reranked_all = [sc("A", 0.99), sc("B", 0.5), sc("C", 0.1)]
    rows = build_candidate_table(answer_with(retrieved, reranked_all, rerank_n=1))

    assert [r.source for r in rows] == ["A", "B", "C"]
    assert [r.in_context for r in rows] == [True, False, False]


def test_a_retrieved_chunk_the_reranker_never_scored_still_appears():
    retrieved = [sc("A", 0.9), sc("ghost", 0.4)]
    reranked_all = [sc("A", 0.99)]
    rows = build_candidate_table(answer_with(retrieved, reranked_all, rerank_n=1))

    ghost = next(r for r in rows if r.source == "ghost")
    assert ghost.rerank_rank is None
    assert ghost.rank_delta is None
    assert ghost.movement == "—"
    assert ghost.in_context is False


def test_table_falls_back_to_reranked_when_reranked_all_is_absent():
    """An Answer rebuilt from an older trace must still render."""
    retrieved = [sc("A", 0.9)]
    answer = Answer(
        text="x", sources=[], best_score=0.9, used_llm=True,
        retrieved=retrieved, reranked=[sc("A", 0.95)], gate="answered", meta={},
    )
    rows = build_candidate_table(answer)
    assert [r.source for r in rows] == ["A"]
    assert rows[0].in_context is True


def test_corpus_mix_counts_source_kinds():
    retrieved = [
        sc("TIC-001", 0.9, source_type="ticket"),
        sc("policy.md", 0.8, source_type="doc"),
        sc("handbook.pdf", 0.7, source_type="pdf"),
        sc("manual.pdf", 0.6, source_type="pdf"),
    ]
    rows = build_candidate_table(answer_with(retrieved, list(retrieved), rerank_n=4))
    assert corpus_mix(rows) == {"ticket": 1, "doc": 1, "pdf": 2}


def test_pdf_rows_are_labelled_as_pdfs():
    retrieved = [sc("h.pdf", 0.9, source_type="pdf")]
    rows = build_candidate_table(answer_with(retrieved, list(retrieved), rerank_n=1))
    assert rows[0].kind == "pdf"
    assert rows[0].detail == "PDF"


def test_retrieval_score_label_tracks_the_mode():
    """An RRF score is not a cosine and must not be labelled as one."""
    dense = answer_with([sc("A", 0.9)], [sc("A", 0.9)], 1)
    assert retrieval_score_label(dense) == "cosine"
    hybrid = answer_with(
        [sc("A", 0.03)], [sc("A", 0.9)], 1,
        meta={"threshold": 0.2, "retrieval_mode": "hybrid"},
    )
    assert retrieval_score_label(hybrid) == "rrf"


# --- gate ------------------------------------------------------------------


def test_below_threshold_is_reported_as_never_reaching_the_llm(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    answer = answer_with(
        [sc("A", 0.9)], [sc("A", 0.1)], 1, gate="below-threshold", used_llm=False, best_score=0.1
    )
    gate = explain_gate(answer, cfg)
    assert gate.llm_called is False
    assert gate.passed is False
    assert gate.margin == pytest.approx(-0.4)
    assert "never called" in gate.detail


def test_model_refused_is_distinguished_from_the_score_gate(tmp_path):
    """Identical to a user, opposite fixes — the distinction must survive."""
    cfg = make_config(tmp_path, score_threshold=0.5)
    answer = answer_with(
        [sc("A", 0.9)], [sc("A", 0.9)], 1, gate="model-refused", used_llm=True, best_score=0.9
    )
    gate = explain_gate(answer, cfg)
    assert gate.llm_called is True
    assert gate.passed is False
    assert gate.margin == pytest.approx(0.4)
    assert "the LLM was called" in gate.detail


def test_no_candidates_has_no_score_to_report(tmp_path):
    cfg = make_config(tmp_path)
    answer = answer_with([], [], 0, gate="no-candidates", used_llm=False, best_score=float("-inf"))
    gate = explain_gate(answer, cfg)
    assert gate.best_score is None
    assert gate.margin is None
    assert "nothing" in gate.headline.lower() or "nothing" in gate.detail.lower()


def test_answered_reports_the_passing_margin(tmp_path):
    cfg = make_config(tmp_path, score_threshold=0.5)
    answer = answer_with([sc("A", 0.9)], [sc("A", 0.95)], 1, best_score=0.95)
    gate = explain_gate(answer, cfg)
    assert gate.passed is True
    assert gate.margin == pytest.approx(0.45)


# --- citations -------------------------------------------------------------


def test_audit_separates_grounded_from_invented():
    answer = answer_with(
        [sc("A", 0.9)], [sc("A", 0.9)], 1,
        sources=["TIC-001"], hallucinated_citations=["TIC-999"],
    )
    audit = audit_citations(answer)
    assert audit.grounded == ["TIC-001"]
    assert audit.invented == ["TIC-999"]
    assert audit.clean is False


def test_audit_flags_the_quiet_no_citation_fallback():
    """Sources present, but the model never claimed them."""
    answer = answer_with(
        [sc("A", 0.9)], [sc("A", 0.9)], 1,
        sources=["TIC-001"],
        meta={"threshold": 0.2, "retrieval_mode": "dense", "cited": False},
    )
    audit = audit_citations(answer)
    assert audit.fell_back is True
    assert audit.clean is False


def test_a_refusal_has_no_grounded_citations():
    answer = answer_with(
        [sc("A", 0.9)], [sc("A", 0.9)], 1, gate="model-refused", refused=True, sources=[]
    )
    audit = audit_citations(answer)
    assert audit.grounded == []
    assert audit.clean is True


# --- config / encoding -----------------------------------------------------


def test_query_encoding_shows_the_prefix_an_asymmetric_model_needs(tmp_path):
    cfg = make_config(tmp_path, bi_encoder_model="intfloat/e5-small-v2")
    enc = query_encoding(cfg, "how long do refunds take")
    assert enc["family"] == "asymmetric"
    assert enc["query_prefix"] == "query: "
    assert enc["encoded"] == "query: how long do refunds take"


def test_query_encoding_reports_a_symmetric_model_honestly(tmp_path):
    cfg = make_config(tmp_path, bi_encoder_model="sentence-transformers/all-MiniLM-L6-v2")
    enc = query_encoding(cfg, "hello")
    assert enc["family"] == "symmetric"
    assert enc["query_prefix"] == "(none)"
    assert enc["encoded"] == "hello"


def test_config_rows_cover_every_setting_that_shaped_the_run(tmp_path):
    cfg = make_config(tmp_path)
    labels = {label for label, _ in config_rows(cfg, "C")}
    assert {"preset", "chunking", "bi-encoder", "cross-encoder", "K → N",
            "score_threshold", "score scale", "retrieval mode"} <= labels


def test_config_rows_report_the_chunking_that_actually_built_the_index(tmp_path):
    """Chunking is an ingest-time setting, so the config value may name
    something that had no part in producing the retrieved chunks."""
    cfg = make_config(tmp_path)  # preset C is 2000/200
    indexed = StoreMeta("fake", 8, 500, 50, n_chunks=12)
    rows = dict(config_rows(cfg, "C", indexed))
    assert "chunking (as indexed)" in rows
    value = rows["chunking (as indexed)"]
    assert "500 chars, 50 overlap" in value
    assert "needs re-ingest" in value


def test_config_rows_stay_quiet_when_index_and_settings_agree(tmp_path):
    cfg = make_config(tmp_path)
    preset = cfg.chunk_presets["C"]
    indexed = StoreMeta("fake", 8, preset.chunk_size, preset.overlap, 12)
    value = dict(config_rows(cfg, "C", indexed))["chunking (as indexed)"]
    assert "needs re-ingest" not in value


# --- rerank_all ------------------------------------------------------------


def test_rerank_all_returns_every_candidate_sorted():
    candidates = [sc("A", 0.1, "alpha"), sc("B", 0.1, "beta"), sc("C", 0.1, "gamma")]
    scorer = stub_scorer({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
    out = rerank_all("q", candidates, scorer)
    assert [item.chunk.source for item in out] == ["B", "C", "A"]


def test_rerank_is_exactly_rerank_all_truncated():
    candidates = [sc("A", 0.1, "alpha"), sc("B", 0.1, "beta"), sc("C", 0.1, "gamma")]
    scorer = stub_scorer({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
    assert rerank("q", candidates, 2, scorer) == rerank_all("q", candidates, scorer)[:2]


def test_ask_populates_reranked_all_with_the_cut_candidates(tmp_path):
    """The UI's 'what got cut' panel is only honest if ask() actually fills this."""
    embedder = FakeEmbedder()
    chunks = [
        Chunk(chunk_id=f"TIC-00{i}::0", source=f"TIC-00{i}", text=f"body {i}", metadata={})
        for i in range(1, 5)
    ]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = make_qdrant_store(tmp_path, chunks, vectors)
    cfg = make_config(tmp_path, retrieve_k=4, rerank_n=2, score_threshold=0.0)

    answer = ask(
        "anything",
        preset="C",
        config=cfg,
        store=store,
        embedder=embedder,
        reranker=FakeReranker(0.5),
        generate_fn=lambda q, c, k: "Answer.",
    )
    assert len(answer.reranked) == 2
    assert len(answer.reranked_all) == 4
    assert answer.reranked == answer.reranked_all[:2]
    store.close()


def test_reranked_all_is_populated_even_when_the_gate_refuses(tmp_path):
    embedder = FakeEmbedder()
    chunks = [
        Chunk(chunk_id=f"TIC-00{i}::0", source=f"TIC-00{i}", text=f"body {i}", metadata={})
        for i in range(1, 4)
    ]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = make_qdrant_store(tmp_path, chunks, vectors)
    cfg = make_config(tmp_path, retrieve_k=3, rerank_n=1, score_threshold=0.9)

    answer = ask(
        "anything",
        preset="C",
        config=cfg,
        store=store,
        embedder=embedder,
        reranker=FakeReranker(0.1),
        generate_fn=lambda q, c, k: "should never run",
    )
    assert answer.gate == "below-threshold"
    assert answer.used_llm is False
    assert len(answer.reranked_all) == 3, "the refusal path must still expose the evidence"
    store.close()
