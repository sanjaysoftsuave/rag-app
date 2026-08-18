from rag_app.bm25 import BM25Index, tokenize
from rag_app.chunking import Chunk
from rag_app.filters import MetaFilter


def _chunks():
    # Ticket id inside the TEXT, mirroring tickets.render_ticket()'s
    # "Ticket TIC-1001: ..." header — BM25 indexes chunk.text, not chunk.source.
    return [
        Chunk("a::0", "TIC-1001",
              "Ticket TIC-1001: Free plan is capped at 60 requests per minute per API key.",
              {"customer_tier": "free"}),
        Chunk("b::0", "TIC-1002",
              "Ticket TIC-1002: Pro plan allows 600 requests per minute per API key.",
              {"customer_tier": "pro"}),
        Chunk("c::0", "TIC-1004",
              "Ticket TIC-1004: Refunds go back to the original payment method.",
              {"customer_tier": "pro"}),
    ]


def test_tokenize_keeps_hyphenated_ids_whole():
    toks = tokenize("See TIC-1001 or ERR-4032 for details, sk_test keys are exempt.")
    assert "tic-1001" in toks
    assert "err-4032" in toks
    assert "sk_test" in toks
    # split on real punctuation
    assert "details" in toks and "for" in toks


def test_exact_term_match_outranks_unrelated_document():
    index = BM25Index(_chunks())
    hits = index.search("TIC-1001", k=3)
    assert hits[0].chunk.source == "TIC-1001"


def test_shared_vocabulary_still_discriminates_by_specific_terms():
    """'requests per minute' appears in both rate-limit tickets, but 'Free'
    is the term that should tip the ranking toward TIC-1001."""
    index = BM25Index(_chunks())
    hits = index.search("Free plan requests per minute", k=3)
    assert hits[0].chunk.source == "TIC-1001"


def test_no_term_overlap_returns_nothing():
    index = BM25Index(_chunks())
    assert index.search("completely unrelated automobile maintenance", k=3) == []


def test_empty_query_returns_nothing():
    index = BM25Index(_chunks())
    assert index.search("", k=3) == []
    assert index.search("   ", k=3) == []


def test_empty_corpus_returns_nothing():
    index = BM25Index([])
    assert index.search("anything", k=3) == []
    assert len(index) == 0


def test_filter_restricts_before_ranking():
    index = BM25Index(_chunks())
    hits = index.search("plan requests per minute", k=5, flt=MetaFilter({"customer_tier": "pro"}))
    assert all(h.chunk.metadata["customer_tier"] == "pro" for h in hits)
    assert "TIC-1001" not in {h.chunk.source for h in hits}


def test_k_limits_result_count():
    index = BM25Index(_chunks())
    hits = index.search("plan requests per minute API key", k=1)
    assert len(hits) == 1


def test_from_store_reads_all_chunks():
    class FakeStore:
        def all_chunks(self):
            return _chunks()

    index = BM25Index.from_store(FakeStore())
    assert len(index) == 3
