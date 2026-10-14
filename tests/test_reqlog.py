"""Per-request logging, real cost, the exact-match cache, and the
failure -> test loop. See CLAUDE.md's "Watching the app" section.
"""

from __future__ import annotations

import pytest
from conftest import FakeEmbedder, FakeReranker, make_config, make_qdrant_store

from rag_app.chunking import Chunk
from rag_app.evaluate import load_gold
from rag_app.reqlog import (
    AnswerCache,
    RequestLog,
    cache_path,
    caching_generate,
    cost_report,
    find_by_id,
    instrumented_ask,
    load_log,
    log_ask,
    log_path,
    plant_bad_answer,
    promote_to_gold,
    record_ask,
    search_log,
)


class FakeClient:
    """Duck-types `client.chat.completions.create`, same pattern as test_llm.py."""

    def __init__(self, reply="ok", usage=None):
        self.reply = reply
        self.usage = usage
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls += 1

                class _Msg:
                    content = outer.reply

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]
                    usage = outer.usage

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def seeded_store(tmp_path, text="The link lasts 60 minutes."):
    embedder = FakeEmbedder()
    chunks = [Chunk("doc.pdf::0", "doc.pdf", text, {})]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = make_qdrant_store(tmp_path, chunks, vectors)
    return embedder, store


def record(**kw) -> RequestLog:
    base = dict(
        request_id="rq-1", timestamp="2026-01-01T00:00:00+00:00",
        question="q?", answer_text="a.",
    )
    base.update(kw)
    return RequestLog(**base)


# ---------------------------------------------------------------------------
# AnswerCache
# ---------------------------------------------------------------------------


def test_cache_round_trips_through_the_file(tmp_path):
    path = tmp_path / "answers.json"
    cache = AnswerCache(path)
    key = "k1"
    assert cache.get(key) is None
    cache.put(key, {"text": "the answer"})
    assert cache.get(key) == {"text": "the answer"}

    reopened = AnswerCache(path)
    assert reopened.get(key)["text"] == "the answer"


def test_cache_key_is_stable_for_the_same_question_and_context(tmp_path):
    chunk = type("C", (), {"chunk": Chunk("doc.pdf::0", "doc.pdf", "x", {})})()
    k1 = AnswerCache.key("How long?", [chunk], "model-a")
    k2 = AnswerCache.key("how long?  ", [chunk], "model-a")  # case/whitespace-insensitive
    assert k1 == k2


def test_cache_key_changes_when_the_retrieved_chunks_change(tmp_path):
    a = type("C", (), {"chunk": Chunk("doc.pdf::0", "doc.pdf", "x", {})})()
    b = type("C", (), {"chunk": Chunk("doc.pdf::1", "doc.pdf", "y", {})})()
    assert AnswerCache.key("q", [a], "m") != AnswerCache.key("q", [b], "m")


def test_cache_key_changes_when_the_model_changes():
    a = type("C", (), {"chunk": Chunk("doc.pdf::0", "doc.pdf", "x", {})})()
    assert AnswerCache.key("q", [a], "model-a") != AnswerCache.key("q", [a], "model-b")


def test_cache_key_changes_when_the_preset_changes():
    """chunk_id is POSITIONAL ("doc.pdf::0"), not a content hash: two presets
    of the same corpus can produce the SAME id over DIFFERENT text. Without
    preset in the key, asking the same question against two presets could
    collide on one cached answer."""
    a = type("C", (), {"chunk": Chunk("doc.pdf::0", "doc.pdf", "x", {})})()
    assert AnswerCache.key("q", [a], "m", preset="A") != AnswerCache.key("q", [a], "m", preset="C")


def test_cache_clear_empties_every_entry(tmp_path):
    path = tmp_path / "answers.json"
    cache = AnswerCache(path)
    cache.put("k1", {"text": "x"})
    cache.put("k2", {"text": "y"})
    cache.clear()
    assert cache.get("k1") is None and cache.get("k2") is None
    assert AnswerCache(path).get("k1") is None  # the file itself was wiped, not just memory


def test_caching_generate_calls_the_real_generator_once_then_not_again(tmp_path):
    cache = AnswerCache(tmp_path / "answers.json")
    calls = {"n": 0}

    def real(q, c, cfg):
        calls["n"] += 1
        return "the real answer"

    hit_flag = {"hit": False}
    wrapped = caching_generate(cache, real, hit_flag=hit_flag)
    cfg = make_config(tmp_path)

    text1 = wrapped("q?", [], cfg)
    assert text1 == "the real answer" and calls["n"] == 1 and hit_flag["hit"] is False

    text2 = wrapped("q?", [], cfg)
    assert text2 == "the real answer" and calls["n"] == 1  # not called again
    assert hit_flag["hit"] is True


# ---------------------------------------------------------------------------
# instrumented_ask
# ---------------------------------------------------------------------------


def test_instrumented_ask_times_every_stage(tmp_path):
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)

    answer, timings = instrumented_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        generate_fn=lambda q, c, cfg_: "60 minutes [doc.pdf].",
    )
    store.close()
    for stage in ("embed query", "retrieve (vector search)", "rerank (cross-encoder)",
                  "generate (LLM)", "total"):
        assert stage in timings
    assert answer.sources == ["doc.pdf"]


# ---------------------------------------------------------------------------
# log_ask
# ---------------------------------------------------------------------------


def test_record_ask_lets_a_caller_with_its_own_instrumentation_still_log(tmp_path):
    """The seam ui.py's tab_ask() needs: its own cache-or-skip-or-call
    decision happens OUTSIDE log_ask's composition, so it builds the
    (Answer, timings) pair itself and hands it to record_ask directly."""
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    answer, timings = instrumented_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        generate_fn=lambda q, c, cfg_: "60 minutes [doc.pdf].",
    )
    store.close()
    from rag_app.llm import Usage

    u = Usage()
    u.add(cfg.llm.model, 10, 5)
    rec = record_ask("how long?", cfg, answer, timings, usage=u, cache_hit=False)
    assert rec.answer_text == answer.text
    assert rec.usage["total_tokens"] == 15
    assert load_log(log_path(cfg)) == [rec]


def test_log_ask_writes_one_line_and_returns_a_matching_record(tmp_path):
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    answer, rec = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        generate_fn=lambda q, c, cfg_: "60 minutes [doc.pdf].",
    )
    store.close()
    assert rec.question == "how long?"
    assert rec.answer_text == answer.text
    assert rec.sources == answer.sources
    assert rec.request_id.startswith("rq-")

    records = load_log(log_path(cfg))
    assert len(records) == 1
    assert records[0] == rec


def test_log_ask_with_the_real_default_path_populates_usage_and_hits_the_cache(tmp_path):
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    client = FakeClient("60 minutes [doc.pdf].", usage=_FakeUsage(100, 20))
    cache = AnswerCache(cache_path(cfg))

    _, rec1 = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        cache=cache, client=client,
    )
    assert rec1.cache_hit is False
    assert rec1.usage["total_tokens"] == 120
    assert client.calls == 1

    _, rec2 = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        cache=cache, client=client,
    )
    store.close()
    assert rec2.cache_hit is True
    assert rec2.usage is None  # nothing was metered on a cache hit
    assert client.calls == 1  # the real generator was NOT called again


def test_a_custom_generate_fn_bypasses_both_cache_and_usage(tmp_path):
    """Same documented precedent as run_agent's custom llm_fn: a caller's own
    stand-in (e.g. the UI's 'Call the LLM' off toggle) is its own
    instrumentation and gets neither for free."""
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    cache = AnswerCache(cache_path(cfg))
    calls = {"n": 0}

    def stand_in(q, c, cfg_):
        calls["n"] += 1
        return "60 minutes [doc.pdf]."

    _, rec1 = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        cache=cache, generate_fn=stand_in,
    )
    _, rec2 = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        cache=cache, generate_fn=stand_in,
    )
    store.close()
    assert calls["n"] == 2  # never cached
    assert rec1.usage is None and rec2.usage is None
    assert rec1.cache_hit is False and rec2.cache_hit is False


def test_cache_disabled_in_config_never_engages_even_with_a_cache_object(tmp_path):
    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    client = FakeClient("60 minutes [doc.pdf].", usage=_FakeUsage(1, 1))
    cache = AnswerCache(cache_path(cfg))

    log_ask("how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
             cache=cache, client=client, use_cache=False)
    _, rec = log_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        cache=cache, client=client, use_cache=False,
    )
    store.close()
    assert rec.cache_hit is False
    assert client.calls == 2


# ---------------------------------------------------------------------------
# search_log / find_by_id / cost_report
# ---------------------------------------------------------------------------


def test_search_log_matches_question_answer_and_sources():
    records = [
        record(request_id="a", question="refund window?", answer_text="14 days.", sources=["doc.pdf"]),
        record(request_id="b", question="reset link", answer_text="60 minutes.", sources=["doc.pdf"]),
    ]
    assert [r.request_id for r in search_log(records, query="refund")] == ["a"]
    assert [r.request_id for r in search_log(records, query="60 minutes")] == ["b"]


def test_search_log_filters_by_timestamp_range():
    records = [
        record(request_id="old", timestamp="2026-01-01T00:00:00+00:00"),
        record(request_id="new", timestamp="2026-06-01T00:00:00+00:00"),
    ]
    assert [r.request_id for r in search_log(records, since="2026-03-01")] == ["new"]
    assert [r.request_id for r in search_log(records, until="2026-03-01")] == ["old"]


def test_find_by_id_returns_none_for_an_unknown_id():
    assert find_by_id([record(request_id="a")], "nope") is None
    assert find_by_id([record(request_id="a")], "a") is not None


def test_cost_report_on_an_empty_log():
    assert cost_report([]) == {
        "generated": True, "n_requests": 0, "cache_hit_rate": 0.0, "planted_excluded": 0,
    }


def test_cost_report_records_whether_a_real_model_produced_it():
    """The flag whose absence let a Week-10 agent number be quoted as a
    Week-11 cache number. A snapshot must say what it is."""
    assert cost_report([record()], generated=False)["generated"] is False
    assert cost_report([record()])["generated"] is True


def test_cost_report_excludes_planted_drill_entries():
    """A fabricated drill record has no timings and no usage; averaging it in
    drags every mean toward zero with a request that never happened."""
    real = record(request_id="a", timings={"total": 2.0},
                  usage={"prompt_tokens": 90, "completion_tokens": 10,
                         "total_tokens": 100, "cost_usd": 0.01})
    planted = record(request_id="p", planted_for_drill=True)
    report = cost_report([real, planted])
    assert report["n_requests"] == 1
    assert report["planted_excluded"] == 1
    assert report["mean_latency_s"] == 2.0
    assert report["mean_tokens"] == 100.0


def test_build_record_never_writes_the_log(tmp_path):
    """The dry-run seam: shape without polluting the real request log."""
    from rag_app.reqlog import build_record

    embedder, store = seeded_store(tmp_path)
    cfg = make_config(tmp_path, score_threshold=0.0)
    answer, timings = instrumented_ask(
        "how long?", cfg, embedder=embedder, reranker=FakeReranker(0.9), store=store,
        generate_fn=lambda q, c, cfg_: "(dry run)",
    )
    store.close()
    rec = build_record("how long?", cfg, answer, timings)
    assert rec.answer_text == "(dry run)"
    assert not log_path(cfg).exists()


def test_cost_report_aggregates_latency_tokens_cost_and_cache_hits():
    records = [
        record(request_id="a", timings={"total": 1.0}, cache_hit=False,
               usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100, "cost_usd": 0.01}),
        record(request_id="b", timings={"total": 0.0}, cache_hit=True, usage=None),
    ]
    report = cost_report(records)
    assert report["n_requests"] == 2
    assert report["cache_hit_rate"] == 0.5
    assert report["mean_tokens"] == 50.0
    assert report["mean_cost_usd"] == 0.005
    assert report["unpriced_requests"] == 0


def test_cost_report_flags_unpriced_requests_separately_from_zero_cost():
    records = [
        record(request_id="a",
               usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost_usd": None}),
    ]
    report = cost_report(records)
    assert report["mean_cost_usd"] == 0.0
    assert report["unpriced_requests"] == 1


# ---------------------------------------------------------------------------
# plant_bad_answer
# ---------------------------------------------------------------------------


def test_plant_bad_answer_is_always_marked_and_findable(tmp_path):
    cfg = make_config(tmp_path)
    planted = plant_bad_answer(
        cfg, question="what is the refund window?", wrong_answer="7 days, no exceptions.",
        sources=["doc.pdf"],
    )
    assert planted.planted_for_drill is True

    records = load_log(log_path(cfg))
    assert len(records) == 1
    hits = search_log(records, query="refund")
    assert hits and hits[0].planted_for_drill is True


# ---------------------------------------------------------------------------
# promote_to_gold — the failure -> test step
# ---------------------------------------------------------------------------


def test_promote_to_gold_appends_a_reloadable_entry(tmp_path):
    cfg = make_config(tmp_path)
    found = record(request_id="rq-x", question="what is the refund window?",
                    answer_text="7 days, no exceptions.")
    q = promote_to_gold(found, cfg, must_contain=["14 days"], note="corrected refund window")
    assert q.question == "what is the refund window?"
    assert q.must_contain == ["14 days"]

    reloaded = load_gold(cfg)
    assert len(reloaded) == 1
    assert reloaded[0].must_contain == ["14 days"]


def test_promote_to_gold_preserves_existing_entries(tmp_path):
    cfg = make_config(tmp_path)
    from rag_app.evaluate import GoldQuestion, write_gold
    from rag_app.evaluate import gold_path as _gold_path

    write_gold([GoldQuestion(question="already here", must_contain=["x"])], _gold_path(cfg))

    found = record(request_id="rq-x", question="new one")
    promote_to_gold(found, cfg, must_contain=["y"])

    reloaded = load_gold(cfg)
    assert {q.question for q in reloaded} == {"already here", "new one"}


def test_promote_to_gold_rejects_an_entry_with_nothing_to_check(tmp_path):
    """Same rule parse_gold itself enforces, checked here BEFORE any file is
    touched, so a bad promotion can never corrupt data/gold.yaml."""
    cfg = make_config(tmp_path)
    found = record(request_id="rq-x", question="q")
    with pytest.raises(ValueError, match="names nothing to look"):
        promote_to_gold(found, cfg)
    assert not (cfg.tickets_dir.parent / "gold.yaml").exists()


def test_promote_to_gold_accepts_unanswerable_with_no_must_contain(tmp_path):
    cfg = make_config(tmp_path)
    found = record(request_id="rq-x", question="what does the sun eat for breakfast?")
    q = promote_to_gold(found, cfg, unanswerable=True)
    assert q.unanswerable is True
    assert q.must_contain == []
