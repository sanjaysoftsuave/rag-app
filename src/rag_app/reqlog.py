"""Week 11: per-request logging, real per-request cost, an exact-match answer
cache, and the failure -> test loop. See CLAUDE.md's "Watching the app" section.

WHY THIS LIVES IN ONE NEW MODULE, NOT SCATTERED ACROSS ui.py/cli.py
---------------------------------------------------------------------
`ui.py`'s `tab_ask()` already had two of this week's pieces, independently
invented and never shared: per-stage timing via `_Timed` proxy classes, and a
one-answer-per-session memo keyed on `(question, context chunk ids, model)`.
Both were UI-local and ephemeral — rebuilt every Streamlit rerun, gone when the
tab closes. This module PROMOTES both into durable, reusable pieces so the CLI
gets the same instrumentation the UI always had, and the UI's own memo becomes
a cache that actually survives past one browser session.

PIPELINE.PY STILL HAS NO TIMING CODE
--------------------------------------
Exactly the invariant CLAUDE.md already states for `ui.py`'s original proxies:
timing happens by wrapping the INJECTED `embedder`/`reranker`/`store`/
`generate_fn`, never by instrumenting `ask()` itself. `instrumented_ask()`
below is the only thing that knows about stopwatches; `pipeline.ask()` is
unchanged except for the one additive `usage=` kwarg Week 10 already
established the shape of for `run_agent`/`run_team`.

THE CACHE KEY IS CHUNK IDS, NOT QUESTION TEXT ALONE
------------------------------------------------------
Keying on `(question, model)` only would keep serving a cached answer after
the corpus is re-ingested and the SAME question now retrieves different text —
a correct-looking answer built on stale evidence, with no error anywhere.
Including the retrieved chunk ids makes the cache invalidate itself the
instant retrieval changes, for free.

THE PLANTED DRILL ENTRY IS MARKED, NEVER HIDDEN
---------------------------------------------------
`plant_bad_answer()` exists because real traffic cannot be relied on to
produce a wrong answer on demand, and "practice finding a bad answer" needs
one to find. Same disclosed-fabrication precedent `data/redteam/`'s attack
corpus already set: `planted_for_drill=True` sits in the record itself, found
by `search_log()` like any other entry, never disguised as organic traffic.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_app.agent_eval import percentile
from rag_app.config import AppConfig
from rag_app.evaluate import GoldQuestion, gold_path, load_gold, write_gold
from rag_app.generate import Answer, generate_answer
from rag_app.llm import Usage, usage_summary
from rag_app.pipeline import ask
from rag_app.store import ScoredChunk

LOG_DIRNAME = "logs"
LOG_FILENAME = "requests.jsonl"
CACHE_DIRNAME = "cache"
CACHE_FILENAME = "answers.json"


def log_path(cfg: AppConfig) -> Path:
    return cfg.tickets_dir.parent / LOG_DIRNAME / LOG_FILENAME


def cache_path(cfg: AppConfig) -> Path:
    return cfg.cache.path or (cfg.tickets_dir.parent / CACHE_DIRNAME / CACHE_FILENAME)


def _new_request_id() -> str:
    now = datetime.now(timezone.utc)
    return f"rq-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stopwatch proxies — moved here from ui.py, verbatim, so the CLI gets the
# same per-stage timing the UI always had instead of a second copy.
# ---------------------------------------------------------------------------


class _Timed:
    """Base proxy: forwards everything it does not explicitly time."""

    def __init__(self, inner, sink: dict, label: str):
        self._inner = inner
        self._sink = sink
        self._label = label

    def _record(self, started: float) -> None:
        self._sink[self._label] = self._sink.get(self._label, 0.0) + (
            time.perf_counter() - started
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedEmbedder(_Timed):
    def encode_queries(self, texts):
        started = time.perf_counter()
        try:
            return self._inner.encode_queries(texts)
        finally:
            self._record(started)

    def encode_documents(self, texts):
        return self._inner.encode_documents(texts)


class TimedReranker(_Timed):
    def predict(self, pairs):
        started = time.perf_counter()
        try:
            return self._inner.predict(pairs)
        finally:
            self._record(started)


class TimedStore(_Timed):
    def search(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._inner.search(*args, **kwargs)
        finally:
            self._record(started)

    def __len__(self):
        return len(self._inner)


def instrumented_ask(
    question: str,
    cfg: AppConfig,
    *,
    preset: str | None = None,
    embedder=None,
    reranker=None,
    store=None,
    generate_fn=None,
    flt=None,
    bm25=None,
    usage: Usage | None = None,
) -> tuple[Answer, dict[str, float]]:
    """Time each stage through the DI seam `ask()` already exposes.

    Pure instrumentation — no cache, no persistence — so something that wants
    timing without logging can use it directly. `embedder`/`reranker`/`store`
    are wrapped only when given; a caller that lets `ask()` build its own
    (the usual case in tests, never in real CLI/UI use, both of which already
    load these once for model reuse) just gets a 0 for that stage rather than
    a crash.
    """
    timings: dict[str, float] = {}
    timed_embedder = TimedEmbedder(embedder, timings, "embed query") if embedder is not None else None
    timed_reranker = (
        TimedReranker(reranker, timings, "rerank (cross-encoder)") if reranker is not None else None
    )
    timed_store = TimedStore(store, timings, "retrieve (vector search)") if store is not None else None

    def timed_generate(q, contexts, cfg_):
        started = time.perf_counter()
        try:
            if generate_fn is not None:
                return generate_fn(q, contexts, cfg_)
            return generate_answer(q, contexts, cfg_, usage=usage)
        finally:
            timings["generate (LLM)"] = timings.get("generate (LLM)", 0.0) + (
                time.perf_counter() - started
            )

    started = time.perf_counter()
    answer = ask(
        question,
        preset=preset,
        config=cfg,
        embedder=timed_embedder,
        reranker=timed_reranker,
        store=timed_store,
        generate_fn=timed_generate,
        flt=flt,
        bm25=bm25,
    )
    timings["total"] = time.perf_counter() - started
    return answer, timings


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class AnswerCache:
    """An exact-match, cross-request cache for generated answers.

    Backed by one JSON file, loaded once and rewritten whole on every write —
    fine at this scale, the same "plain file, not a database" choice this repo
    makes everywhere except Qdrant. Not safe under concurrent writers; this
    app has exactly one.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def key(question: str, contexts: list[ScoredChunk], model: str, preset: str = "") -> str:
        """`preset` matters because `Chunk.chunk_id` is POSITIONAL
        (`"doc.pdf::0"`), not a content hash: two presets of the same corpus
        with different `chunk_size` produce chunks with the SAME id pattern
        over DIFFERENT text spans. Without `preset` in the key, asking the
        same question against two presets could collide on one cached
        answer. This does not, by itself, catch a re-ingest of the SAME
        preset that changes what a given chunk_id now contains — that is
        what `clear()` plus `ui.py`'s re-ingest handler calling it are for.
        """
        normalized = " ".join(question.strip().lower().split())
        chunk_ids = sorted(c.chunk.chunk_id for c in contexts)
        raw = json.dumps([normalized, chunk_ids, model, preset], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, record: dict) -> None:
        self._data[key] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def clear(self) -> None:
        """Wipe every cached answer. A re-ingest can change what an EXISTING
        chunk_id contains (edited source text, or a chunk_size change that
        reuses the same id sequence over different spans) without changing
        the id itself, which `key()`'s chunk-id-based invalidation cannot
        see. `ui.py`'s re-ingest handler calls this for exactly that reason —
        the same blunt fix its old session-local memo already used."""
        self._data = {}
        if self.path.exists():
            self.path.write_text("{}", encoding="utf-8")


def caching_generate(cache: AnswerCache, real_generate_fn, *, hit_flag: dict, preset: str = ""):
    """Wraps a `generate_fn`-shaped callable with the cache.

    `hit_flag` is a mutable one-key dict the caller reads afterward — the same
    technique `ui.py`'s own memo already used (`produced_real_text`), since a
    plain closure has no channel back to the caller once `ask()` has already
    consumed its return value.
    """

    def wrapped(question, contexts, cfg):
        key = cache.key(question, contexts, cfg.llm.model, preset)
        cached = cache.get(key)
        if cached is not None:
            hit_flag["hit"] = True
            return cached["text"]
        hit_flag["hit"] = False
        text = real_generate_fn(question, contexts, cfg)
        cache.put(key, {"text": text, "cached_at": _now_iso()})
        return text

    return wrapped


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestLog:
    """One real request, replayable from the record alone."""

    request_id: str
    timestamp: str
    question: str
    answer_text: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    gate: str = ""
    hallucinated_citations: list[str] = field(default_factory=list)
    bi_encoder_model: str = ""
    cross_encoder_model: str = ""
    llm_model: str = ""
    preset: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    cache_hit: bool = False
    # Never True by accident — see the module docstring. A record with this
    # set is a disclosed fabrication for the support drill, not a real answer.
    planted_for_drill: bool = False

    def describe(self) -> str:
        lines = [
            f"{self.request_id}  {self.timestamp}",
            f"Q: {self.question}",
            f"A: {self.answer_text}",
            f"sources: {', '.join(self.sources) or '(none)'}   gate={self.gate}"
            f"   cache_hit={self.cache_hit}",
        ]
        if self.timings:
            parts = ", ".join(f"{k}={v:.3f}s" for k, v in self.timings.items())
            lines.append(f"timings: {parts}")
        if self.usage:
            lines.append(
                f"usage: {self.usage['total_tokens']} tokens, "
                f"cost={'$%.6f' % self.usage['cost_usd'] if self.usage['cost_usd'] is not None else 'unpriced'}"
            )
        if self.planted_for_drill:
            lines.append("!! planted_for_drill=True — a disclosed, fabricated practice entry")
        return "\n".join(lines)


def append_log(record: RequestLog, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_log(path: Path) -> list[RequestLog]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(RequestLog(**json.loads(line)))
    return records


def search_log(
    records: list[RequestLog],
    *,
    query: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list[RequestLog]:
    """Substring search over question/answer/sources, plus an ISO-timestamp
    range. This is the whole support drill: find a specific record from a
    vague term, not an id you already know."""
    out = records
    if query:
        q = query.lower()
        out = [
            r
            for r in out
            if q in r.question.lower()
            or q in r.answer_text.lower()
            or any(q in s.lower() for s in r.sources)
        ]
    if since:
        out = [r for r in out if r.timestamp >= since]
    if until:
        out = [r for r in out if r.timestamp <= until]
    return out


def find_by_id(records: list[RequestLog], request_id: str) -> RequestLog | None:
    for r in records:
        if r.request_id == request_id:
            return r
    return None


def cost_report(records: list[RequestLog], *, generated: bool = True) -> dict[str, Any]:
    """Mean/p99 latency, tokens and cost, plus the cache hit rate.

    Reuses `agent_eval.percentile` (nearest-rank, no interpolation) rather
    than a third reimplementation of the same rule Week 8 already wrote down.
    A request with no `usage` at all (generation skipped, or cached)
    contributes 0 tokens/cost — correct, not a gap, since nothing was spent.

    A request that WAS metered but used an unpriced model also reads 0 for
    cost — same "None, not 0" precedent RAGAS and Week 10's `unpriced_tasks`
    already set, so `unpriced_requests` is counted separately rather than
    letting that 0 pass as a real measurement.

    `planted_for_drill` records are EXCLUDED and counted separately: a
    fabricated drill entry has no timings and no usage, so averaging it in
    would drag every mean toward zero with a request that never happened.
    `generated` says whether a real model produced these requests, so a
    snapshot of this report can never be quoted as a real number it isn't.
    """
    planted = sum(1 for r in records if r.planted_for_drill)
    records = [r for r in records if not r.planted_for_drill]
    if not records:
        return {
            "generated": generated, "n_requests": 0, "cache_hit_rate": 0.0,
            "planted_excluded": planted,
        }
    hits = sum(1 for r in records if r.cache_hit)
    unpriced = sum(1 for r in records if r.usage and r.usage.get("cost_usd") is None)
    latencies = [r.timings.get("total", 0.0) for r in records]
    tokens = [float(r.usage.get("total_tokens", 0)) if r.usage else 0.0 for r in records]
    costs = [float(r.usage.get("cost_usd") or 0.0) if r.usage else 0.0 for r in records]
    return {
        "generated": generated,
        "n_requests": len(records),
        "planted_excluded": planted,
        "cache_hit_rate": round(hits / len(records), 4),
        "unpriced_requests": unpriced,
        "mean_latency_s": round(sum(latencies) / len(latencies), 4),
        "p99_latency_s": round(percentile(latencies, 99), 4),
        "mean_tokens": round(sum(tokens) / len(tokens), 2),
        "p99_tokens": round(percentile(tokens, 99), 2),
        "mean_cost_usd": round(sum(costs) / len(costs), 6),
        "p99_cost_usd": round(percentile(costs, 99), 6),
    }


def build_record(
    question: str,
    cfg: AppConfig,
    answer: Answer,
    timings: dict[str, float],
    *,
    preset: str | None = None,
    usage: Usage | None = None,
    cache_hit: bool = False,
) -> RequestLog:
    """A `RequestLog` from an already-computed `(Answer, timings)` pair,
    WITHOUT writing it anywhere. Pure, so a dry run (`logs --replay-gold`
    without `--generate`) can still produce a cost report of its own shape
    while never polluting the real request log with "(dry run)" answers."""
    return RequestLog(
        request_id=_new_request_id(),
        timestamp=_now_iso(),
        question=question,
        answer_text=answer.text,
        sources=list(answer.sources),
        refused=answer.refused,
        gate=answer.gate,
        hallucinated_citations=list(answer.hallucinated_citations),
        bi_encoder_model=cfg.bi_encoder_model,
        cross_encoder_model=cfg.cross_encoder_model,
        llm_model=cfg.llm.model,
        preset=preset or cfg.default_preset,
        timings=timings,
        usage=usage_summary(usage, cfg),
        cache_hit=cache_hit,
    )


def record_ask(
    question: str,
    cfg: AppConfig,
    answer: Answer,
    timings: dict[str, float],
    *,
    preset: str | None = None,
    usage: Usage | None = None,
    cache_hit: bool = False,
    file: Path | None = None,
) -> RequestLog:
    """Build and persist a `RequestLog` from an already-computed
    `(Answer, timings)` pair. The lower-level half of `log_ask`, split out so
    a caller with its own instrumentation needs — `ui.py`'s `tab_ask()`, which
    must decide PER RERUN whether a cache miss is even allowed to reach the
    real LLM (never pay for a slider drag) — can still get one durable log
    line without going through `log_ask`'s own cache/usage composition.
    """
    record = build_record(
        question, cfg, answer, timings, preset=preset, usage=usage, cache_hit=cache_hit,
    )
    append_log(record, file or log_path(cfg))
    return record


def log_ask(
    question: str,
    cfg: AppConfig,
    *,
    preset: str | None = None,
    embedder=None,
    reranker=None,
    store=None,
    generate_fn=None,
    flt=None,
    bm25=None,
    cache: AnswerCache | None = None,
    use_cache: bool | None = None,
    client: Any | None = None,
    file: Path | None = None,
) -> tuple[Answer, RequestLog]:
    """The production entry point: real per-stage timing, real token/cost,
    the exact-match cache, and one durable JSON line per request.

    `generate_fn`, when given, is the caller's own stand-in (e.g. a dry-run
    script) and bypasses BOTH the cache and usage metering — same bypass
    precedent as `run_agent`'s custom `llm_fn`. The common case (real
    generation) leaves it `None` and gets both. `client` threads to
    `generate_answer` exactly as it always has, so a test can inject a fake
    one and exercise the real default path (and therefore the cache) without
    a key or the network.

    `ui.py`'s `tab_ask()` does NOT use this — it needs finer control over
    whether a cache miss may even reach the real LLM (never pay for a slider
    drag) than a single `generate_fn` override can express, so it composes
    `instrumented_ask` + `record_ask` directly. See `record_ask`.
    """
    want_cache = cfg.cache.enabled if use_cache is None else use_cache
    hit_flag: dict[str, bool] = {"hit": False}
    usage = Usage()

    if generate_fn is not None:
        gen = generate_fn
    else:
        real_gen = lambda q, c, cfg_: generate_answer(q, c, cfg_, client=client, usage=usage)  # noqa: E731
        resolved_preset = preset or cfg.default_preset
        gen = (
            caching_generate(cache, real_gen, hit_flag=hit_flag, preset=resolved_preset)
            if (cache and want_cache)
            else real_gen
        )

    answer, timings = instrumented_ask(
        question,
        cfg,
        preset=preset,
        embedder=embedder,
        reranker=reranker,
        store=store,
        generate_fn=gen,
        flt=flt,
        bm25=bm25,
    )
    record = record_ask(
        question, cfg, answer, timings, preset=preset, usage=usage,
        cache_hit=hit_flag["hit"], file=file,
    )
    return answer, record


def plant_bad_answer(
    cfg: AppConfig,
    *,
    question: str,
    wrong_answer: str,
    sources: list[str],
    file: Path | None = None,
) -> RequestLog:
    """Appends ONE record openly marked `planted_for_drill=True`. See the
    module docstring for why this is fabricated but never hidden."""
    record = RequestLog(
        request_id=_new_request_id(),
        timestamp=_now_iso(),
        question=question,
        answer_text=wrong_answer,
        sources=list(sources),
        refused=False,
        gate="answered",
        bi_encoder_model=cfg.bi_encoder_model,
        cross_encoder_model=cfg.cross_encoder_model,
        llm_model=cfg.llm.model,
        preset=cfg.default_preset,
        planted_for_drill=True,
    )
    append_log(record, file or log_path(cfg))
    return record


def promote_to_gold(
    record: RequestLog,
    cfg: AppConfig,
    *,
    must_contain: list[str] | None = None,
    unanswerable: bool = False,
    note: str = "",
) -> GoldQuestion:
    """The failure -> test step. Builds a `GoldQuestion` from a found record
    and appends it to `data/gold.yaml` via `evaluate.write_gold` — the exact
    file `eval`/`debug` already read, so the promoted entry is graded on the
    very next `eval --generate` with no second code path.

    The minimum-viable-entry rule is `parse_gold`'s own, checked here with the
    SAME message, so a promoted record can never silently fail validation
    later: it is answerable (the default) and names nothing to check for.
    """
    must = list(must_contain or [])
    if not unanswerable and not must:
        raise ValueError(
            f"Promoting {record.request_id!r} is answerable but names nothing to look "
            f"for. Add 'must_contain', or mark it 'unanswerable: true'."
        )
    question = GoldQuestion(
        question=record.question,
        must_contain=must,
        unanswerable=unanswerable,
        note=note or f"promoted from {record.request_id} ({record.timestamp})",
    )
    path = gold_path(cfg)
    try:
        existing = load_gold(cfg, path)
    except FileNotFoundError:
        existing = []
    write_gold([*existing, question], path)
    return question
