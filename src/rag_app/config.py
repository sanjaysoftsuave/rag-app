from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Name given to the single configuration from a `chunking:` block. It becomes
# the store directory suffix (data/store/qdrant_default), so changing it
# orphans an existing store rather than breaking anything.
DEFAULT_PRESET = "default"


@dataclass(frozen=True)
class ChunkPreset:
    """How text is windowed. There is only one method — see chunking.py."""

    chunk_size: int
    overlap: int

    def describe(self) -> str:
        return f"{self.chunk_size}/{self.overlap}"


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    temperature: float
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class QdrantConfig:
    url: str | None = None
    m: int = 16
    ef_construct: int = 100
    hnsw_ef: int = 128


@dataclass(frozen=True)
class RetrievalConfig:
    """Which candidate-generation strategy `ask()` uses.

    'dense'  — bi-encoder cosine search only.
    'hybrid' — dense + BM25 keyword search, fused by Reciprocal Rank Fusion.
               See hybrid.py for why RRF rather than a weighted score blend.

    `query_mode` transforms the question before retrieval (rewrite.py) and
    `mmr` re-selects the reranked candidates for coverage (mmr.py). Both are
    off by default: each is a real change to what the model reads, and neither
    is worth enabling without a gold set to show it helped.
    """

    mode: str = "dense"
    rrf_k: int = 60
    bm25_pool: int = 20
    # Query transform applied BEFORE retrieval: "off" | "rewrite" | "hyde".
    # Both non-off modes cost an LLM call per question — see rewrite.py.
    query_mode: str = "off"
    # Maximal Marginal Relevance over the reranked candidates. lambda 1.0 is
    # pure relevance (identical to off); lower trades relevance for coverage.
    mmr: bool = False
    mmr_lambda: float = 0.7


@dataclass(frozen=True)
class EvaluationConfig:
    """How answers are graded, and by whom.

    The judge is a SEPARATE, STRONGER model than the generator, on purpose. A
    model grading its own output shows measurable self-preference: it rates its
    own phrasing above an equivalent answer worded differently. The whole value
    of a judge is a second opinion, so it defaults to a different model — and
    `judge_validation.py` exists because that assumption is itself worth
    checking against human labels rather than trusted.

    Everything here is inert until an explicit `eval` flag is typed. See
    judge.py (LLM-as-Judge, G-Eval) and ragas_metrics.py.
    """

    judge_model: str = "openai/gpt-4o"
    judge_temperature: float = 0.0
    # Longer than generation's 30s: a judge prompt carries the answer AND the
    # contexts, and is asked to reason before it decides.
    judge_timeout_seconds: float = 60.0
    judge_base_url: str = ""  # "" = share llm.base_url
    # G-Eval averages N sampled scores in place of the paper's logprob
    # weighting, which OpenRouter does not reliably forward. See judge.py.
    geval_samples: int = 5
    geval_temperature: float = 1.0
    # Hard ceiling, checked BEFORE the first call rather than partway through.
    max_llm_calls: int = 200


BUILTIN_TOOL_NAMES = (
    "search_documents",
    "keyword_search",
    "list_sources",
    "read_source",
)


@dataclass(frozen=True)
class AgentMemoryConfig:
    """What the agent remembers between turns, and where it lives."""

    enabled: bool = False
    backend: str = "plain"          # "plain" | "mem0"
    dir: Path | None = None         # None -> store_dir.parent / "agent_memory"
    short_term_turns: int = 8
    summary_trigger_chars: int = 4000
    summary_target_chars: int = 800
    vector_recall_k: int = 3


# What a tool is allowed to reach. Two read grades and NO write grade, because
# this app has nothing to write — every tool reads an already-built index.
#
# The value of the field today is not the list: it is that a tool added later
# gets no grant by default and is therefore DENIED. Default-deny is the
# property; the vocabulary is just how it is expressed.
READ_INDEX = "read:index"        # search, see labels, see excerpts
READ_DOCUMENT = "read:document"  # pull one whole document end to end
# Week 9: a tool discovered over MCP. Sharper than the other two grades — an
# MCP tool arrives with no capability annotation from the protocol AT ALL, so
# every discovered tool is tagged READ_EXTERNAL regardless of what it actually
# does, and mcp_client.build_mcp_registry grants it only when the CLI is
# passed --mcp-allow. Registering it here (so Tool.register accepts it) is not
# the same as granting it — see build_mcp_registry's docstring.
READ_EXTERNAL = "read:external"
CAPABILITIES = (READ_INDEX, READ_DOCUMENT, READ_EXTERNAL)


@dataclass(frozen=True)
class DefenceConfig:
    """The injection defences, each switchable so the A/B is a measured delta.

    Every one defaults ON. The undefended arm exists to answer "what did this
    buy?" without checking out a previous commit — which is the difference
    between a measurement and an assertion.

    They are layered on purpose, because each covers what the others miss:
    delimiters and DATA_BOUNDARY tell the model what is data; `neutralize`
    removes the imperative a model might obey anyway; `verify_evidence` refuses
    a citation the store cannot vouch for; and `forged_citation_gate` throws
    away an answer that cited one regardless.
    """

    neutralize: bool = True
    verify_evidence: bool = True
    data_delimiters: bool = True
    enforce_capabilities: bool = True
    forged_citation_gate: bool = True
    question_scan: bool = True


@dataclass(frozen=True)
class AgentConfig:
    """The ReAct loop's budgets and tool set.

    Every budget produces a VISIBLE failure when it trips: DONT_KNOW, no
    sources, and a stop_reason naming the number. An agent that ran out of steps
    has not answered, and attaching partial prose to a truncated run is the
    dishonesty the failed=True precedent exists to prevent.

    max_steps is the cost knob: each step is one LLM call, so a 6-step agent can
    cost 6x what ask() spends on a question ask() answers in one. See compare.py
    before raising it.
    """

    implementation: str = "plain"   # "plain" | "langgraph"
    max_steps: int = 6
    max_tool_calls: int = 8
    # Separate from max_steps because a parse failure burns an LLM call without
    # producing a step's worth of progress.
    max_llm_calls: int = 8
    # CHARACTERS, not tokens (~4 chars/token). A real count needs a tokenizer
    # dependency this project deliberately does not carry — the same honesty
    # chunk_size already requires.
    max_prompt_chars: int = 24000
    wall_clock_seconds: float = 60.0
    max_parse_failures: int = 2
    repeat_action_limit: int = 2
    max_observation_chars: int = 2000
    tools: tuple[str, ...] = BUILTIN_TOOL_NAMES
    # Which documents read_source may open. () means unrestricted - and the
    # tool's own description says so, because a scope that silently is not one
    # is worse than no scope at all.
    read_source_allow: tuple[str, ...] = ()
    memory: AgentMemoryConfig = AgentMemoryConfig()
    defences: DefenceConfig = DefenceConfig()


@dataclass(frozen=True)
class AppConfig:
    docs_dir: Path
    tickets_dir: Path
    store_dir: Path
    chunk_presets: dict[str, ChunkPreset]
    default_preset: str
    bi_encoder_model: str
    cross_encoder_model: str
    retrieve_k: int
    rerank_n: int
    score_threshold: float
    llm: LlmConfig
    llm_api_key: str | None
    rerank_score_scale: str = "sigmoid"
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    agent: AgentConfig = AgentConfig()


def judge_llm(cfg: AppConfig) -> LlmConfig:
    """The LlmConfig the judge runs on: same key and account, different model.

    Returned as an `LlmConfig` so `llm.chat_once` cannot tell a judge call from
    a generation call — the only difference that should exist between them is
    the model, the temperature and the timeout, and this makes that literally
    true rather than a convention.
    """
    return LlmConfig(
        base_url=cfg.evaluation.judge_base_url or cfg.llm.base_url,
        model=cfg.evaluation.judge_model,
        temperature=cfg.evaluation.judge_temperature,
        timeout_seconds=cfg.evaluation.judge_timeout_seconds,
    )


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(config_path: Path | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")
    path = config_path or (ROOT / "config.yaml")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Two accepted shapes, and the rest of the app cannot tell them apart:
    #
    #   chunking: {strategy, chunk_size, overlap}   -> one config named "default"
    #   chunk_presets: {A: {...}, B: {...}}         -> several, plus default_preset
    #
    # Everything downstream iterates `sorted(cfg.chunk_presets)`, so the single
    # form is just a one-entry map. That is what makes going back to several a
    # YAML-only change — `compare`, `eval --all`, `chunks --all` and the UI's
    # preset selector all start working again with no code touched.
    raw_presets = raw.get("chunk_presets")
    if raw_presets:
        presets = {
            name: ChunkPreset(
                chunk_size=int(vals["chunk_size"]),
                overlap=int(vals["overlap"]),
            )
            for name, vals in raw_presets.items()
        }
        default_preset = str(raw["default_preset"])
        # Fail at load time with a useful message rather than later with a
        # missing store directory that looks like an ingest problem.
        if default_preset not in presets:
            raise ValueError(
                f"default_preset {default_preset!r} is not defined in chunk_presets "
                f"{sorted(presets)}"
            )
    else:
        chunking = raw.get("chunking")
        if not chunking:
            raise ValueError(
                "config.yaml needs either a 'chunking' block (one configuration) or a "
                "'chunk_presets' map plus 'default_preset' (several). Found neither."
            )
        presets = {
            DEFAULT_PRESET: ChunkPreset(
                chunk_size=int(chunking["chunk_size"]),
                overlap=int(chunking["overlap"]),
            )
        }
        default_preset = DEFAULT_PRESET

    retrieve_k = int(raw["retrieve_k"])
    rerank_n = int(raw["rerank_n"])
    if rerank_n > retrieve_k:
        raise ValueError(
            f"rerank_n ({rerank_n}) exceeds retrieve_k ({retrieve_k}); the reranker can "
            f"only ever see {retrieve_k} candidates, so the extra slots are unreachable."
        )

    scale = str(raw.get("rerank_score_scale", "sigmoid"))
    if scale not in {"sigmoid", "raw"}:
        raise ValueError(f"rerank_score_scale must be 'sigmoid' or 'raw', got {scale!r}")

    threshold = float(raw["score_threshold"])
    if scale == "sigmoid" and not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"score_threshold {threshold} is outside 0-1, but rerank_score_scale is "
            f"'sigmoid' so scores are probabilities. Did you mean 'raw'?"
        )

    r_raw = raw.get("retrieval") or {}
    retrieval_mode = str(r_raw.get("mode", "dense"))
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError(
            f"retrieval.mode must be 'dense' or 'hybrid', got {retrieval_mode!r}"
        )

    query_mode = str(r_raw.get("query_mode", "off"))
    if query_mode not in {"off", "rewrite", "hyde"}:
        raise ValueError(
            f"retrieval.query_mode must be 'off', 'rewrite' or 'hyde', got {query_mode!r}"
        )

    mmr_lambda = float(r_raw.get("mmr_lambda", 0.7))
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError(
            f"retrieval.mmr_lambda is a mix between relevance (1.0) and diversity "
            f"(0.0), so it must be between 0 and 1; got {mmr_lambda}"
        )

    e_raw = raw.get("evaluation") or {}
    geval_samples = int(e_raw.get("geval_samples", 5))
    if geval_samples < 1:
        raise ValueError(
            f"evaluation.geval_samples is {geval_samples}; G-Eval averages N sampled "
            f"scores, so N must be at least 1 or there is nothing to average."
        )
    geval_temperature = float(e_raw.get("geval_temperature", 1.0))
    if geval_samples > 1 and geval_temperature <= 0.0:
        raise ValueError(
            f"evaluation.geval_samples is {geval_samples} but geval_temperature is "
            f"{geval_temperature}: sampling {geval_samples} times at temperature 0 returns "
            f"the same score {geval_samples} times, costing {geval_samples}x for no "
            f"variance estimate. Either set geval_samples: 1 or raise geval_temperature."
        )
    judge_temperature = float(e_raw.get("judge_temperature", 0.0))
    if not 0.0 <= judge_temperature <= 2.0:
        raise ValueError(
            f"evaluation.judge_temperature is {judge_temperature}, outside 0-2. A judge "
            f"is asked for a verdict, not for variety; 0 is the usual value."
        )
    judge_model = str(e_raw.get("judge_model", "openai/gpt-4o")).strip()
    if not judge_model:
        raise ValueError(
            "evaluation.judge_model is empty. Name the model that grades answers, or "
            "remove the key to accept the default."
        )
    max_llm_calls = int(e_raw.get("max_llm_calls", 200))
    if max_llm_calls < 0:
        raise ValueError(
            f"evaluation.max_llm_calls is {max_llm_calls}; it is a ceiling on spending, "
            f"so it cannot be negative. Use 0 for no ceiling."
        )

    a_raw = raw.get("agent") or {}
    m_raw = a_raw.get("memory") or {}
    implementation = str(a_raw.get("implementation", "plain"))
    if implementation not in {"plain", "langgraph"}:
        raise ValueError(
            f"agent.implementation must be 'plain' or 'langgraph', got "
            f"{implementation!r}"
        )
    memory_backend = str(m_raw.get("backend", "plain"))
    if memory_backend not in {"plain", "mem0"}:
        raise ValueError(
            f"agent.memory.backend must be 'plain' or 'mem0', got {memory_backend!r}"
        )
    max_steps = int(a_raw.get("max_steps", 6))
    if max_steps < 1:
        raise ValueError(
            f"agent.max_steps is {max_steps}, so the loop would stop before its first "
            f"thought and every question would return DONT_KNOW."
        )
    summary_trigger = int(m_raw.get("summary_trigger_chars", 4000))
    summary_target = int(m_raw.get("summary_target_chars", 800))
    if summary_target >= summary_trigger:
        raise ValueError(
            f"agent.memory.summary_target_chars ({summary_target}) must be below "
            f"summary_trigger_chars ({summary_trigger}); otherwise summarizing never "
            f"shrinks the buffer and the trigger fires on every turn."
        )
    d_raw = a_raw.get("defences") or {}
    for key in d_raw:
        if key not in DefenceConfig.__dataclass_fields__:
            raise ValueError(
                f"agent.defences has unknown switch {key!r}; the legal ones are "
                f"{sorted(DefenceConfig.__dataclass_fields__)}. A misspelt switch "
                f"would silently leave that defence ON while you believed it off."
            )
    tool_names = tuple(a_raw.get("tools", BUILTIN_TOOL_NAMES))
    unknown = [t for t in tool_names if t not in BUILTIN_TOOL_NAMES]
    if unknown:
        raise ValueError(
            f"agent.tools names unknown tools {unknown}; known tools are "
            f"{list(BUILTIN_TOOL_NAMES)}."
        )

    q_raw = raw.get("qdrant") or {}
    llm_raw = raw["llm"]

    return AppConfig(
        # Vestigial: nothing reads docs_dir since the corpus moved to
        # tickets_dir, which holds .jsonl, .md/.txt AND .pdf together. It was
        # a required key purely because the loader asked for it unconditionally
        # — now optional, so config.yaml need not carry a dead setting.
        docs_dir=_resolve(raw.get("docs_dir", "data/docs")),
        tickets_dir=_resolve(raw.get("tickets_dir", "data/tickets")),
        store_dir=_resolve(raw.get("store_dir", "data/store")),
        chunk_presets=presets,
        default_preset=default_preset,
        bi_encoder_model=str(raw["bi_encoder_model"]),
        cross_encoder_model=str(raw["cross_encoder_model"]),
        retrieve_k=retrieve_k,
        rerank_n=rerank_n,
        score_threshold=threshold,
        rerank_score_scale=scale,
        retrieval=RetrievalConfig(
            mode=retrieval_mode,
            rrf_k=int(r_raw.get("rrf_k", 60)),
            bm25_pool=int(r_raw.get("bm25_pool", 20)),
            query_mode=query_mode,
            mmr=bool(r_raw.get("mmr", False)),
            mmr_lambda=mmr_lambda,
        ),
        evaluation=EvaluationConfig(
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            judge_timeout_seconds=float(e_raw.get("judge_timeout_seconds", 60.0)),
            judge_base_url=str(e_raw.get("judge_base_url", "") or ""),
            geval_samples=geval_samples,
            geval_temperature=geval_temperature,
            max_llm_calls=max_llm_calls,
        ),
        agent=AgentConfig(
            implementation=implementation,
            max_steps=max_steps,
            max_tool_calls=int(a_raw.get("max_tool_calls", 8)),
            max_llm_calls=int(a_raw.get("max_llm_calls", 8)),
            max_prompt_chars=int(a_raw.get("max_prompt_chars", 24000)),
            wall_clock_seconds=float(a_raw.get("wall_clock_seconds", 60.0)),
            max_parse_failures=int(a_raw.get("max_parse_failures", 2)),
            repeat_action_limit=int(a_raw.get("repeat_action_limit", 2)),
            max_observation_chars=int(a_raw.get("max_observation_chars", 2000)),
            tools=tool_names,
            read_source_allow=tuple(a_raw.get("read_source_allow", ()) or ()),
            defences=DefenceConfig(
                **{k: bool(v) for k, v in d_raw.items()}
            ),
            memory=AgentMemoryConfig(
                enabled=bool(m_raw.get("enabled", False)),
                backend=memory_backend,
                dir=_resolve(m_raw["dir"]) if m_raw.get("dir") else None,
                short_term_turns=int(m_raw.get("short_term_turns", 8)),
                summary_trigger_chars=summary_trigger,
                summary_target_chars=summary_target,
                vector_recall_k=int(m_raw.get("vector_recall_k", 3)),
            ),
        ),
        qdrant=QdrantConfig(
            url=q_raw.get("url") or None,
            m=int(q_raw.get("m", 16)),
            ef_construct=int(q_raw.get("ef_construct", 100)),
            hnsw_ef=int(q_raw.get("hnsw_ef", 128)),
        ),
        llm=LlmConfig(
            base_url=str(llm_raw["base_url"]),
            model=str(llm_raw["model"]),
            temperature=float(llm_raw["temperature"]),
            timeout_seconds=float(llm_raw.get("timeout_seconds", 30.0)),
        ),
        llm_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY"),
    )
