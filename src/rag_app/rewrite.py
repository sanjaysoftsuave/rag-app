"""Query transforms — change what gets embedded, not what gets retrieved.

Both techniques here address the same mismatch: the question and the passage
that answers it are written differently. A user asks "why do I keep getting
cut off"; the document says "connections are terminated after 30 seconds of
inactivity". Dense retrieval has to bridge that gap in embedding space alone.

    REWRITE   Ask the LLM to restate the question in the vocabulary a document
              would use, then embed the restatement.

    HyDE      Ask the LLM to *invent* an answer, then embed that. The fake
              answer is usually wrong on facts, which does not matter: it is
              being used as a search key, and a wrong answer about refund
              windows still looks far more like a real refund passage than the
              question does. (Hypothetical Document Embeddings, Gao et al. 2022.)

WHAT THEY COST
--------------
An LLM call *before* retrieval, on every question — latency and money on the
path that used to be free, and a second place the model can go wrong. A bad
rewrite retrieves confidently wrong chunks, and the trace will show retrieval
"working" perfectly against a question the user never asked. That is why
`Answer.meta` records the transformed query: without it the retrieval list
becomes inexplicable.

Both are OFF by default. Neither is worth enabling without a gold set to prove
it helped — see evaluate.py.

FAILURE POLICY
--------------
If the transform call fails or returns nothing usable, fall back to the
original question rather than raising. A degraded search beats no answer, and
the fallback is recorded in `meta` so it is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_app.config import AppConfig

REWRITE_SYSTEM = (
    "Rewrite the user's question as a short statement using the vocabulary a "
    "reference document would use to answer it. Keep every specific term, name, "
    "number and identifier from the original — they are the strongest retrieval "
    "signal. Do not answer the question. Reply with the rewritten query and "
    "nothing else."
)

HYDE_SYSTEM = (
    "Write a short, plausible passage that would answer the user's question, as "
    "if excerpted from a reference document. Two or three sentences. Invent "
    "specifics freely — this text is used only as a search key and is never "
    "shown to anyone. Reply with the passage and nothing else."
)

MODES = ("off", "rewrite", "hyde")


@dataclass(frozen=True)
class QueryTransform:
    """What was searched for, and how it got that way."""

    original: str
    transformed: str
    mode: str
    failed: bool = False

    @property
    def changed(self) -> bool:
        return self.transformed != self.original

    def describe(self) -> str:
        if self.mode == "off":
            return "off"
        if self.failed:
            return f"{self.mode} (failed, fell back to the original question)"
        return self.mode


def _call(system: str, question: str, cfg: AppConfig, client=None) -> str:
    """One short completion. Shares the LLM config with generation."""
    if not cfg.llm_api_key:
        raise RuntimeError("no API key")
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout_seconds,
            default_headers={
                "HTTP-Referer": "https://localhost/rag-app",
                "X-Title": "rag-app documents",
            },
        )
    response = client.chat.completions.create(
        model=cfg.llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=cfg.llm.temperature,
    )
    return (response.choices[0].message.content or "").strip()


def transform_query(
    question: str,
    cfg: AppConfig,
    mode: str | None = None,
    *,
    transform_fn=None,
) -> QueryTransform:
    """Return what should be embedded for retrieval.

    `transform_fn(system, question, cfg) -> str` is the injection seam, the same
    idea as `ask(generate_fn=...)`: the suite exercises every branch here with
    no network.
    """
    mode = mode or cfg.retrieval.query_mode
    if mode not in MODES:
        raise ValueError(f"query_mode must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return QueryTransform(question, question, "off")

    system = REWRITE_SYSTEM if mode == "rewrite" else HYDE_SYSTEM
    caller = transform_fn or _call
    try:
        produced = (caller(system, question, cfg) or "").strip()
    except Exception:
        produced = ""

    if not produced:
        # Degraded search beats no answer — but say so, do not hide it.
        return QueryTransform(question, question, mode, failed=True)
    return QueryTransform(question, produced, mode)
