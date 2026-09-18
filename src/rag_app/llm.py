"""The one place an OpenAI-compatible client is constructed, and a call is made.

WHY THIS MODULE EXISTS
----------------------
The client-construction block was duplicated byte-for-byte in `generate.py` and
`rewrite.py`. Evaluation adds six more call sites (the judge, G-Eval, and four
RAGAS prompts) and the agent adds a seventh, so the duplication was about to go
from "mildly annoying" to "eight places to fix when the attribution headers or
the timeout policy change".

Everything here is deliberately dumb: build a client, send messages, return the
text. No retries, no streaming, no caching. Policy — which model, what to do
when it fails, whether to call at all — belongs to the caller, because the right
answer differs per caller: `generate_answer` raises on a missing key because the
UI depends on generation failing loudly, while `transform_query` swallows the
same failure and degrades to the original question.

THE INJECTION SEAM IS WHY THE SUITE IS OFFLINE
-----------------------------------------------
`build_client` returns an injected client untouched, and does so BEFORE the API
key check. Two consequences, both load-bearing:

  * `openai` is imported only in the `client is None` branch, so a test that
    injects a fake never imports the SDK.
  * a test never needs an API key. An injected client already carries whatever
    auth it needs; demanding a key to use a fake would be theatre.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag_app.config import AppConfig, LlmConfig

# OpenRouter attribution headers. They identify the app in OpenRouter's rankings
# and have no effect on any other OpenAI-compatible endpoint.
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://localhost/rag-app",
    "X-Title": "rag-app documents",
}

MISSING_KEY = "OPENROUTER_API_KEY is missing. Copy .env.example to .env and set your key."


def build_client(cfg: AppConfig, llm: LlmConfig, client: Any | None = None) -> Any:
    """Return `client` if given, else construct one from `llm` + `cfg.llm_api_key`.

    `llm` is passed separately from `cfg` rather than read off it, because the
    judge deliberately runs on a *different* `LlmConfig` (a stronger model, a
    longer timeout) while sharing the same API key and the same account.
    """
    if client is not None:
        return client
    if not cfg.llm_api_key:
        raise RuntimeError(MISSING_KEY)
    from openai import OpenAI

    return OpenAI(
        api_key=cfg.llm_api_key,
        base_url=llm.base_url,
        timeout=llm.timeout_seconds,
        default_headers=DEFAULT_HEADERS,
    )


def chat_messages(
    messages: list[dict[str, str]],
    llm: LlmConfig,
    cfg: AppConfig,
    *,
    client: Any | None = None,
    temperature: float | None = None,
) -> str:
    """One completion from a full message list. Returns stripped text."""
    client = build_client(cfg, llm, client)
    response = client.chat.completions.create(
        model=llm.model,
        messages=messages,
        temperature=llm.temperature if temperature is None else temperature,
    )
    return (response.choices[0].message.content or "").strip()


def chat_once(
    system: str,
    user: str,
    llm: LlmConfig,
    cfg: AppConfig,
    *,
    client: Any | None = None,
    temperature: float | None = None,
) -> str:
    """The common case: one system message, one user message."""
    return chat_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        llm,
        cfg,
        client=client,
        temperature=temperature,
    )


def extract_json(text: str) -> dict | list | None:
    """Pull a JSON object or array out of a model reply, or return None.

    Models wrap JSON in ```json fences and preface it with "Here is my
    evaluation:". Two attempts, in order:

      1. parse the whole string
      2. slice from the first brace/bracket to the matching last one, and retry

    and then give up. Deliberately NO regex that guesses at individual fields:
    a scorer that half-parses a malformed reply produces a number that looks
    like a measurement and is not one. Returning None lets the caller record an
    honest "unscored" instead.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, (dict, list)) else None

    # Try whichever delimiter opens EARLIEST, not objects first. In
    # `prose [{"index": 1}, {"index": 2}] more` the first "{" belongs to an
    # element of the array, so preferring objects would return one verdict where
    # the caller asked for a list of them — silently scoring fewer contexts than
    # were judged, which is precisely the failure RAGAS context precision has to
    # catch rather than commit.
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append((start, text[start : end + 1]))

    for _, blob in sorted(candidates):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Spending: estimated before, capped during
# ---------------------------------------------------------------------------


@dataclass
class CallBudget:
    """A hard cap on LLM calls, checked before each one.

    Mutable on purpose — it is threaded through a whole evaluation run and its
    entire job is to accumulate. `limit=0` means unlimited.

    The policy this enables is "degrade, do not raise": a scorer that cannot
    afford another call records `unscored` for the remaining questions and says
    so in the report, rather than dying halfway through and losing the work
    already paid for.
    """

    limit: int = 0
    used: int = 0

    def spend(self, n: int = 1) -> bool:
        """Reserve `n` calls. False (and nothing reserved) if that would exceed."""
        if self.limit and self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def exhausted(self) -> bool:
        return bool(self.limit) and self.used >= self.limit

    def describe(self) -> str:
        if not self.limit:
            return f"{self.used} LLM calls (no cap)"
        return f"{self.used}/{self.limit} LLM calls"


@dataclass(frozen=True)
class CallEstimate:
    """What a run is about to cost, printed before the first call is made."""

    n_questions: int
    generation: int = 0
    judge: int = 0
    geval: int = 0
    ragas: int = 0
    gen_model: str = ""
    judge_model: str = ""

    @property
    def total(self) -> int:
        return self.generation + self.judge + self.geval + self.ragas

    @property
    def on_judge_model(self) -> int:
        return self.judge + self.geval + self.ragas

    def describe(self) -> str:
        parts = []
        if self.generation:
            parts.append(f"{self.generation} generation")
        if self.judge:
            parts.append(f"{self.judge} judge")
        if self.geval:
            parts.append(f"{self.geval} G-Eval")
        if self.ragas:
            parts.append(f"{self.ragas} RAGAS")
        breakdown = ", ".join(parts) or "none"
        line = (
            f"about to make ~{self.total} LLM calls for {self.n_questions} questions "
            f"({breakdown})"
        )
        if self.on_judge_model and self.judge_model:
            line += (
                f"\n  ~{self.generation} on {self.gen_model}, "
                f"~{self.on_judge_model} on {self.judge_model}"
            )
        return line


def estimate_calls(
    n_questions: int,
    *,
    generate: bool = False,
    judge: bool = False,
    geval: bool = False,
    ragas: bool = False,
    geval_samples: int = 5,
    n_answered: int | None = None,
    n_with_reference: int = 0,
    gen_model: str = "",
    judge_model: str = "",
) -> CallEstimate:
    """Worst-case call count, computed without making any.

    `n_answered` is how many questions are expected to reach the LLM — the score
    gate refuses the rest for free. It defaults to `n_questions` because the
    honest pre-flight number is the one that cannot surprise you upwards.

    Deliberately pure so a test can pin the arithmetic offline. RAGAS is 4 calls
    per answered question (2 faithfulness + 1 relevancy + 1 context precision)
    plus 1 more for each question that has a reference answer to recall against.
    """
    answered = n_questions if n_answered is None else n_answered
    return CallEstimate(
        n_questions=n_questions,
        generation=n_questions if generate else 0,
        judge=answered if judge else 0,
        geval=answered * geval_samples if geval else 0,
        ragas=(answered * 4 + n_with_reference) if ragas else 0,
        gen_model=gen_model,
        judge_model=judge_model,
    )
