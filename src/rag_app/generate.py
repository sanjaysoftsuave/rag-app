from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag_app.config import AppConfig
from rag_app.store import ScoredChunk

DONT_KNOW = "I don't know — that information is not in the provided documents."

# Matches [TIC-1001] style citations.
CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._\-]*)\]")


@dataclass
class Answer:
    text: str
    sources: list[str]
    best_score: float
    used_llm: bool
    retrieved: list[ScoredChunk]
    reranked: list[ScoredChunk]
    refused: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)
    gate: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def normalize(text: str) -> str:
    """Fold the punctuation an LLM silently rewrites."""
    folded = text.replace("—", "-").replace("–", "-")
    folded = folded.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", folded).strip().lower()


def is_refusal(text: str) -> bool:
    """Detect layer-2 refusal without demanding a byte-exact match.

    The canonical string contains an em dash and a curly-safe apostrophe, both
    of which models routinely rewrite. An `==` check would classify a correct
    refusal as a real answer and then attach sources to it.
    """
    folded = normalize(text)
    return folded.startswith(normalize(DONT_KNOW)[:20]) or folded.startswith("i don't know")


def build_prompt(question: str, contexts: list[ScoredChunk]) -> list[dict[str, str]]:
    """Label each excerpt with the exact token the model must cite.

    Numbering the blocks [1], [2], [3] while asking for [TIC-1001] teaches the
    model the wrong format by example — it will emit the numbers it can see.
    The label and the requested citation are deliberately identical here.
    """
    blocks = []
    for item in contexts:
        meta = item.chunk.metadata or {}
        descriptor = " | ".join(
            f"{k}={meta[k]}"
            for k in ("product", "category", "customer_tier", "status")
            if meta.get(k)
        )
        header = f"[{item.chunk.source}]"
        if descriptor:
            header += f" ({descriptor})"
        blocks.append(f"{header}\n{item.chunk.text}")
    context = "\n\n".join(blocks)

    system = (
        "You answer questions about customer support tickets using ONLY the excerpts provided. "
        "Every factual claim must be followed by a citation in square brackets containing the "
        "exact ticket id shown in the excerpt header, for example [TIC-1001]. "
        "Never cite a ticket id that does not appear in the excerpts. "
        "If several excerpts disagree, say so and cite each. "
        "If the excerpts do not contain the answer, reply with exactly this and nothing else: "
        f"{DONT_KNOW}"
    )
    user = f"Excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_answer(
    question: str,
    contexts: list[ScoredChunk],
    cfg: AppConfig,
    client: Any | None = None,
) -> str:
    if not cfg.llm_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and set your key."
        )
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.timeout_seconds,
            default_headers={
                "HTTP-Referer": "https://localhost/rag-app",
                "X-Title": "rag-app support tickets",
            },
        )
    response = client.chat.completions.create(
        model=cfg.llm.model,
        messages=build_prompt(question, contexts),
        temperature=cfg.llm.temperature,
    )
    return (response.choices[0].message.content or "").strip()


def cited_sources(text: str, contexts: list[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Split the model's citations into (grounded, hallucinated).

    A citation naming a ticket that was never in the context window is the
    clearest possible grounding failure, and it is invisible unless checked.
    """
    available = {item.chunk.source for item in contexts}
    grounded: list[str] = []
    invented: list[str] = []
    for match in CITATION_RE.findall(text):
        if match in available:
            if match not in grounded:
                grounded.append(match)
        elif match not in invented:
            invented.append(match)
    return grounded, invented


def sources_from_contexts(contexts: list[ScoredChunk]) -> list[str]:
    seen: list[str] = []
    for item in contexts:
        if item.chunk.source not in seen:
            seen.append(item.chunk.source)
    return seen
