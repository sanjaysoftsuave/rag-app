from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag_app.config import AppConfig
from rag_app.llm import MISSING_KEY, chat_messages
from rag_app.store import ScoredChunk

DONT_KNOW = "I don't know — that information is not in the provided documents."

# Matches [handbook.pdf] / [refund-policy.md] style citations. Dots, dashes and
# underscores are allowed, which is why a filename works as a citation token
# unchanged — see docs.citation_label for the fold that guarantees it.
CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._\-]*)\]")

# The citation contract, stated once.
#
# Extracted from build_prompt so the agent's system prompt can use the SAME
# words rather than a paraphrase. Two prompts that both "explain citations"
# but differ in wording are two different contracts, and only one of them
# would have been debugged against the [TIC-1001] failure recorded in
# CLAUDE.md. A test pins that agent.SYSTEM contains this verbatim.
CITATION_RULES = (
    "Every factual claim must be followed by a citation in square brackets containing "
    "the exact source label shown in that excerpt's header line, copied verbatim — "
    "for example [handbook.pdf] or [refund-policy.md]. "
    "The excerpt text may itself mention identifiers, reference numbers or document "
    "names; those are NOT source labels. Cite only the header label. "
    "Never cite a label that does not appear as an excerpt header above."
)


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
    # Every candidate the cross-encoder scored, sorted, INCLUDING the ones cut
    # below rerank_n. `reranked` is this list truncated to what the LLM saw.
    # Kept because the reranker computes these anyway and they are the only
    # record of what it demoted out of context — see rerank.rerank_all.
    reranked_all: list[ScoredChunk] = field(default_factory=list)


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

    Numbering the blocks [1], [2], [3] while asking for [handbook.pdf] teaches
    the model the wrong format by example — it will emit the numbers it can
    see. The label and the requested citation are deliberately identical.

    The prompt also has to say that identifiers *inside* the text are not
    labels: a PDF of support tickets contains strings like "Ticket CS-1001",
    and a model told to cite "the exact id" will happily cite that instead of
    the file it came from.
    """
    # The header is the label and nothing else. Any extra descriptor would
    # repeat information the label already carries, in a second format —
    # exactly the kind of near-miss that invites the model to cite the wrong one.
    blocks = [f"[{item.chunk.source}]\n{item.chunk.text}" for item in contexts]
    context = "\n\n".join(blocks)

    system = (
        "You answer questions using ONLY the excerpts provided. "
        f"{CITATION_RULES} "
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
    # Checked here, not left to llm.build_client, because generation is the one
    # caller that must fail loudly: the UI surfaces this message directly, and a
    # silent degrade would look like the model refusing rather than never being
    # asked. `build_client` raises the same message for every other caller.
    if not cfg.llm_api_key:
        raise RuntimeError(MISSING_KEY)
    return chat_messages(
        build_prompt(question, contexts), cfg.llm, cfg, client=client
    )


def cited_sources(text: str, contexts: list[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Split the model's citations into (grounded, hallucinated).

    A citation naming a source that was never in the context window is the
    clearest possible grounding failure, and it is invisible unless checked.
    In practice the common cause is not invention but confusion: the excerpt
    text mentions an identifier (a ticket number inside a PDF, say) and the
    model cites that instead of the header label it was given.
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
