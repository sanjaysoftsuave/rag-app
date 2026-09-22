"""Naming what went wrong on a trajectory, automatically.

THE SIBLING MODULE, AND THE DIVISION OF LABOUR
-----------------------------------------------
`error_analysis.py` is HUMAN open coding over ANSWERS. It can see that an answer
is subtly wrong, that a citation is technically right and useless, that a
confident sentence is bluffing — judgements no rule can make.

This module is AUTOMATIC classification over TRAJECTORIES. It can see only what
is structurally observable in an `AgentResult`: which tools ran, in what order,
what the step said, how the run ended. It cannot tell a good answer from a bad
one and does not try.

Both exist because neither can do the other's job. Week 5 reads answers; this
reads routes.

FLAGS, NOT A LABEL
------------------
`classify_failures()` returns a LIST of flags, not one verdict, for three
reasons:

  1. The modes co-occur, and the co-occurrence IS the finding. A trajectory that
     loops and then exhausts `max-steps` exhibits two modes; collapsing them
     would need a precedence order, that order would be a guess, and the guess
     would hide the cause behind the symptom.
  2. The headline metric needs an *any-of* predicate. "Right answer, wrong path"
     is `success and any(flag in PATH_MODES)` — a single label cannot express
     that without a catch-all bucket.
  3. `error_analysis.py` already owns single-label semantics for human coding.
     Two modules with the same shape and different meanings get confused.

So there is no `primary` and no precedence order. Ranking for "top failure" is
by COUNT across the corpus, following `error_analysis.build_taxonomy`, and the
report says out loud that the rows overlap and therefore sum above the number of
failing trajectories.

WHAT THIS CANNOT SEE, STATED PLAINLY
-------------------------------------
The commonest form of "made-up input" is a plausible natural-language search
string the model invented — and it has NO OBSERVABLE SIGNAL. A well-chosen query
and a fabricated one are the same kind of string. `invented-input` therefore
catches only the structural cases (a forged citation, a hallucinated tool name,
an identifier that appeared nowhere) and UNDER-COUNTS, always.

That is a property of what a trajectory records, not a gap in the rules. Fixing
it would mean asking a model whether each tool input was grounded — which is
`judge.py`'s job, not a classifier's, and would make this module cost money.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_app.agent import BUDGET_STOPS, REFUSAL_STOPS
from rag_app.generate import normalize

FAILURE_MODES: dict[str, str] = {
    "loop": "called the same tool with the same input more than once",
    "wrong-tool": "skipped a tool the task expects, or used one it forbids",
    "wrong-sequence": "used the expected tools, but not in the expected order",
    "invented-input": "cited, called or looked up something that was never there",
    "quiet-give-up": "refused an answerable task instead of answering it",
    "budget-exhausted": "ran out of a budget mid-thought rather than deciding to stop",
    "step-target-missed": "finished outside the step window the task allows",
}

# Modes with no provable signal — fired from a proxy, and honest about it.
INFERRED_MODES = frozenset({"invented-input"})

# The modes that make a ROUTE bad. `quiet-give-up` is excluded on purpose: it is
# an outcome problem (the answer is missing), not a route problem, and folding it
# in would make "right answer, wrong path" unsatisfiable by construction.
PATH_MODES = (
    "loop",
    "wrong-tool",
    "wrong-sequence",
    "invented-input",
    "budget-exhausted",
    "step-target-missed",
)

# Identifier-shaped tokens: ticket ids, error codes, filenames. Deliberately
# narrow — a rule that flagged ordinary words would fire on every search.
_IDENTIFIER = re.compile(r"\b(?:[A-Z]{2,}-\d+|[A-Za-z0-9_-]+\.(?:pdf|md|txt|csv|json))\b")


@dataclass(frozen=True)
class FailureFlag:
    mode: str
    evidence: str  # one sentence naming the step or the number that fired it
    proven: bool = True  # False -> inferred from a proxy; see INFERRED_MODES

    def describe(self) -> str:
        tag = "" if self.proven else " (inferred)"
        return f"{self.mode}{tag}: {self.evidence}"


def _loop(result) -> FailureFlag | None:
    if result.stop_reason == "repeated-action":
        return FailureFlag("loop", result.meta.get("stop_detail", "stopped on a repeat"))
    seen: dict[tuple[str, str], list[int]] = {}
    for step in result.steps:
        if not step.tool:
            continue
        seen.setdefault((step.tool, normalize(step.tool_input)), []).append(step.index)
    for (tool, _), indexes in seen.items():
        if len(indexes) > 1:
            where = ", ".join(str(i) for i in indexes)
            return FailureFlag(
                "loop", f"called {tool} with the same input at steps {where}"
            )
    return None


def _wrong_tool(task, result) -> FailureFlag | None:
    used = [s.tool for s in result.steps if s.tool]
    missing = [t for t in getattr(task, "expect_tools", []) or [] if t not in used]
    forbidden = [t for t in getattr(task, "forbid_tools", []) or [] if t in used]
    if missing:
        return FailureFlag(
            "wrong-tool",
            f"expected {', '.join(missing)}; never called it "
            f"(used {', '.join(used) or 'nothing'})",
        )
    if forbidden:
        return FailureFlag(
            "wrong-tool", f"called {', '.join(forbidden)}, which the task forbids"
        )
    return None


def _invented_input(result) -> FailureFlag | None:
    if result.hallucinated_citations:
        return FailureFlag(
            "invented-input",
            f"cited {', '.join(result.hallucinated_citations)}, which no tool returned",
        )
    for step in result.steps:
        if not step.ok and step.observation.startswith("Unknown tool"):
            return FailureFlag(
                "invented-input", f"step {step.index} called a tool that does not exist"
            )

    # The inferred rule: an identifier in a tool input that appeared in no
    # earlier observation and is not in the question. Narrow on purpose.
    question_tokens = set(_IDENTIFIER.findall(result.question))
    seen_tokens: set[str] = set()
    for step in result.steps:
        for token in _IDENTIFIER.findall(step.tool_input or ""):
            if token not in question_tokens and token not in seen_tokens:
                return FailureFlag(
                    "invented-input",
                    f"step {step.index} looked up {token!r}, which appears in no "
                    f"earlier observation and not in the question",
                    proven=False,
                )
        seen_tokens.update(_IDENTIFIER.findall(step.observation or ""))
    return None


def _step_target(task, result) -> FailureFlag | None:
    steps = len(result.steps)
    max_steps = getattr(task, "max_steps", 0) or 0
    min_steps = getattr(task, "min_steps", 0) or 0
    if max_steps and steps > max_steps:
        return FailureFlag(
            "step-target-missed", f"took {steps} steps; the task allows {max_steps}"
        )
    if min_steps and steps < min_steps:
        # A multi-hop task answered in one step probably answered from the
        # model's own weights — right answer, wrong route.
        return FailureFlag(
            "step-target-missed",
            f"took {steps} steps; the task expects at least {min_steps}, so it "
            f"reached an answer without the work the task describes",
        )
    return None


def classify_failures(task, result) -> list[FailureFlag]:
    """Every failure mode this trajectory exhibits. Pure, duck-typed, no I/O.

    `task` needs `expect_tools`, `forbid_tools`, `expect_sequence`,
    `sequence_match`, `max_steps`, `min_steps`, `answerable`; `result` needs
    `steps`, `stop_reason`, `hallucinated_citations`, `meta`, `question`. Both
    are read with `getattr(..., default)` so a partial stand-in works in a test.
    """
    flags: list[FailureFlag] = []

    loop = _loop(result)
    if loop:
        flags.append(loop)

    wrong_tool = _wrong_tool(task, result)
    if wrong_tool:
        flags.append(wrong_tool)

    expected = list(getattr(task, "expect_sequence", []) or [])
    if expected:
        from rag_app.agent_eval import matches_sequence

        used = [s.tool for s in result.steps if s.tool]
        mode = getattr(task, "sequence_match", "subsequence")
        if not matches_sequence(used, expected, mode):
            flags.append(
                FailureFlag(
                    "wrong-sequence",
                    f"expected {' -> '.join(expected)} ({mode}); "
                    f"got [{', '.join(used) or 'nothing'}]",
                )
            )

    invented = _invented_input(result)
    if invented:
        flags.append(invented)

    if getattr(task, "answerable", True) and result.stop_reason in REFUSAL_STOPS:
        flags.append(
            FailureFlag(
                "quiet-give-up",
                f"stopped at {result.stop_reason} on a task the corpus can answer",
            )
        )

    if result.stop_reason in BUDGET_STOPS:
        flags.append(
            FailureFlag(
                "budget-exhausted",
                result.meta.get("stop_detail", f"stopped at {result.stop_reason}"),
            )
        )

    target = _step_target(task, result)
    if target:
        flags.append(target)

    return flags


def rank_modes(all_flags: list[list[FailureFlag]]) -> list[tuple[str, int]]:
    """Modes by how many trajectories exhibit them, most first.

    Counts TRAJECTORIES, not flags: a mode that fires twice on one run is one
    broken run, and ranking by flag count would promote whichever mode happens
    to be chatty. Ties break by the order in FAILURE_MODES so the table is
    stable across runs.
    """
    counts: dict[str, int] = {}
    for flags in all_flags:
        for mode in {f.mode for f in flags}:
            counts[mode] = counts.get(mode, 0) + 1
    order = list(FAILURE_MODES)
    return sorted(counts.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
