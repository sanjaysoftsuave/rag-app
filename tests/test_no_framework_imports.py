"""The core path imports no agent framework, and this test is what keeps it true.

CLAUDE.md's rule used to be "no frameworks anywhere". Week 7 relaxed it to the
only version that survives a side-by-side comparison: **no frameworks on the
path that answers a question.** `agent_langgraph.py` and `memory_mem0.py` may
import langgraph and mem0; nothing else may, and neither may be reachable by
importing the pipeline, the CLI or the plain agent.

WHY A SUBPROCESS
----------------
The first version of this file inspected `sys.modules` in-process. It passed
alone and failed in the full suite, because `test_agent_langgraph.py` legitimately
imports langgraph first and `sys.modules` is global — so the assertion was really
"no test before me imported a framework", which is not the property worth having
and depends on collection order.

Each check therefore runs in a clean interpreter. It costs a process spawn per
test and buys an assertion that means what it says.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.util import find_spec

import pytest

# Every module that must stay framework-free. Ones that do not exist yet are
# skipped, so this file is correct before and after Week 7 lands.
CORE_MODULES = (
    "rag_app.pipeline",
    "rag_app.cli",
    "rag_app.generate",
    "rag_app.rewrite",
    "rag_app.evaluate",
    "rag_app.judge",
    "rag_app.ragas_metrics",
    "rag_app.judge_validation",
    "rag_app.before_after",
    "rag_app.error_analysis",
    "rag_app.agent",
    "rag_app.tools",
    "rag_app.memory",
    "rag_app.compare",
    "rag_app.agent_eval",
)

FRAMEWORK_PREFIXES = ("langchain", "langgraph", "llama_index", "mem0", "mcp", "fastmcp")

# The modules allowed to name a framework. They are the exhibit, not the
# backbone: all of them import lazily, inside function bodies, behind a
# _require().
EXEMPT_MODULES = (
    "rag_app.agent_langgraph",
    "rag_app.memory_mem0",
    "rag_app.mcp_server",
    "rag_app.mcp_client",
)

_PROBE = """
import importlib, sys, json
from importlib.util import find_spec

prefixes = {prefixes!r}
leaked = []
for name in {names!r}:
    if find_spec(name) is None:
        continue
    importlib.import_module(name)
for name in sys.modules:
    if any(name == p or name.startswith(p + ".") for p in prefixes):
        leaked.append(name)
print(json.dumps(sorted(set(leaked))))
"""


def _leaked_after_importing(names: tuple[str, ...]) -> list[str]:
    """Import `names` in a FRESH interpreter and report framework modules pulled in."""
    code = textwrap.dedent(
        _PROBE.format(prefixes=FRAMEWORK_PREFIXES, names=names)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_importing_the_core_modules_pulls_in_no_framework():
    """Checks sys.modules rather than grepping source: an indirect import three
    modules deep is exactly the one a grep would miss."""
    leaked = _leaked_after_importing(CORE_MODULES)
    assert leaked == [], (
        f"A framework reached the core import path: {leaked}. The rule is 'no "
        f"frameworks on the path that answers a question' — keep langgraph and "
        f"mem0 imports inside function bodies in {' or '.join(EXEMPT_MODULES)}."
    )


@pytest.mark.parametrize("name", EXEMPT_MODULES)
def test_the_exempt_modules_import_their_framework_lazily(name):
    """Importing the exhibit modules must also not pull the framework in.

    They are exempt from never mentioning a framework, not from importing it
    lazily. `import rag_app.agent_langgraph` has to succeed, and cost nothing, on
    a machine where the extra was never installed — otherwise the CLI could not
    even offer `--impl langgraph` as an option it then rejects.
    """
    if find_spec(name) is None:
        pytest.skip(f"{name} does not exist yet")
    leaked = _leaked_after_importing((name,))
    assert leaked == [], (
        f"{name} imported {leaked} at module scope. Move the import inside the "
        f"function body so the module stays importable without the extra."
    )


def test_the_probe_would_actually_catch_a_leak():
    """A guard nobody has seen fail is a guard nobody knows works."""
    if find_spec("langgraph") is None:
        pytest.skip("langgraph is not installed, so nothing could leak")
    leaked = _leaked_after_importing(("langgraph.graph",))
    assert any(name.startswith("langgraph") for name in leaked)
