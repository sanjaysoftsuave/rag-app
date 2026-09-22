"""Command line entry point.

The browser UI is the primary surface — it is where documents are added and
the index is built. What survives here is what is genuinely easier without a
browser, or needed to launch one:

  ui       start the Streamlit app
  ask      answer one question and print the full retrieval trace
  chunks   preview how the corpus would be split, loading no models
  models   show the embedding + reranker registries and their rules
  eval     hit-rate@k, recall@k, MRR, rerank lift, refusal accuracy
  debug    label each failure: wrong text retrieved, or right text misused
  codes    rank your own open-coding categories by frequency x severity
  judge    measure the LLM judge against your own human labels
  compare  diff two saved eval snapshots - what a change actually bought
  agent    answer one question with the ReAct loop, printing every step
  arena    the same questions through ask() and through the agent
  atasks   trajectory-level agent evaluation against data/agent_tasks.yaml
  redteam  run the injection suite against a separate attack index

Ingest deliberately has no CLI command: building the index is a UI action, so
there is one place it happens rather than two that can drift apart.
"""

from __future__ import annotations

import argparse
import sys

from rag_app.chunking import chunk_docs
from rag_app.config import load_config
from rag_app.docs import load_docs
from rag_app.embed import MODEL_REGISTRY, Embedder, spec_for
from rag_app.filters import MetaFilter
from rag_app.pdfs import load_pdfs
from rag_app.pipeline import ask, format_answer
from rag_app.rerank import RERANKER_REGISTRY, build_reranker, reranker_spec


# The stand-in the dry-run arms use. REAL newlines: parse_action is
# line-oriented, and a literal backslash-n collapses the whole turn onto one
# line with no Action:, which reads as a parse failure and silently inflates
# the dry run's reported call count by 3x.
DRY_AGENT_TURN = "Thought: dry run\nAction: final_answer\nAction Input: (dry run)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_app",
        description="Ask my documents — dense retrieval + cross-encoder rerank + grounded LLM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ui", help="Launch the Streamlit app (add documents, ask, inspect)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument(
        "--headless", action="store_true", help="Do not open a browser window automatically"
    )

    p = sub.add_parser("ask", help="Ask a question against the index")
    p.add_argument("question")
    p.add_argument("--preset", default=None)
    p.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help="Metadata filter, repeatable (e.g. --filter source_type=pdf)",
    )
    p.add_argument("--quiet", action="store_true", help="Hide retrieve/rerank debug lists")

    p = sub.add_parser("chunks", help="Preview chunking without embedding anything")
    p.add_argument("--preset", default=None)
    p.add_argument("--all", action="store_true", help="Every configured preset")
    p.add_argument("--show", type=int, default=0, help="Print the first N chunks verbatim")

    sub.add_parser("models", help="Show the embedding and reranker registries")

    p = sub.add_parser("eval", help="Score retrieval and the refusal gate against a gold set")
    p.add_argument("--preset", default=None)
    p.add_argument("--gold", default=None, help="Gold set path (default data/gold.yaml)")
    p.add_argument(
        "--generate", action="store_true",
        help="Also call the LLM and score answer accuracy (one call per question)",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable summary")
    p.add_argument(
        "--judge", action="store_true",
        help="Also grade each answer with the judge model (1 call per answered question)",
    )
    p.add_argument(
        "--geval", action="store_true",
        help="G-Eval 1-5 rubric score, averaged over N samples (N calls per question)",
    )
    p.add_argument(
        "--ragas", action="store_true",
        help="Faithfulness, answer relevancy, context precision, context recall (4-5 calls)",
    )
    p.add_argument(
        "--snapshot", default=None, metavar="LABEL",
        help="Save the metrics to data/eval/LABEL.json for `compare`",
    )
    p.add_argument("--limit", type=int, default=0, help="Score only the first N questions")
    p.add_argument(
        "--note", default=None, metavar="TEXT",
        help="What you changed. Recorded in the snapshot and printed first by `compare`",
    )

    p = sub.add_parser(
        "debug",
        help="Label each failure as 'wrong text retrieved' vs 'right text, wrong answer'",
    )
    p.add_argument("--preset", default=None)
    p.add_argument("--gold", default=None)
    p.add_argument(
        "--generate", action="store_true",
        help="Call the LLM to resolve 'unconfirmed' into pass/generation",
    )
    p.add_argument("--show-pass", action="store_true", help="Also list questions that passed")

    p = sub.add_parser(
        "codes",
        help="Rank your open-coding categories by frequency x severity",
    )
    p.add_argument("--traces", default=None, help="Trace .jsonl (default the newest in data/traces)")
    p.add_argument("--labels", default=None, help="Labels YAML (default alongside the traces)")
    p.add_argument(
        "--init", action="store_true",
        help="Write a blank labelling sheet, one row per trace, and stop",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable ranking")

    p = sub.add_parser(
        "judge",
        help="Measure the LLM judge against the labels you wrote (agreement + kappa)",
    )
    p.add_argument("--traces", default=None, help="Trace .jsonl (default the newest)")
    p.add_argument("--labels", default=None, help="Labels YAML (default alongside the traces)")
    p.add_argument(
        "--init", action="store_true", help="Write a blank labelling sheet and stop"
    )
    p.add_argument("--json", action="store_true", help="Machine-readable summary")

    p = sub.add_parser(
        "compare",
        help="Diff two saved eval snapshots: what changed, and what it bought",
    )
    p.add_argument("before", help="Snapshot label (or path) recorded before the change")
    p.add_argument("after", help="Snapshot label (or path) recorded after it")
    p.add_argument("--json", action="store_true", help="Machine-readable diff")

    p = sub.add_parser("agent", help="Answer one question with the ReAct loop")
    p.add_argument("question")
    p.add_argument("--preset", default=None)
    p.add_argument(
        "--impl", default=None, choices=["plain", "langgraph"],
        help="Which loop implementation to run (default from config)",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--memory", action="store_true", help="Recall and record conversation memory")
    p.add_argument("--session", default="default", help="Memory session id")
    p.add_argument("--json", action="store_true", help="Machine-readable trajectory")
    p.add_argument("--quiet", action="store_true", help="Hide the trajectory")

    p = sub.add_parser(
        "arena",
        help="Run the gold questions through ask() and the agent, side by side",
    )
    p.add_argument("--preset", default=None)
    p.add_argument("--gold", default=None)
    p.add_argument(
        "--arms", default="workflow,agent",
        help="Comma-separated: workflow, agent, langgraph",
    )
    p.add_argument(
        "--generate", action="store_true",
        help="Actually call the LLM (without it a scripted stand-in exercises the harness)",
    )
    p.add_argument("--json", action="store_true", help="Machine-readable comparison")

    p = sub.add_parser("atasks", help="Trajectory-level agent evaluation")
    p.add_argument("--preset", default=None)
    p.add_argument("--tasks", default=None, help="Task YAML (default data/agent_tasks.yaml)")
    p.add_argument("--impl", default=None, choices=["plain", "langgraph"])
    p.add_argument("--generate", action="store_true", help="Actually call the LLM")
    p.add_argument("--json", action="store_true", help="Machine-readable summary")
    p.add_argument(
        "--snapshot", default=None, metavar="LABEL",
        help="Save the metrics to data/eval/LABEL.json for `compare`",
    )
    p.add_argument(
        "--note", default=None, metavar="TEXT",
        help="What you changed. Recorded in the snapshot and printed first by `compare`",
    )

    p = sub.add_parser(
        "redteam",
        help="Run the injection suite against a SEPARATE attack index",
    )
    p.add_argument(
        "--build-index", action="store_true",
        help="Build the attack index from data/redteam/corpus (never touches data/tickets)",
    )
    p.add_argument("--arm", default="both", choices=["defended", "undefended", "both"])
    p.add_argument("--attacks", default=None, help="Attack suite YAML")
    p.add_argument("--preset", default=None)
    p.add_argument("--generate", action="store_true", help="Actually call the LLM")
    p.add_argument("--json", action="store_true", help="Machine-readable summary")
    p.add_argument("--snapshot", default=None, metavar="LABEL")
    p.add_argument("--note", default=None, metavar="TEXT")

    return parser


def check_atasks_args(args) -> str | None:
    """Return an error message, or None. Pure: no config, no I/O.

    A separate function from `check_eval_args` because `eval --snapshot` without
    `--generate` is legitimate — the retrieval metrics cost nothing and are real.
    An agent dry run is not: every task stops `unparseable`, so the snapshot
    would record the harness's shape as if it were the agent's behaviour, and a
    later `compare` against it would be measuring nothing.
    """
    if getattr(args, "snapshot", None) and not getattr(args, "generate", False):
        return (
            "--snapshot saves a snapshot for `compare`, but without --generate no LLM is "
            "called, so every task stops 'unparseable' and the snapshot would record the "
            "harness's shape rather than the agent's behaviour. Add --generate, or drop "
            "--snapshot."
        )
    return None


def check_eval_args(args) -> str | None:
    """Return an error message, or None. Pure: no config, no I/O.

    argparse cannot express "--judge requires --generate", and the combination
    matters: without --generate there is no generated answer, so the judge would
    be grading the empty string. Deliberately NOT resolved by turning --generate
    on automatically — silently enabling a flag that spends money is exactly the
    affordance this check exists to prevent.
    """
    for flag in ("judge", "geval", "ragas"):
        if getattr(args, flag, False) and not getattr(args, "generate", False):
            return (
                f"--{flag} scores generated answers, and without --generate there is no "
                f"answer to score. Add --generate (one LLM call per question to generate, "
                f"plus the scoring calls on the judge model)."
            )
    if getattr(args, "limit", 0) < 0:
        return "--limit is a count of questions to score, so it cannot be negative."
    return None


def _load_gold_or_explain(args, cfg):
    from pathlib import Path

    from rag_app.evaluate import load_gold

    path = Path(args.gold) if args.gold else None
    try:
        return load_gold(cfg, path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return None


def cmd_eval(args: argparse.Namespace) -> int:
    import json as _json

    from rag_app.evaluate import evaluate, report_to_json
    from rag_app.llm import CallBudget, estimate_calls

    problem = check_eval_args(args)
    if problem:
        print(problem, file=sys.stderr)
        return 1

    cfg = load_config()
    gold = _load_gold_or_explain(args, cfg)
    if gold is None:
        return 1
    if args.limit:
        gold = gold[: args.limit]

    scoring = args.judge or args.geval or args.ragas
    budget = None
    if args.generate or scoring:
        answerable = [g for g in gold if g.answerable]
        estimate = estimate_calls(
            len(gold),
            generate=args.generate,
            judge=args.judge,
            geval=args.geval,
            ragas=args.ragas,
            geval_samples=cfg.evaluation.geval_samples,
            n_answered=len(answerable),
            n_with_reference=sum(1 for g in answerable if g.reference_answer),
            gen_model=cfg.llm.model,
            judge_model=cfg.evaluation.judge_model,
        )
        # Printed BEFORE the first call, to stderr so --json stays pipeable.
        print(estimate.describe(), file=sys.stderr)
        ceiling = cfg.evaluation.max_llm_calls
        if ceiling and estimate.total > ceiling:
            print(
                f"That exceeds evaluation.max_llm_calls ({ceiling}). Raise the ceiling in "
                f"config.yaml, or narrow the run with --limit N. Nothing was called.",
                file=sys.stderr,
            )
            return 1
        budget = CallBudget(limit=ceiling)

    try:
        report = evaluate(
            gold,
            preset=args.preset,
            config=cfg,
            embedder=Embedder(cfg.bi_encoder_model),
            reranker=build_reranker(cfg.cross_encoder_model),
            use_llm=args.generate,
            judge=args.judge,
            geval=args.geval,
            ragas=args.ragas,
            budget=budget,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        # A missing key or a missing index is a thing the user can fix. A stack
        # trace buries the one sentence that says how.
        print(str(exc), file=sys.stderr)
        return 1
    payload = report_to_json(report)
    print(payload if args.json else report.describe())

    if args.snapshot:
        from rag_app.before_after import make_snapshot, snapshot_path, write_snapshot

        path = snapshot_path(cfg, args.snapshot)
        write_snapshot(
            make_snapshot(
                args.snapshot, cfg, _json.loads(payload), gold, note=args.note or ""
            ),
            path,
        )
        print(f"Snapshot written to {path}", file=sys.stderr)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from rag_app.error_analysis import load_labels
    from rag_app.judge_validation import (
        load_traces,
        validate_judge,
        validation_to_json,
        write_labels_template_for,
    )

    cfg = load_config()
    traces_path, labels_path = _resolve_trace_paths(args, cfg)
    if traces_path is None:
        return 1

    try:
        if args.init:
            n = write_labels_template_for(traces_path, labels_path)
            print(f"Wrote {n} blank rows to {labels_path}")
            return 0
        traces = load_traces(traces_path)
        labels = load_labels(labels_path)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"about to judge {len(labels)} labelled traces on "
        f"{cfg.evaluation.judge_model} (~{len(labels)} LLM calls)",
        file=sys.stderr,
    )
    report = validate_judge(traces, labels, cfg)
    print(validation_to_json(report) if args.json else report.describe())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rag_app.before_after import (
        comparison_to_json,
        compare,
        load_snapshot,
        snapshot_path,
    )

    cfg = load_config()

    def resolve(name: str) -> Path:
        candidate = Path(name)
        return candidate if candidate.suffix == ".json" else snapshot_path(cfg, name)

    try:
        before = load_snapshot(resolve(args.before))
        after = load_snapshot(resolve(args.after))
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    result = compare(before, after)
    print(comparison_to_json(result) if args.json else result.describe())
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    from rag_app.evaluate import label_failures

    cfg = load_config()
    gold = _load_gold_or_explain(args, cfg)
    if gold is None:
        return 1
    try:
        report = label_failures(
            gold,
            preset=args.preset,
            config=cfg,
            embedder=Embedder(cfg.bi_encoder_model),
            reranker=build_reranker(cfg.cross_encoder_model),
            use_llm=args.generate,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(report.describe(show_pass=args.show_pass))
    return 0


def _resolve_trace_paths(args, cfg):
    """Work out which trace file and which labels file to use.

    Defaults to the newest .jsonl in data/traces and its `_labels.yaml` sibling,
    so the common case is a bare `codes` with no flags. Returns (traces, labels)
    or (None, None) after printing why it could not.
    """
    from pathlib import Path

    if args.traces:
        traces = Path(args.traces)
    else:
        trace_dir = cfg.tickets_dir.parent / "traces"
        found = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not found:
            print(
                f"No trace files in {trace_dir}. Error analysis reads a batch of real "
                f"answers; generate one first, then label it.",
                file=sys.stderr,
            )
            return None, None
        traces = found[-1]

    labels = Path(args.labels) if args.labels else traces.with_name(f"{traces.stem}_labels.yaml")
    return traces, labels


def cmd_codes(args: argparse.Namespace) -> int:
    from rag_app.error_analysis import (
        build_taxonomy,
        load_labels,
        load_trace_stubs,
        taxonomy_to_json,
        write_labels_template,
    )

    cfg = load_config()
    traces, labels_path = _resolve_trace_paths(args, cfg)
    if traces is None:
        return 1

    try:
        if args.init:
            stubs = load_trace_stubs(traces)
            write_labels_template(stubs, labels_path, source=traces.name)
            print(f"Wrote {len(stubs)} blank rows to {labels_path}")
            print(
                "\nRead each trace and write one honest sentence per row BEFORE inventing "
                "any category.\nThen run `python -m rag_app codes` to rank them."
            )
            return 0
        labels = load_labels(labels_path)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    tax = build_taxonomy(labels)
    print(taxonomy_to_json(tax) if args.json else tax.describe())
    return 0


def _agent_context(cfg, preset):
    """Open the store once and build the tool registry over it.

    The store is opened HERE and injected, never inside a tool: embedded Qdrant
    locks a directory for one process, so a tool calling open_store() while
    another handle is live fails on Windows.
    """
    from rag_app.pipeline import open_store
    from rag_app.tools import build_registry

    name = preset or cfg.default_preset
    store = open_store(cfg, name)
    embedder = Embedder(cfg.bi_encoder_model)
    reranker = build_reranker(cfg.cross_encoder_model)
    tools = build_registry(cfg, store=store, embedder=embedder, reranker=reranker)
    return store, embedder, reranker, tools


def _resolve_runner(impl: str):
    if impl == "langgraph":
        from rag_app.agent_langgraph import run_agent_langgraph

        return run_agent_langgraph
    from rag_app.agent import run_agent

    return run_agent


def cmd_agent(args: argparse.Namespace) -> int:
    import json as _json

    cfg = load_config()
    impl = args.impl or cfg.agent.implementation
    try:
        store, embedder, _, tools = _agent_context(cfg, args.preset)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    memory = None
    try:
        runner = _resolve_runner(impl)
        if args.memory:
            from rag_app.memory import MemoryStore, Turn

            memory = MemoryStore(cfg, embedder=embedder, session=args.session)
        result = runner(
            args.question, cfg, tools=tools, memory=memory, max_steps=args.max_steps
        )
        if memory is not None:
            memory.remember(Turn.now("user", args.question))
            memory.remember(Turn.now("agent", result.text))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if memory is not None:
            memory.close()
        store.close()

    if args.json:
        print(_json.dumps({
            "question": result.question,
            "answer": result.text,
            "sources": result.sources,
            "stop_reason": result.stop_reason,
            "failed": result.failed,
            "llm_calls": result.llm_calls,
            "tool_calls": result.tool_calls,
            "hallucinated_citations": result.hallucinated_citations,
            "steps": [
                {"index": s.index, "thought": s.thought, "tool": s.tool,
                 "input": s.tool_input, "observation": s.observation}
                for s in result.steps
            ],
        }, indent=2))
    else:
        print(result.text if args.quiet else result.describe())
    return 0


def cmd_arena(args: argparse.Namespace) -> int:
    from rag_app.compare import compare_arms, comparison_to_json

    cfg = load_config()
    gold = _load_gold_or_explain(args, cfg)
    if gold is None:
        return 1
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    try:
        store, embedder, reranker, tools = _agent_context(cfg, args.preset)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.generate:
        print(
            f"dry run: a scripted stand-in drives both arms. The real thing would cost "
            f"roughly {len(gold)} generation calls plus up to "
            f"{len(gold) * cfg.agent.max_steps} agent calls. Pass --generate.",
            file=sys.stderr,
        )
    try:
        report = compare_arms(
            gold, cfg, arms=arms, preset=args.preset, store=store,
            embedder=embedder, reranker=reranker, tools=tools,
            generate_fn=None if args.generate else (lambda q, c, k: "(dry run)"),
            llm_fn=None if args.generate else (lambda m, c: DRY_AGENT_TURN),
            generated=args.generate,
        )
    finally:
        store.close()
    print(comparison_to_json(report) if args.json else report.describe())
    return 0


def cmd_atasks(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rag_app.agent_eval import (
        agent_metrics,
        agent_gold_view,
        agent_report_to_json,
        evaluate_agent,
        load_agent_tasks,
    )

    problem = check_atasks_args(args)
    if problem:
        print(problem, file=sys.stderr)
        return 1

    cfg = load_config()
    try:
        tasks = load_agent_tasks(cfg, Path(args.tasks) if args.tasks else None)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        store, _, _, tools = _agent_context(cfg, args.preset)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not args.generate:
        print(
            f"dry run: no LLM is called, so every task stops as unparseable. The real "
            f"run costs up to {len(tasks) * cfg.agent.max_steps} calls. Pass --generate.",
            file=sys.stderr,
        )
    try:
        report = evaluate_agent(
            tasks, cfg, tools=tools,
            llm_fn=None if args.generate else (lambda m, c: "(dry run)"),
            runner=_resolve_runner(args.impl or cfg.agent.implementation),
            generated=args.generate,
        )
    finally:
        store.close()
    print(agent_report_to_json(report) if args.json else report.describe())

    if args.snapshot:
        # After the store is closed: embedded Qdrant holds a directory lock and
        # the snapshot write must not happen inside the finally.
        from rag_app.before_after import make_snapshot, snapshot_path, write_snapshot

        path = snapshot_path(cfg, args.snapshot)
        write_snapshot(
            make_snapshot(
                args.snapshot, cfg, agent_metrics(report), agent_gold_view(tasks),
                note=args.note or "",
            ),
            path,
        )
        print(f"Snapshot written to {path}", file=sys.stderr)
    return 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from rag_app.redteam import (
        attack_metrics,
        obedient_llm,
        attack_report_to_json,
        build_redteam_index,
        load_attacks,
        redteam_config,
        run_attacks,
        undefended,
    )

    cfg = load_config()
    rt_cfg = redteam_config(cfg)

    if args.build_index:
        # An ingest on the CLI, which this app otherwise forbids. It stays
        # because it is STRUCTURALLY incapable of touching the real corpus -
        # redteam_config redirects both tickets_dir and store_dir - and because
        # requiring a UI click to build an ATTACK index would make the exercise
        # unreproducible for anyone else.
        try:
            report = build_redteam_index(cfg, preset=args.preset)
        except (ValueError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(report.describe())
        print(f"\nAttack index built at {rt_cfg.store_dir}", file=sys.stderr)
        print(f"The real corpus at {cfg.tickets_dir} was not touched.", file=sys.stderr)
        return 0

    from pathlib import Path

    try:
        cases = load_attacks(cfg, Path(args.attacks) if args.attacks else None)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    arms = ["defended", "undefended"] if args.arm == "both" else [args.arm]
    if not args.generate:
        print(
            f"dry run: a scripted stand-in drives the agent, so these rates measure the "
            f"CODE, not the model. The real thing costs up to "
            f"{len(cases) * len(arms) * cfg.agent.max_steps} calls. Pass --generate.",
            file=sys.stderr,
        )

    reports = {}
    for arm in arms:
        arm_cfg = rt_cfg if arm == "defended" else undefended(rt_cfg)
        try:
            store, embedder, reranker, tools = _agent_context(arm_cfg, args.preset)
        except (FileNotFoundError, RuntimeError) as exc:
            print(
                f"{exc}\n\nBuild the attack index first: "
                f"python -m rag_app redteam --build-index",
                file=sys.stderr,
            )
            return 1
        try:
            reports[arm] = run_attacks(
                cases, arm_cfg, tools=tools, store=store,
                llm_fn=None if args.generate else obedient_llm(cases),
                profile=arm, generated=args.generate,
            )
        finally:
            store.close()

    if args.json:
        import json as _json

        print(_json.dumps(
            {arm: _json.loads(attack_report_to_json(r)) for arm, r in reports.items()},
            indent=2,
        ))
    else:
        for arm, report in reports.items():
            print(report.describe())
            print()
        if len(reports) == 2:
            before = reports["undefended"].injection_success_rate
            after = reports["defended"].injection_success_rate
            print(
                f"  injection success: {before:.1%} undefended -> {after:.1%} defended"
            )
            print(
                "  Read that beside the refusal rate: a defence that refuses everything"
            )
            print("  scores zero injections and is useless.")

    if args.snapshot:
        from rag_app.before_after import make_snapshot, snapshot_path, write_snapshot

        # Snapshot the config THAT ARM RAN WITH, not always the defended one:
        # otherwise both sides record identical defence switches and `compare`
        # reports "no configuration difference" on the A/B's own headline.
        arm = "defended" if "defended" in reports else args.arm
        arm_cfg = rt_cfg if arm == "defended" else undefended(rt_cfg)
        path = snapshot_path(cfg, args.snapshot)
        write_snapshot(
            make_snapshot(args.snapshot, arm_cfg, attack_metrics(reports[arm]), cases,
                          note=args.note or ""),
            path,
        )
        print(f"Snapshot written to {path}", file=sys.stderr)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config()
    flt = MetaFilter.parse(args.filter)
    answer = ask(
        args.question,
        preset=args.preset,
        config=cfg,
        embedder=Embedder(cfg.bi_encoder_model),
        reranker=build_reranker(cfg.cross_encoder_model),
        flt=flt or None,
    )
    print(format_answer(answer, verbose=not args.quiet))
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    cfg = load_config()
    names = sorted(cfg.chunk_presets) if args.all else [args.preset or cfg.default_preset]

    docs = load_docs(cfg.tickets_dir)
    pdf_docs, pdf_reports = load_pdfs(cfg.tickets_dir)
    parts = []
    if docs:
        parts.append(f"{len(docs)} documents (.md/.txt)")
    if pdf_reports:
        pages = sum(r.n_text_pages for r in pdf_reports)
        parts.append(f"{len(pdf_reports)} PDFs ({pages} pages of text)")
    print(f"{' and '.join(parts) or 'nothing'} in {cfg.tickets_dir}\n")
    for report in pdf_reports:
        if report.looks_scanned or report.encrypted:
            print(f"  !! {report.describe()}")

    def all_chunks(pc):
        return (
            chunk_docs(docs, pc.chunk_size, pc.overlap)
            + chunk_docs(pdf_docs, pc.chunk_size, pc.overlap, source_type="pdf")
        )

    header = f"{'preset':<10}{'size':<8}{'overlap':<10}{'chunks':<9}{'avg chars'}"
    print(header)
    print("-" * len(header))
    for name in names:
        pc = cfg.chunk_presets[name]
        chunks = all_chunks(pc)
        avg = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        print(f"{name:<10}{pc.chunk_size:<8}{pc.overlap:<10}{len(chunks):<9}{avg:.0f}")

    if args.show:
        for name in names:
            chunks = all_chunks(cfg.chunk_presets[name])
            print(f"\n{'='*72}\nPreset {name} — first {args.show} chunks\n{'='*72}")
            for c in chunks[: args.show]:
                print(f"\n--- {c.chunk_id} cited as [{c.source}]")
                print(c.text)
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    cfg = load_config()
    print("Embedding model registry\n")
    for name, spec in MODEL_REGISTRY.items():
        active = "  <-- active" if name == cfg.bi_encoder_model else ""
        print(f"{name}{active}")
        print(f"  dim={spec.dim}  asymmetric={spec.asymmetric}")
        print(f"  query prefix   : {spec.query_prefix!r}")
        print(f"  passage prefix : {spec.passage_prefix!r}")
        print(f"  {spec.note}\n")
    if cfg.bi_encoder_model not in MODEL_REGISTRY:
        spec = spec_for(cfg.bi_encoder_model)
        print(f"{cfg.bi_encoder_model}  <-- active, inferred")
        print(f"  query prefix   : {spec.query_prefix!r}")
        print(f"  passage prefix : {spec.passage_prefix!r}")
        print(f"  {spec.note}\n")
    print(
        "Using the wrong one degrades retrieval with NO error. That is why Embedder\n"
        "exposes encode_queries and encode_documents separately."
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Hand off to Streamlit, which needs to own the process.

    Launches `app.py` at the repo root rather than `ui.py` directly: app.py
    puts `src/` on sys.path first, so the UI starts whether or not the package
    happens to be pip-installed. Invoked as `sys.executable -m streamlit`, not
    via `streamlit.exe`, so a stale console-script shim cannot break it.
    """
    import subprocess

    from rag_app.config import ROOT

    script = ROOT / "app.py"
    if not script.exists():
        print(f"Cannot find {script}. The UI entry point is missing.", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(script),
        "--server.port", str(args.port),
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "Streamlit is not installed. Run:\n"
            '    pip install -e ".[ui]"      (or: pip install streamlit)',
            file=sys.stderr,
        )
        return 1


# Dispatch, in the same order as the parser blocks above. Module level rather
# than local to main() so `test_cli_help.py` can assert it covers every
# subcommand — a subparser added without a handler otherwise fails only at
# runtime, and only for the person who typed the new command.
_HANDLERS = {
    "ui": cmd_ui,
    "ask": cmd_ask,
    "chunks": cmd_chunks,
    "models": cmd_models,
    "eval": cmd_eval,
    "debug": cmd_debug,
    "codes": cmd_codes,
    "judge": cmd_judge,
    "compare": cmd_compare,
    "agent": cmd_agent,
    "arena": cmd_arena,
    "atasks": cmd_atasks,
    "redteam": cmd_redteam,
}


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot encode the arrow in the
    # summary line or the em dash inside DONT_KNOW — printing either raises
    # UnicodeEncodeError and takes down an otherwise correct answer.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.error(f"Unknown command {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
