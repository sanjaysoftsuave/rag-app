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

    return parser


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
    from rag_app.evaluate import evaluate, report_to_json

    cfg = load_config()
    gold = _load_gold_or_explain(args, cfg)
    if gold is None:
        return 1
    report = evaluate(
        gold,
        preset=args.preset,
        config=cfg,
        embedder=Embedder(cfg.bi_encoder_model),
        reranker=build_reranker(cfg.cross_encoder_model),
        use_llm=args.generate,
    )
    print(report_to_json(report) if args.json else report.describe())
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    from rag_app.evaluate import label_failures

    cfg = load_config()
    gold = _load_gold_or_explain(args, cfg)
    if gold is None:
        return 1
    report = label_failures(
        gold,
        preset=args.preset,
        config=cfg,
        embedder=Embedder(cfg.bi_encoder_model),
        reranker=build_reranker(cfg.cross_encoder_model),
        use_llm=args.generate,
    )
    print(report.describe(show_pass=args.show_pass))
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
    handlers = {
        "ui": cmd_ui,
        "ask": cmd_ask,
        "chunks": cmd_chunks,
        "models": cmd_models,
        "eval": cmd_eval,
        "debug": cmd_debug,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
