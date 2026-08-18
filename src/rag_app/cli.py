from __future__ import annotations

import argparse
import sys

from rag_app.bm25 import BM25Index
from rag_app.chunking import build_chunks
from rag_app.config import AppConfig, load_config
from rag_app.embed import MODEL_REGISTRY, Embedder, spec_for
from rag_app.evaluate import (
    boundary_bleed,
    compare_retrieval_modes,
    evaluate,
    format_sweep,
    label_failures,
)
from rag_app.filters import MetaFilter
from rag_app.generate import generate_answer
from rag_app.ingest import run_ingest
from rag_app.pipeline import ask, format_answer, open_store
from rag_app.rerank import CrossEncoderReranker
from rag_app.tickets import load_all

# Loading a bi-encoder and a cross-encoder costs seconds and hundreds of MB.
# Any command that sweeps presets must reuse them instead of paying that per
# preset — which is exactly what the dependency-injection seam is for.
_MODELS: dict[str, object] = {}


def shared_models(cfg: AppConfig) -> tuple[Embedder, CrossEncoderReranker]:
    ekey = f"e:{cfg.bi_encoder_model}"
    rkey = f"r:{cfg.cross_encoder_model}"
    if ekey not in _MODELS:
        _MODELS[ekey] = Embedder(cfg.bi_encoder_model)
    if rkey not in _MODELS:
        _MODELS[rkey] = CrossEncoderReranker(cfg.cross_encoder_model)
    return _MODELS[ekey], _MODELS[rkey]  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_app",
        description="Ask my support tickets — dense retrieval + cross-encoder rerank + grounded LLM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Load tickets, chunk, embed, and persist")
    p.add_argument("--preset", default=None, help="Chunk preset name")
    p.add_argument("--all", action="store_true", help="Ingest every preset")

    p = sub.add_parser("ask", help="Ask a question against the store")
    p.add_argument("question")
    p.add_argument("--preset", default=None)
    p.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help="Metadata filter, repeatable (e.g. --filter product=API --filter customer_tier=pro)",
    )
    p.add_argument("--quiet", action="store_true", help="Hide retrieve/rerank debug lists")

    p = sub.add_parser("compare", help="Compare presets side by side (retrieval only by default)")
    p.add_argument("question")
    p.add_argument("--ingest-first", action="store_true")
    p.add_argument("--generate", action="store_true", help="Also call the LLM for each preset")

    p = sub.add_parser("eval", help="Score retrieval and the refusal gate against the gold set")
    p.add_argument("--preset", default=None)
    p.add_argument("--all", action="store_true", help="Evaluate every preset")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Replay the gate across thresholds to pick score_threshold from evidence",
    )
    p.add_argument(
        "--compare-retrieval",
        action="store_true",
        help="Before/after hit-rate@3: dense-only vs dense+BM25 hybrid (the one retrieval change)",
    )
    p.add_argument(
        "--k", type=int, default=3, help="Cutoff for --compare-retrieval's hit-rate@k (default 3)"
    )

    p = sub.add_parser("chunks", help="Inspect chunking for a preset without embedding")
    p.add_argument("--preset", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--show", type=int, default=0, help="Print the first N chunks verbatim")

    sub.add_parser("models", help="Show the embedding model registry and prefix rules")

    p = sub.add_parser("chat", help="Interactive session — models load once, then ask freely")
    p.add_argument("--preset", default=None)

    p = sub.add_parser(
        "debug",
        help="Label each gold failure as 'wrong document' vs 'right document, wrong answer'",
    )
    p.add_argument("--preset", default=None)
    p.add_argument(
        "--generate",
        action="store_true",
        help="Actually call the LLM to confirm generation-stage failures (costs one call per question)",
    )
    p.add_argument("--show-pass", action="store_true", help="Also print questions that passed")

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = load_config()
    names = sorted(cfg.chunk_presets) if args.all else [args.preset or cfg.default_preset]
    embedder, _ = shared_models(cfg)
    for name in names:
        print(run_ingest(preset=name, config=cfg, embedder=embedder).describe())
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config()
    flt = MetaFilter.parse(args.filter)
    embedder, reranker = shared_models(cfg)
    answer = ask(
        args.question,
        preset=args.preset,
        config=cfg,
        embedder=embedder,
        reranker=reranker,
        flt=flt or None,
    )
    print(format_answer(answer, verbose=not args.quiet))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare presets on the SAME question.

    Retrieval-only by default: the interesting difference is which chunks come
    back and how they are attributed, and that is free to inspect. Generation
    costs a call per preset, so it is opt-in.
    """
    cfg = load_config()
    presets = sorted(cfg.chunk_presets)
    embedder, reranker = shared_models(cfg)

    if args.ingest_first:
        for name in presets:
            print(run_ingest(preset=name, config=cfg, embedder=embedder).describe())
        print()

    print(f"Question: {args.question}\n")
    rows = []
    for name in presets:
        preset_cfg = cfg.chunk_presets[name]
        try:
            store = open_store(cfg, name)
        except FileNotFoundError as exc:
            print(f"Preset {name}: missing store — {exc}\n")
            continue

        answer = ask(
            args.question,
            preset=name,
            config=cfg,
            embedder=embedder,
            reranker=reranker,
            store=store,
            generate_fn=None if args.generate else _no_generate,
        )
        top = answer.reranked[0] if answer.reranked else None
        rows.append(
            {
                "preset": name,
                "config": preset_cfg.describe(),
                "top": top.chunk.source if top else "-",
                "score": f"{top.score:.4f}" if top else "n/a",
                "bleed": "yes" if top and top.chunk.metadata.get("bleed") else "no",
                "gate": answer.gate,
                "chars": str(len(top.chunk.text)) if top else "-",
            }
        )
        print("=" * 72)
        print(f"Preset {name}  ({preset_cfg.describe()})")
        print("=" * 72)
        print(format_answer(answer, verbose=True))
        print()

    if rows:
        print("=" * 72)
        print("SUMMARY")
        print("=" * 72)
        header = f"{'preset':<8}{'config':<22}{'top chunk':<12}{'ce score':<11}{'bleed':<8}{'chars':<8}{'gate'}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['preset']:<8}{r['config']:<22}{r['top']:<12}{r['score']:<11}"
                f"{r['bleed']:<8}{r['chars']:<8}{r['gate']}"
            )
    return 0


def _no_generate(question, contexts, cfg):
    return "(generation skipped — pass --generate to call the LLM)"


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_config()
    names = sorted(cfg.chunk_presets) if args.all else [args.preset or cfg.default_preset]
    embedder, reranker = shared_models(cfg)
    reports = []
    for name in names:
        try:
            store = open_store(cfg, name)
        except FileNotFoundError as exc:
            print(f"Preset {name}: {exc}\n")
            continue
        report = evaluate(store, cfg, name, embedder=embedder, reranker=reranker)
        reports.append(report)
        print(report.describe())
        print()
        if args.sweep:
            print(format_sweep(report))
            print()
        if args.compare_retrieval:
            bm25 = BM25Index.from_store(store)
            compare = compare_retrieval_modes(
                store, cfg, name, embedder=embedder, bm25=bm25, k=args.k
            )
            print(compare.describe())
            print()

    if len(reports) > 1:
        print("=" * 72)
        print("PRESET COMPARISON")
        print("=" * 72)
        header = f"{'preset':<8}{'strategy':<10}{'hit@K':<9}{'top1 bi':<10}{'top1 rerank':<13}{'refusals':<10}{'false refusals'}"
        print(header)
        print("-" * len(header))
        for r in reports:
            print(
                f"{r.preset:<8}{r.strategy:<10}{r.hit_at_k:<9.0%}{r.top1_retrieval:<10.0%}"
                f"{r.top1_rerank:<13.0%}{r.refusal_accuracy:<10.0%}{r.false_refusals}"
            )
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    cfg = load_config()
    names = sorted(cfg.chunk_presets) if args.all else [args.preset or cfg.default_preset]
    tickets = load_all(cfg.tickets_dir)
    print(f"{len(tickets)} tickets loaded from {cfg.tickets_dir}\n")

    header = f"{'preset':<8}{'strategy':<10}{'size':<8}{'overlap':<10}{'chunks':<9}{'avg chars':<12}{'boundary bleed'}"
    print(header)
    print("-" * len(header))
    for name in names:
        pc = cfg.chunk_presets[name]
        chunks = build_chunks(tickets, pc.strategy, pc.chunk_size, pc.overlap)
        bleeding, total = boundary_bleed(chunks)
        avg = sum(len(c.text) for c in chunks) / total if total else 0
        pct = f"{bleeding}/{total} ({100*bleeding/total:.0f}%)" if total else "-"
        print(
            f"{name:<8}{pc.strategy:<10}{pc.chunk_size:<8}{pc.overlap:<10}"
            f"{total:<9}{avg:<12.0f}{pct}"
        )

    if args.show:
        for name in names:
            pc = cfg.chunk_presets[name]
            chunks = build_chunks(tickets, pc.strategy, pc.chunk_size, pc.overlap)
            print(f"\n{'='*72}\nPreset {name} — first {args.show} chunks\n{'='*72}")
            for c in chunks[: args.show]:
                flag = "  <-- SPANS MULTIPLE TICKETS" if c.metadata.get("bleed") else ""
                print(f"\n--- {c.chunk_id} cited as [{c.source}]{flag}")
                print(c.text)
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    cfg = load_config()
    print("Embedding model registry\n")
    for name, spec in MODEL_REGISTRY.items():
        active = "  <-- active" if name == cfg.bi_encoder_model else ""
        print(f"{name}{active}")
        print(f"    dim={spec.dim}  asymmetric={spec.asymmetric}")
        if spec.query_prefix:
            print(f"    query prefix   : {spec.query_prefix!r}")
        if spec.passage_prefix:
            print(f"    passage prefix : {spec.passage_prefix!r}")
        print(f"    {spec.note}\n")

    if cfg.bi_encoder_model not in MODEL_REGISTRY:
        spec = spec_for(cfg.bi_encoder_model)
        print(f"Active model {cfg.bi_encoder_model} is not in the registry.")
        print(f"    inferred: {spec.note}, asymmetric={spec.asymmetric}\n")

    print("Pick a model on MTEB's *Retrieval* column, not the overall average:")
    print("  https://huggingface.co/spaces/mteb/leaderboard")
    print("\nSwitching models requires a re-ingest — vectors from different")
    print("models are not comparable, and meta.json will refuse to load a mismatch.")
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    cfg = load_config()
    name = args.preset or cfg.default_preset
    embedder, reranker = shared_models(cfg)
    try:
        store = open_store(cfg, name)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    bm25 = BM25Index.from_store(store) if cfg.retrieval.mode == "hybrid" else None
    report = label_failures(
        store, cfg, name,
        embedder=embedder, reranker=reranker,
        generate_fn=generate_answer if args.generate else None,
        use_llm=args.generate,
        bm25=bm25,
    )
    print(report.describe(show_pass=args.show_pass))
    if not args.generate:
        print("\n(retrieval-only pass — rerun with --generate to confirm which "
              "'unconfirmed' cases are real generation failures)")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from rag_app.repl import run

    cfg = load_config()
    embedder, reranker = shared_models(cfg)
    return run(cfg, embedder, reranker, args.preset or cfg.default_preset)


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
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "compare": cmd_compare,
        "eval": cmd_eval,
        "chunks": cmd_chunks,
        "models": cmd_models,
        "chat": cmd_chat,
        "debug": cmd_debug,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command {args.command}")
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
