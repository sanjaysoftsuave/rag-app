"""Interactive session — load the models once, ask many questions.

Every `ask` invocation from the shell pays ~10s reloading a bi-encoder and a
cross-encoder from disk. That cost is fixed per process, not per question, so
a REPL amortises it to zero after the first prompt.

State (preset, filters, verbosity) lives in `Session` rather than in the loop,
so the command handling is testable without stdin. `run()` is the only part
that touches the terminal.

Note on the word "chat": each question is retrieved independently. There is no
conversational memory, so a follow-up like "what about Pro?" will not inherit
the previous question's subject — ask it in full. Resolving follow-ups properly
needs query rewriting (condensing the history into a standalone question), which
costs an extra LLM call per turn and is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_app.bm25 import BM25Index
from rag_app.config import AppConfig
from rag_app.filters import MetaFilter
from rag_app.generate import Answer
from rag_app.pipeline import ask, format_answer, open_store

BANNER = """\
rag-app interactive session
Models stay loaded, so every question after the first is fast.
Type a question, or /help for commands. /exit to leave."""

HELP = """\
  /help                     show this
  /preset <name>            switch chunk preset (reopens the store)
  /filter <field=value> ..  set metadata filters (replaces existing)
  /filter                   show current filters
  /nofilter                 clear all filters
  /verbose  /quiet          toggle the retrieve/rerank dump
  /show <TIC-1001>          print a ticket in full, to check an answer
  /last                     re-print the last answer, verbosely
  /config                   show preset, backend, threshold, filters
  /exit                     quit (Ctrl+C also works)"""


class Quit(Exception):
    """Raised by /exit to unwind the loop."""


@dataclass
class Session:
    cfg: AppConfig
    embedder: Any
    reranker: Any
    preset: str
    flt: MetaFilter = field(default_factory=MetaFilter)
    verbose: bool = False
    last: Answer | None = None
    # Same injection seam as ask()/run_ingest(). Without it this class is only
    # testable against a live API, which the suite must never touch.
    generate_fn: Any | None = None
    _stores: dict[str, Any] = field(default_factory=dict)
    _bm25: dict[str, BM25Index] = field(default_factory=dict)
    _tickets: dict[str, Any] | None = None

    # -- store handling ------------------------------------------------------

    def store(self):
        """Open each preset's store at most once per session."""
        if self.preset not in self._stores:
            self._stores[self.preset] = open_store(self.cfg, self.preset)
        return self._stores[self.preset]

    def bm25(self) -> BM25Index | None:
        """Build each preset's BM25 index at most once — same caching reason
        as the store itself. Only needed in hybrid mode."""
        if self.cfg.retrieval.mode != "hybrid":
            return None
        if self.preset not in self._bm25:
            self._bm25[self.preset] = BM25Index.from_store(self.store())
        return self._bm25[self.preset]

    def close(self) -> None:
        for store in self._stores.values():
            closer = getattr(store, "close", None)
            if closer is not None:
                closer()
        self._stores.clear()
        self._bm25.clear()

    def tickets(self) -> dict[str, Any]:
        if self._tickets is None:
            from rag_app.tickets import load_all

            self._tickets = {t.ticket_id: t for t in load_all(self.cfg.tickets_dir)}
        return self._tickets

    # -- the two things a line can be ---------------------------------------

    def answer(self, question: str) -> Answer:
        result = ask(
            question,
            preset=self.preset,
            config=self.cfg,
            embedder=self.embedder,
            reranker=self.reranker,
            store=self.store(),
            flt=self.flt or None,
            generate_fn=self.generate_fn,
            bm25=self.bm25(),
        )
        self.last = result
        return result

    def command(self, line: str) -> str:
        """Handle a /command. Returns text to print."""
        parts = line.strip().split()
        name, args = parts[0].lower(), parts[1:]

        if name in ("/exit", "/quit", "/q"):
            raise Quit

        if name == "/help":
            return HELP

        if name == "/config":
            return (
                f"  preset    {self.preset} ({self.cfg.chunk_presets[self.preset].describe()})\n"
                f"  backend   {self.cfg.backend}\n"
                f"  retrieval {self.cfg.retrieval.mode}\n"
                f"  model     {self.cfg.bi_encoder_model}\n"
                f"  threshold {self.cfg.score_threshold} ({self.cfg.rerank_score_scale})\n"
                f"  K -> N    {self.cfg.retrieve_k} -> {self.cfg.rerank_n}\n"
                f"  filters   {self.flt.describe()}"
            )

        if name == "/preset":
            if not args:
                return f"Current preset: {self.preset}. Available: {sorted(self.cfg.chunk_presets)}"
            wanted = args[0]
            if wanted not in self.cfg.chunk_presets:
                return f"Unknown preset {wanted!r}. Available: {sorted(self.cfg.chunk_presets)}"
            self.preset = wanted
            try:
                self.store()
            except FileNotFoundError as exc:
                return f"Preset {wanted} has no store yet — {exc}"
            return f"Preset -> {wanted} ({self.cfg.chunk_presets[wanted].describe()})"

        if name == "/filter":
            if not args:
                return f"Filters: {self.flt.describe()}"
            try:
                self.flt = MetaFilter.parse(args)
            except ValueError as exc:
                return f"Bad filter: {exc}"
            return f"Filters -> {self.flt.describe()}"

        if name == "/nofilter":
            self.flt = MetaFilter()
            return "Filters cleared."

        if name in ("/verbose", "/quiet"):
            self.verbose = name == "/verbose"
            return f"Verbose {'on' if self.verbose else 'off'}."

        if name == "/last":
            if self.last is None:
                return "Nothing asked yet."
            return format_answer(self.last, verbose=True)

        if name == "/show":
            if not args:
                return "Usage: /show TIC-1001"
            from rag_app.tickets import render_ticket

            ticket = self.tickets().get(args[0])
            if ticket is None:
                return f"No ticket {args[0]!r} in {self.cfg.tickets_dir}"
            return render_ticket(ticket)

        return f"Unknown command {name}. /help for the list."


def _summarise(answer: Answer) -> str:
    """One compact status line under each answer."""
    score = "n/a" if answer.best_score == float("-inf") else f"{answer.best_score:.3f}"
    bits = [
        f"gate={answer.gate}",
        f"score={score}",
        f"llm={'yes' if answer.used_llm else 'no'}",
    ]
    line = "  " + "  ".join(bits)
    if answer.hallucinated_citations:
        line += f"\n  !! invented citations: {', '.join(answer.hallucinated_citations)}"
    return line


def run(cfg: AppConfig, embedder, reranker, preset: str) -> int:
    session = Session(cfg=cfg, embedder=embedder, reranker=reranker, preset=preset)

    # Open the store before the banner so a missing one fails immediately with
    # the ingest hint, rather than after the user has typed a question.
    session.store()

    print(BANNER)
    print(
        f"\npreset {session.preset} · {cfg.backend} · threshold {cfg.score_threshold}\n"
    )

    while True:
        try:
            line = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            try:
                print(session.command(line))
            except Quit:
                break
            print()
            continue

        try:
            answer = session.answer(line)
        except Exception as exc:  # keep the session alive on any failure
            print(f"  error: {type(exc).__name__}: {exc}\n")
            continue

        print(f"\n{answer.text}")
        if answer.sources:
            print(f"  sources: {', '.join(answer.sources)}")
        print(_summarise(answer))
        if session.verbose:
            print()
            print(format_answer(answer, verbose=True))
        print()

    session.close()
    print("bye")
    return 0
