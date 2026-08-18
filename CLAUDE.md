# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The checked-in `.venv/` is **broken** — its `pyvenv.cfg` points at `C:\Users\Softsuave\...`, a base interpreter that does not exist on this machine, so `.venv\Scripts\python.exe` fails with "did not find executable". Recreate it before doing anything:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # set OPENROUTER_API_KEY
```

As a stopgap that avoids a reinstall, the existing `site-packages` is still valid for Python 3.14 and can be borrowed:

```bash
PYTHONPATH="d:/rag-app/.venv/Lib/site-packages;d:/rag-app/src" python -m pytest -q
```

Qdrant's embedded mode needs `pywin32` on Windows (portalocker file locking). A proper `pip install -e .` wires it up via a `.pth` file; the borrowed-site-packages path above needs it added explicitly:

```bash
PYTHONPATH="...;d:/rag-app/.venv/Lib/site-packages/win32/lib;d:/rag-app/.venv/Lib/site-packages/win32;d:/rag-app/src"
```

## Commands

```bash
pytest -q                              # 114 tests, <1s (pyproject sets pythonpath=src, testpaths=tests)
pytest tests/test_grounding.py -q      # grounding guarantees
pytest tests/test_pipeline_gate.py -q  # the score-gate invariant

python -m rag_app chunks --all [--show N]   # chunk stats + boundary bleed; no models loaded
python -m rag_app ingest [--preset X | --all]
python -m rag_app ask "..." [--preset X] [--filter k=v] [--quiet]
python -m rag_app chat [--preset X]                       # REPL — models load once, ask freely
python -m rag_app compare "..." [--generate]              # every preset side by side + summary table
python -m rag_app eval [--all] [--sweep] [--compare-retrieval] [--k N]   # see below
python -m rag_app debug [--preset X] [--generate] [--show-pass]         # see below
python -m rag_app models                       # embedding registry + prefix rules
```

`eval --compare-retrieval` measures hit-rate@k for dense-only vs dense+BM25-hybrid retrieval, same embedder/store/pool, isolating that one variable. `debug` runs every answerable gold question through the real pipeline and labels each as `retrieval` (wrong document — never reached the LLM's context) or, with `--generate`, `generation`/`pass` (right document — did the answer actually use it). Both are read-only and idempotent; `--generate` is the only flag that spends real LLM calls.

`data/store/` is gitignored, so a fresh clone must `ingest` before `ask` — otherwise `ask` raises `FileNotFoundError` naming the exact ingest command.

The suite never downloads models: every test injects a fake or constructs vectors by hand.

## Architecture

```
data/tickets/*.jsonl → chunk (3 strategies) → bi-encoder embed → numpy or Qdrant
question → embed → [dense top-K | dense+BM25 fused by RRF] → cross-encoder rerank top-N → score gate → LLM
```

Plain Python by design. **No LangChain / LlamaIndex, no web UI.** The numpy store is the inspectable default; Qdrant is a second backend behind the same protocol so the vector-DB concepts are real rather than described.

`ask()` in [pipeline.py](src/rag_app/pipeline.py) is the only orchestrator; every other module is a single stage with no knowledge of its neighbours.

### Grounding is enforced in four places

All four must survive any refactor:

1. [pipeline.py](src/rag_app/pipeline.py) — if the best *cross-encoder* score is below `score_threshold`, return `DONT_KNOW` and **never call the LLM**. `used_llm=False` is the observable signal. There are three gate paths: `no-candidates`, `below-threshold`, `model-refused`.
2. [generate.py](src/rag_app/generate.py) — the system prompt restricts the model to the supplied excerpts and requires `[TIC-####]` citations. Context blocks are labelled with the **same** token the model is asked to cite; labelling them `[1]`/`[2]` while demanding `[TIC-1001]` teaches the wrong format by example.
3. `is_refusal()` — a refusal that passes the score gate is stripped of its sources. Matching is punctuation-normalized because models rewrite the em dash in `DONT_KNOW`; an `==` check silently misclassifies a correct refusal as a real answer.
4. `cited_sources()` — splits the model's citations into grounded and invented. A citation naming a ticket never shown is reported as `hallucinated_citations`.

The gate reads the rerank score, not the retrieval score — cosine similarity is always positive-ish and would not gate anything useful.

### Rerank scores are sigmoid-scaled

The cross-encoder emits **unbounded logits** (~-11..+11), not a 0-1 similarity. `rerank_score_scale: sigmoid` puts the gate on a calibrated probability scale. Sigmoid is monotonic so scaling never reorders — it only makes the threshold interpretable. `raw` is supported; `config.py` then permits a threshold outside 0-1 and rejects it otherwise.

`score_threshold` is tuned by `eval --sweep`, not by feel. Re-run it after any corpus or model change.

### Chunking strategy matters more than chunk size

Three strategies in [chunking.py](src/rag_app/chunking.py): `flat` (character windows over the concatenated corpus), `ticket` (one chunk per ticket), `section` (turns under a repeated header).

`flat` **bleeds across ticket boundaries** — a window starting in TIC-1042 and ending in TIC-1043 is cited as TIC-1042. Each chunk records `spans_tickets` and `bleed` so this is measured, not argued. Larger windows bleed *more*: 70% at 500 chars, 97% at 1000.

`flat` is retained deliberately as the negative control. Do not "fix" it.

### Hybrid retrieval (BM25 + RRF) exists, measured, and is NOT the default

[bm25.py](src/rag_app/bm25.py) is a from-scratch BM25 Okapi implementation (no `rank_bm25` dependency, consistent with the numpy store being hand-rolled too). [hybrid.py](src/rag_app/hybrid.py) fuses a dense ranking and a BM25 ranking by Reciprocal Rank Fusion (rank position only — never blend a cosine score and a BM25 score directly, they live on incomparable scales). `cfg.retrieval.mode` switches `ask()` between `"dense"` (default) and `"hybrid"`.

Measured with `eval --all --compare-retrieval` (hit-rate@3, dense vs hybrid, same embedder/store/pool — the *only* variable that changes): **hybrid did not help on this corpus.** Flat-chunked presets A/B regressed slightly (-5%); ticket/section presets C/D showed no change (already at 95-100%, a ceiling effect). Root cause on A/B: BM25 latches onto boilerplate/header vocabulary shared by boundary-bled neighbour chunks (the `flat` defect above) and RRF lets that dilute an already-correct dense pick. See the comment block above `retrieval:` in [config.yaml](config.yaml) for the exact numbers. Keep `mode: dense` unless re-measuring on a different corpus changes this.

When comparing modes, pin `dense_pool` explicitly in both `ask()` and `compare_retrieval_modes()` — `hybrid_retrieve()`'s own default silently widens it to `max(k*2, 10)`, which would let the hybrid arm see more dense candidates than the dense-only arm and invalidate the comparison.

### Failure separation: "wrong document" vs "right document, wrong answer"

[evaluate.py](src/rag_app/evaluate.py)'s `label_failures()` sorts every answerable gold question into exactly one bucket, using `ask()` itself (not a re-derivation of gate logic) so a label reflects what the live app actually did:

- **`retrieval`** — the ticket never reached the final top-`rerank_n` context. No LLM, however good, could have answered correctly. Evidence distinguishes *where* it was lost: never retrieved at all (bi-encoder's fault) vs retrieved but reranked out (cross-encoder's fault).
- **`generation`** — the ticket *was* in context, and the app still produced a wrong answer anyway. This includes the score gate refusing despite a passing document (`used_llm=False` with `reranked_rank` set) and the model itself refusing or omitting `GoldQuestion.must_contain` despite having the right excerpt. A gate false-refusal counts here, not as a retrieval bug — retrieval did its job.
- **`unconfirmed`** — ticket reached context, but `--generate` wasn't passed, so nothing checked what the LLM did with it. `use_llm=False` (default) costs nothing and can only prove `retrieval` or leave a question `unconfirmed`; only `--generate` can resolve `unconfirmed` into `pass`/`generation`.

The boundary is `reranked_rank is not None`, i.e. membership in the final `rerank_n`-sized list — not "rank exactly 1". A ticket at rank 2 of 3 still reached the LLM.

### Dependency injection is the test seam

`ask()` accepts `embedder`, `reranker`, `generate_fn`, `store` and `bm25` overrides; `run_ingest()` accepts `embedder`; `generate_answer()` accepts `client`. Preserve these — they are the only reason the suite runs without network access or a torch download.

`Embedder` and `CrossEncoderReranker` import `sentence_transformers` *inside* `__init__`, not at module scope, so importing `rag_app.pipeline` does not drag in torch. Keep those imports lazy.

Anything sweeping presets must reuse models via `cli.shared_models()` — constructing `Embedder`/`CrossEncoderReranker` per preset reloads both torch models each time. Same reasoning for `BM25Index.from_store()` in hybrid mode: build it once per preset, not per question — [repl.py](src/rag_app/repl.py)'s `Session` caches both the store and the BM25 index per preset for exactly this reason.

### Embedding models are asymmetric

`Embedder` exposes `encode_queries` and `encode_documents` separately. E5 needs `query: `/`passage: `; BGE needs an instruction on the query only; MiniLM needs neither. Using the wrong one degrades retrieval **with no error**. `MODEL_REGISTRY` in [embed.py](src/rag_app/embed.py) holds the per-family rules and infers from the checkpoint name for unknown models.

### Vectors are pre-normalized

`Embedder` passes `normalize_embeddings=True`, so `VectorStore.search` computes cosine as a plain dot product. Any code writing vectors into a store directly (tests included) must L2-normalize first or the ranking is silently wrong.

### Stores carry provenance

`meta.json` records the embedding model, dim, strategy and preset params. `open_store()` refuses a store built with a different embedding model — otherwise a model swap either explodes with a numpy shape error or, when dims happen to match (common among MiniLM-class models), silently returns garbage.

### Qdrant embedded mode is NOT HNSW

`QdrantClient(path=...)` is a pure-Python prototype implementation: it accepts `hnsw_config`, ignores it, and does exact brute-force search. It also warns that payload indexes have no effect. Real HNSW requires the server (`docker run -p 6333:6333 qdrant/qdrant`, then set `qdrant.url`). Do not quote HNSW performance numbers from embedded mode.

### Config and presets

`load_config()` reads `config.yaml` and `.env` from the repo root, resolved as `parents[2]` of [config.py](src/rag_app/config.py) — moving that file breaks path resolution. `AppConfig` and friends are frozen dataclasses, so tests build variants by constructing a whole new `AppConfig` rather than mutating (see `tests/conftest.py::make_config`).

`load_config` validates: `default_preset` exists, `rerank_n <= retrieve_k`, threshold matches the score scale, backend is known, `retrieval.mode` is known.

Presets `A`/`B` differ only in chunk size so the size effect is isolated; `C`/`D` change strategy. Adding a preset to `config.yaml` is enough — `compare`, `eval` and `chunks` iterate `sorted(cfg.chunk_presets)`.

API key resolution is `OPENROUTER_API_KEY`, falling back to `LLM_API_KEY`.

### Windows console encoding

`cli.main()` reconfigures stdout/stderr to UTF-8. Windows consoles default to cp1252, which cannot encode the `→` in the summary line or the em dash in `DONT_KNOW` — printing either raises `UnicodeEncodeError` and takes down an otherwise correct answer.

## Known cruft

- `docs_dir` and `data/docs/*.md` are vestigial — the SDK docs from the original version. Nothing reads `docs_dir` since the corpus moved to `data/tickets/`. Harmless, kept as reference.
- The plan and design docs under [docs/superpowers/](docs/superpowers/) describe an earlier design: generation on **Grok (xAI)** via `GROK_API_KEY`, and SDK markdown rather than tickets. The code targets **OpenRouter** (`https://openrouter.ai/api/v1`, default `openai/gpt-4o-mini`) through the `openai` SDK, over a ticket corpus. Trust `config.yaml` and the source over those documents.
- Chunking is character-based, not token-based, despite `chunk_size` reading like tokens.
