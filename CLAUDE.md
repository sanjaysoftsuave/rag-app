# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Working: **Python 3.14.7**, `.venv` active, `pytest -q` green in ~3s. `streamlit 1.62.0` and `pypdf 6.15.0` installed.

**Use `python -m <tool>`, never the bare command.** `.venv/Scripts/*.exe` are console-script shims with an absolute interpreter path baked in at install time, and this venv was originally built at `D:\Notes\AI Learing\rag-app` on a different machine. `python.exe`/`pythonw.exe` were regenerated when the venv was repaired; every other shim (`pytest.exe`, `pip.exe`, `huggingface-cli.exe`, `torchrun.exe`, …) still points at the dead path and fails with *"Fatal error in launcher: Unable to create process"*. So:

```bash
python -m pytest -q          # not: pytest
python -m pip install ...    # not: pip
python -m rag_app ui         # this one is fine either way — see below
```

`python -m rag_app ui` launches Streamlit as `sys.executable -m streamlit`, deliberately not via `streamlit.exe`, so it works regardless of shim state. Keep it that way.

How the venv got fixed, in case it recurs: `pyvenv.cfg` pointed at a `home =` interpreter that does not exist here. Running `py -m venv .venv` over the existing directory rewrites `pyvenv.cfg` and the `python.exe` shims **while leaving `Lib/site-packages` intact** — repairing it without re-downloading ~2 GB of torch. That only works because the wheels are `cp314` and the new interpreter is also 3.14.

[rag.cmd](rag.cmd) is a leftover from that broken period — it borrows `site-packages` via `PYTHONPATH`. Harmless, but unnecessary now that the venv works; prefer activating the venv.

## Commands

```bash
python -m pytest -q                    # 510 passed, 1 skipped, ~5s — real embedded Qdrant per test
python -m pytest tests/test_grounding.py -q            # the four grounding guarantees
python -m pytest tests/test_pdfs.py tests/test_explain.py -q   # PDF loading + UI introspection

python -m rag_app ui [--port N] [--headless]   # the app; or: streamlit run app.py
python -m rag_app ask "..." [--filter k=v] [--quiet]   # one question, full trace, no browser
python -m rag_app chunks [--all] [--show N]            # preview chunking; loads no models
python -m rag_app models                               # embedding + reranker registries
python -m rag_app eval [--generate] [--json]           # hit-rate@k, recall@k, MRR, rerank lift
python -m rag_app eval --generate --judge --geval --ragas [--snapshot LABEL] [--limit N]
python -m rag_app debug [--generate] [--show-pass]     # retrieval vs generation failures
python -m rag_app codes [--init] [--json]              # rank your open-coding categories
python -m rag_app judge [--init] [--json]              # judge vs your labels: agreement + kappa
python -m rag_app compare BEFORE AFTER [--json]        # diff two eval snapshots
python -m rag_app agent "..." [--impl plain|langgraph] [--max-steps N] [--memory]
python -m rag_app arena [--arms workflow,agent] [--generate]   # workflow vs agent
python -m rag_app atasks [--tasks PATH] [--generate]   # trajectory-level agent eval
```

**`eval` with no flags makes zero LLM calls, and always will.** The substring
metrics are the free regression tripwire; every scorer that spends money has to be
typed. `--judge`/`--geval`/`--ragas` require `--generate` (there is no answer to
score otherwise), the estimate is printed to stderr *before* the first call, and
`evaluation.max_llm_calls` refuses the run above its ceiling rather than partway
through. Calls per question, at `rerank_n=3`: judge 1, G-Eval `geval_samples` (5),
RAGAS 4 plus 1 more for each question with a `reference_answer`.

**There is no `ingest` command.** Building the index is a UI action (*Add documents → Run ingest*), deliberately in one place so the two surfaces cannot drift. `run_ingest()` is still a plain function and is what the tests call.

`data/store/` is gitignored, so a fresh clone must add documents and build the index before `ask` works — `open_store()` raises `FileNotFoundError` pointing at the UI, and `run_ingest()` raises `ValueError` naming the accepted extensions.

The suite never downloads models: every test injects a fake or constructs vectors by hand. PDF reading is faked through `pdfs.read_pdf(reader_factory=...)`, so no binary fixtures are checked in.

## Architecture

```
data/tickets/*.md, *.txt, *.pdf  (each file windowed on its own) → chunk → bi-encoder embed → Qdrant
question → embed → [dense top-K | dense+BM25 fused by RRF] → cross-encoder rerank top-N → score gate → LLM
```

**Plain Python is the backbone; the frameworks are a labelled exhibit.** The
retrieval pipeline, the CLI, the UI and the entire offline suite import no
LangChain, LangGraph, LlamaIndex or mem0 — `tests/test_no_framework_imports.py`
fails the build if that stops being true. What changed in Week 7 is that
[agent_langgraph.py](src/rag_app/agent_langgraph.py) and
[memory_mem0.py](src/rag_app/memory_mem0.py) exist as a deliberate side-by-side
comparison against `agent.py`/`memory.py`: **same tools, same prompt, same
grounding gate**, different control flow and different memory machinery, so each
framework's cost is measurable rather than asserted. They sit behind
`pip install -e ".[agents]"` / `".[mem0]"`, import lazily inside function bodies,
and nothing on the path that answers a question imports them. The old rule was
"no frameworks anywhere"; the rule now is **"no frameworks on the path that
answers a question."** The Streamlit UI ([ui.py](src/rag_app/ui.py)) is the primary surface, but it is a *second surface over the same `ask()`*, never a second implementation — and `ask`/`chunks`/`models` still run with streamlit uninstalled. Qdrant is the only vector store; there is no in-process numpy fallback.

**A structured `.jsonl` ticket corpus used to be the point of this app and is now gone**, along with `tickets.py`, the three chunk strategies, `evaluate.py`/`GOLD`, `sampling.py`, `traces.py` and `repl.py`. What remains is documents and PDFs. If you find a comment referring to tickets, ticket ids or chunking strategies, it is stale — `git log` has the old code if a measurement harness is ever wanted back.

`ask()` in [pipeline.py](src/rag_app/pipeline.py) is the only orchestrator; every other module is a single stage with no knowledge of its neighbours.

### Grounding is enforced in four places

All four must survive any refactor:

1. [pipeline.py](src/rag_app/pipeline.py) — if the best *cross-encoder* score is below `score_threshold`, return `DONT_KNOW` and **never call the LLM**. `used_llm=False` is the observable signal. There are three gate paths: `no-candidates`, `below-threshold`, `model-refused`.
2. [generate.py](src/rag_app/generate.py) — the system prompt restricts the model to the supplied excerpts and requires a citation copied verbatim from the excerpt's header label. Context blocks are labelled with the **same** token the model is asked to cite, and with nothing else: labelling them `[1]`/`[2]` teaches the wrong format by example, and a `(pdf_file=… | page=…)` descriptor would repeat the label back in a second format, inviting the model to cite the wrong one.

   The prompt must also say that **identifiers inside the text are not labels.** This was found live, not by inspection: a PDF containing `Ticket CS-1001` produced the answer `…after changing the password [TIC-1001]`, which `cited_sources()` correctly reported as a hallucinated citation. The model was following the prompt's own `[TIC-1001]` example. Fixing the example and adding the explicit exclusion made the same query cite `[RAG-Test-Document-1.pdf-p1]`.
3. `is_refusal()` — a refusal that passes the score gate is stripped of its sources. Matching is punctuation-normalized because models rewrite the em dash in `DONT_KNOW`; an `==` check silently misclassifies a correct refusal as a real answer.
4. `cited_sources()` — splits the model's citations into grounded and invented. A citation naming a source never shown is reported as `hallucinated_citations`. The common cause is not invention but confusion with an identifier in the text (see #2).

The gate reads the rerank score, not the retrieval score — cosine similarity is always positive-ish and would not gate anything useful.

### Rerank scores are sigmoid-scaled

The cross-encoder emits **unbounded logits** (~-11..+11), not a 0-1 similarity. `rerank_score_scale: sigmoid` puts the gate on a calibrated probability scale. Sigmoid is monotonic so scaling never reorders — it only makes the threshold interpretable. `raw` is supported; `config.py` then permits a threshold outside 0-1 and rejects it otherwise.

`score_threshold` has no automated sweep any more (that lived in the deleted `evaluate.py`). Tune it in the UI: drag the slider and watch which questions flip between answered and refused. Re-check it after any corpus or model change.

### PDFs are ordinary documents, and the citation label is load-bearing

[pdfs.py](src/rag_app/pdfs.py) loads a `*.pdf` the same way [docs.py](src/rag_app/docs.py) loads a `.md`: **pages are joined into one block of text before windowing**, so the file is chunked as a unit and cited by its filename. `chunk_docs(..., source_type="pdf")` is the only thing that distinguishes the two paths.

An earlier version windowed each page separately and cited `[handbook.pdf-p3]`. That gave a citation you could open, but split any sentence straddling a page break with `overlap` unable to bridge it — pages were windowed independently. Joining first removes that; the cost is that `[handbook.pdf]` names a document, not a location. Extraction still knows which page each piece came from, so page-level locality is recoverable if it is ever wanted back.

**The label must satisfy `generate.CITATION_RE`** (`\[([A-Za-z0-9][A-Za-z0-9._\-]*)\]`): alphanumerics, dots, underscores, hyphens, first character alphanumeric. `handbook.pdf` passes unchanged, which is why a filename works as a citation with no special case. `docs.citation_label()` folds anything that would not — an uploaded `Q3 report.pdf` becomes `Q3-report.pdf`, because a space fails the regex, `cited_sources()` then classes the citation as *invented*, and a perfectly correct answer gets reported as a hallucination. That fold also closed the same latent bug for `.md` filenames.

Blank pages are skipped when joining. A PDF whose pages are all images extracts to nothing; `PdfLoadReport.looks_scanned` says so loudly in `chunks`, ingest and the UI rather than reporting a successful ingest of zero chunks. There is no OCR.

`read_pdf`/`load_pdfs` take a `reader_factory` — resolved at call time, not as a default argument, so `monkeypatch.setattr("rag_app.pdfs._open_reader", ...)` actually takes effect. The suite exercises real extraction logic with no binary fixtures and no pypdf import.

### The Streamlit UI shows stages, it does not re-derive them

[ui.py](src/rag_app/ui.py) renders; [explain.py](src/rag_app/explain.py) computes and has **no streamlit import**, so the interesting logic stays testable in the offline suite instead of only being exercisable by clicking. Keep that split.

Three things it does that the CLI cannot, all worth preserving:

1. **What the reranker cut.** `Answer.reranked_all` holds every candidate the cross-encoder scored, not just the top `rerank_n`. This costs nothing — `rerank_all()` already scored all K to rank them, and `rerank()` is now literally `rerank_all(...)[:n]`. It is the only record of which candidate was demoted out of the LLM's context.
2. **The raw logit behind a sigmoid score.** `explain.logit()` inverts `rerank.sigmoid()`; 0.9999 and 0.99999 look identical on the gate's scale and are 2.3 logits apart. Returns `None` at the saturated ends rather than inventing precision.
3. **Timing per stage, without instrumenting the pipeline.** The UI wraps the injected `embedder`/`store`/`reranker`/`generate_fn` in stopwatch proxies. `pipeline.py` has no timing code and behaves identically.

**`chunk_size` and `overlap` are the only chunking knobs, and they are editable in the sidebar.** They are *ingest-time* settings — `ask()` never reads them, so moving those sliders changes nothing until the index is rebuilt. Three things keep that from looking like a broken control:

- The widgets default to the **store's own provenance** (`read_provenance()`, which reads `provenance.json` without opening the DB and taking its file lock), not to `config.yaml`. So the sidebar describes what you are actually querying, and a divergence is real rather than an artefact of a page reload.
- `render_index_drift()` warns explicitly when the selection no longer matches the index, and says a re-ingest is the missing step.
- `explain.config_rows()` takes the `StoreMeta` and labels the row **"chunking (as indexed)"**, because reporting the config value there would name a setting that had no part in producing the retrieved chunks.

The overlap slider's max is `chunk_size - 50`; `chunk_text()` raises when `overlap >= chunk_size`, so the widget must not be able to offer it.

Two traps the UI has to actively avoid, both already handled — do not regress them:

- With "Call the LLM" off, `ask()` still runs a stand-in `generate_fn` (shared with `evaluate._skip_generation`), so `used_llm` is `True` in the strict sense. The UI must report "skipped", never "yes", or the trace lies about whether the network was touched.
- Embedded Qdrant holds a **real file lock**. The UI keeps open store handles in a registry it owns and calls `close_handles(preset)` before ingest writes to that directory. A plain `@st.cache_resource` per store would not give control over *when* the handle closes, and ingest would fail on Windows.

### Hybrid retrieval (BM25 + RRF) exists, measured, and is NOT the default

[bm25.py](src/rag_app/bm25.py) is a from-scratch BM25 Okapi implementation (no `rank_bm25` dependency — this corpus is small enough that the real algorithm is more legible as ~80 lines of Python than as an opaque import). [hybrid.py](src/rag_app/hybrid.py) fuses a dense ranking and a BM25 ranking by Reciprocal Rank Fusion (rank position only — never blend a cosine score and a BM25 score directly, they live on incomparable scales). `cfg.retrieval.mode` switches `ask()` between `"dense"` (default) and `"hybrid"`.

`dense` is the default. Hybrid earns its keep on keyword-heavy corpora (error codes, part numbers, proper nouns) and costs a little precision on prose. **The automated A/B that used to justify this number lived in the deleted `evaluate.py`** — there is now no `--compare-retrieval`, so switching modes is a UI toggle and a judgement call on your own questions, not a measured claim.

When comparing modes, pin `dense_pool` explicitly in both `ask()` and `compare_retrieval_modes()` — `hybrid_retrieve()`'s own default silently widens it to `max(k*2, 10)`, which would let the hybrid arm see more dense candidates than the dense-only arm and invalidate the comparison.

### Dependency injection is the test seam

`ask()` accepts `embedder`, `reranker`, `generate_fn`, `store` and `bm25` overrides; `run_ingest()` accepts `embedder`; `generate_answer()` accepts `client`. Preserve these — they are the only reason the suite runs without network access or a torch download.

Week 6 and 7 add more of the same shape: `judge_fn(system, user, cfg) -> str` is
**one seam for six prompt types** (the gold judge, the trace judge, G-Eval and all
four RAGAS prompts), so a single `conftest.FakeJudge` covers every scorer;
`llm_fn(messages, cfg) -> str` drives the whole agent loop from
`conftest.scripted_llm`; `clock` makes the wall-clock budget deterministic; and
[llm.py](src/rag_app/llm.py) is now the **only** place `openai` is imported —
`build_client` returns an injected client *before* the API-key check, so a test
never needs a key and never imports the SDK.

`Embedder` and `CrossEncoderReranker` import `sentence_transformers` *inside* `__init__`, not at module scope, so importing `rag_app.pipeline` does not drag in torch. Keep those imports lazy.

Both torch models cost seconds and hundreds of MB to construct, so anything that runs more than one query must reuse them. The UI does this with `@st.cache_resource` on `_models()`; without it every widget interaction would reload them. Same reasoning for `BM25Index.from_store()` in hybrid mode — build it once per preset, not per question.

### The embedder's token ceiling bounds chunk_size

`MODEL_REGISTRY` records a **measured** `max_tokens` per model (read off each model's `max_seq_length`, not a model card):

| Model | dim | max_tokens | ~chars |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 256 | ~1000 |
| `BAAI/bge-small-en-v1.5` ← active | 384 | 512 | ~2000 |
| `intfloat/e5-small-v2` | 384 | 512 | ~2000 |

Text past that limit is **silently dropped before embedding** — it sits in the index and can never be retrieved. So `chunk_size` is bounded by the model, not by taste, and `tests/test_config.py::test_chunk_size_stays_under_the_embedders_ceiling` fails the build if a preset crosses it. The UI's slider carries the active model's ceiling in its help text and warns above it.

All three registered models are 384-dim because they are the same size class, which makes swapping among them a drop-in. `dim` is an *output* of the architecture, not a setting — a `base` model would be 768 and change storage and search cost. At small corpus sizes neither matters; the token ceiling does.

### Retrieval vs generation failures

`label_failures()` sorts every answerable gold question into exactly one bucket, calling `ask()` rather than re-deriving gate logic so a label always describes what the live app did:

- **`retrieval`** — the expected text never reached the final context. No LLM could have answered. Evidence distinguishes *never retrieved* (bi-encoder) from *retrieved then reranked out* (cross-encoder) — different fixes.
- **`generation`** — the text WAS in context and the answer still missed. **A false refusal counts here**, not as a retrieval failure: retrieval did its job, the gate is what rejected it. Calling it retrieval would send you tuning the wrong stage.
- **`pass`** / **`unconfirmed`** — `unconfirmed` means the text reached context but `--generate` was not passed, so nothing checked the answer. Claiming `pass` there would assert something nothing verified.

The boundary is membership in the final `rerank_n` context, not "rank 1". A chunk at rank 3 of 3 still reached the model.

### The judge is a different, stronger model — and the judge is itself validated

`answer_ok` in [evaluate.py](src/rag_app/evaluate.py) is substring matching, and it
is wrong in both directions: gold wanting `"3 business days"` fails an answer
saying *"after three business days"*, and gold wanting `"cached"` passes an answer
saying *"the cache was not the problem"*. [judge.py](src/rag_app/judge.py) asks a
model whether the answer conveys the FACT. **Both are reported on adjacent lines
and the disagreement list is printed** — that gap is the measurement, and a judge
agreeing with substring matching everywhere would not be worth its cost.

`evaluation.judge_model` defaults to a *different, stronger* model than
generation, because a model grading its own output prefers its own phrasing. A
same-model judge is deliberately **allowed** (the report warns instead of the
loader refusing) since seeing that effect is the teaching point.

Four decisions worth preserving:

- **`unscored` is never `incorrect`.** Parse failure, exception, exhausted budget,
  and `used_llm=False` all land there and leave the denominator. Counting a judge
  malfunction as a wrong answer makes an unreliable judge look like a broken app.
- **The gold judge never sees the retrieved contexts.** Correctness and
  groundedness are different properties; RAGAS faithfulness measures the second.
  Fusing them lets a fluent answer built on a wrong excerpt score `correct`.
  `judge_trace` is the exception — it has no reference, so context is all it has.
- **JSON key order is `reasoning` → `verdict`.** The model is autoregressive;
  verdict-first makes the reasoning a post-hoc rationalisation.
- **No prose fallback when parsing.** *"The answer is not incorrect"* contains
  `"incorrect"`, so a substring scan would misread it into a real-looking number.

[judge_validation.py](src/rag_app/judge_validation.py) measures the judge against
the labels you wrote in the Week-5 coding sheet, using the **trace** prompt so the
judge sees exactly what you saw. It reports raw agreement **and Cohen's kappa**:
if you labelled 17 of 20 `correct` and the judge says `correct` to everything, raw
agreement is 85% and the judge is a constant function. When both raters used one
label, kappa is 0/0 — it returns `nan` and says so, because reporting 1.0 there
would be the most misleading number in the repo. At n=20 **the deliverable is the
disagreement list, not the coefficient**, and `describe()` prints that limit
unconditionally.

### RAGAS's four metrics are implemented here, not imported

Same argument as `bm25.py`: each is ~40 lines, and reading them is the only way to
know what the number means. Every metric returns its **intermediate evidence** —
the statements, the per-context verdicts — because a bare float is unauditable.

Four traps, each pinned by a test:

- **An answer with no extractable claims is `unscored`, never faithfulness 1.0.**
  Scoring it 1.0 makes a refusal maximally faithful, and the metric then rewards
  the gate for refusing everything.
- **Answer relevancy encodes BOTH sides with `encode_queries`.** bge is
  asymmetric; using `encode_documents` on one side of a question-to-question
  comparison shifts every similarity by a constant with no error anywhere.
- **Context precision is rank-aware** (Average Precision): `[useful, junk, junk]`
  scores 1.0 and `[junk, junk, useful]` scores 1/3. A short verdict array fails
  visibly rather than silently scoring fewer contexts.
- **Context recall needs `GoldQuestion.reference_answer`**, and both shortcuts are
  rejected: `must_contain` holds fragments, and `expect_in_chunk` is *defined* as
  text in a retrieved chunk, so recall from it would be 1.0 by construction. A
  question with no reference reports `None` — **not 0.0**, since averaging a zero
  for missing data reports a regression that never happened.

Absolute values do not transfer across corpora — answer relevancy's floor is
~0.3–0.6, not 0. Only the same metric on the same questions before and after a
change means anything, which is what
[before_after.py](src/rag_app/before_after.py) is for. It fingerprints the gold
set and shouts when two snapshots were taken against different ones, prints the
changed settings *before* the deltas (a number with no stated cause is not a
finding), and states what one question is worth so a sub-noise delta is not read
as a trend.

### G-Eval does not use logprobs, and says so

The paper computes `E[score] = Σ p(s)·s` over the 1–5 token distribution.
OpenRouter does not reliably forward logprobs — it accepts the parameter, and
whether it arrives depends on the upstream provider, with the field simply absent
when it does not. So the same expectation is estimated by **sampling**:
`geval_samples` calls at `geval_temperature`, averaged. `config.py` **rejects**
`geval_samples > 1` with `geval_temperature: 0`, because N identical greedy
samples cost N× and estimate no variance — that one rule encodes the whole
substitution as an invariant. `stdev` is reported next to the mean: `[1,5,1,5,3]`
averages to 3.0 and is not a 3. A score outside 1–5 is **dropped, not clamped**;
clamping a `7` to `5` launders a misunderstood rubric into a maximal score.

Auto-CoT is also dropped: generating the rubric per run would produce a different
rubric every run and destroy the run-to-run comparability before/after depends on.

### The agent does not bypass the gate

`search_documents` wraps retrieve → rerank → top-N, **not `pipeline.ask()`**.
Wrapping `ask()` looks safer and is not: the agent still writes its own prose over
the sub-answers, and that text is a second, ungated generation. The four
guarantees are relocated, and a fifth is added:

| Guarantee | Where it lives in the agent |
|---|---|
| 1. score gate | **Inside the tool.** Below threshold it returns a "nothing matched" string and contributes zero evidence, so the model cannot see the rejected text. Stricter than `ask()`. |
| 2. citation rules | `generate.CITATION_RULES`, shared **verbatim** with `build_prompt`. Two prompts explaining citations in different words are two contracts, and only one was debugged against the `[CS-1001]` failure. |
| 3. refusal strips sources | applied to the final answer, exactly as `ask()` does |
| 4. invented citations | checked against the union of evidence from **every** step |
| **5. no-evidence gate** | **new.** A final answer produced without any tool ever returning an excerpt is forced to `DONT_KNOW`. An agent answering from its own weights is the failure only this shape produces, and nothing in `ask()` ever needed to catch it. |

Memory text is deliberately **not citable**: `cited_sources()` only accepts labels
present in the evidence, so a memory-sourced citation is reported as hallucinated.

`parse_action()` is tolerant about presentation (case, bold, fences, JSON input)
and strict about two things: a reply with no `Action:` line is a parse failure and
**never an implicit final answer**, and anything after a model-written
`Observation:` is **discarded** — models pre-fill fake observations, and keeping
them lets the model invent its own tool results.

### Budgets produce a visible failure, never a truncated answer

Seven budgets (`max_steps`, `max_tool_calls`, `max_llm_calls`, `max_prompt_chars`,
`wall_clock_seconds`, `max_parse_failures`, `repeat_action_limit`). **Every trip
sets `failed=True`, returns `DONT_KNOW` and `sources=[]`**, and `describe()` names
the number that tripped. An agent that ran out of budget has not answered;
attaching partial prose to a truncated run is the dishonesty the `failed=True`
precedent exists to prevent. The trajectory is preserved in `steps` — it just does
not become an answer.

The prompt budget is checked **before** the call, so an over-budget prompt costs
nothing (tripwire-tested). `clock` is injectable so the wall-clock budget is
testable without sleeping.

### Agent memory has its own Qdrant directory, and why

Three options; only one survives. **Same collection as the corpus** would let
`search_documents` retrieve memories and hand them to the model under a `[source]`
header — a remembered guess becomes a citable document. **Same directory** fails
too: `QdrantClient(path=...)` locks the *directory*, not the collection, so
`ui.close_handles(preset)` would close memory mid-conversation. So memory gets
`data/agent_memory/qdrant_memory/`: its own lock, its own lifetime.

**The JSONL is the source of truth; Qdrant is derived.** `QdrantStore.build()`
deletes and recreates a collection, so appending a memory rebuilds from the log —
the same relationship the corpus already has, milliseconds at memory scale, and
decisively **zero changes to `qdrant_store.py`**, the most scarred file here. It
is O(N) per write; at tens of thousands of turns it would need a real upsert path.

### MMR, query rewriting and HyDE are built and OFF by default

Three optional stages, each a real change to what the model reads:

- **[mmr.py](src/rag_app/mmr.py)** — Maximal Marginal Relevance over the reranked candidates. `retrieval.mmr`, `retrieval.mmr_lambda`. **Its first pick is always the most relevant, and that is load-bearing**: `ask()` reads `reranked[0].score` for the gate, so MMR changes what the LLM *reads*, never whether it is *called*. A test pins that invariant at every lambda.
- **[rewrite.py](src/rag_app/rewrite.py)** — `retrieval.query_mode: rewrite` restates the question in document vocabulary; `hyde` invents a plausible answer and searches with that. Both cost an LLM call *before* retrieval.

Two rules the pipeline enforces and that are easy to get wrong:

1. **The transform changes what is searched for, never what is answered.** Generation still receives the original question. Answering the rewrite would let a bad transform silently replace the user's question.
2. **Reranking scores against the ORIGINAL question**, not the transform. The cross-encoder's whole value is judging relevance to what was actually asked, and a HyDE passage is a fabricated answer, not a question.

A failed transform falls back to the original question and records `failed` in `meta` — a degraded search beats no answer, but it must be visible. `Answer.meta["search_text"]` carries what was actually embedded, without which the retrieval list is inexplicable.

Off by default because neither is worth enabling without a gold set to show it helped.

### Rerankers are registered, and swapping one moves the threshold

`RERANKER_REGISTRY` in [rerank.py](src/rag_app/rerank.py) covers the ms-marco family, `bge-reranker-base`/`v2-m3`, and `cohere/rerank-v3.5`. `build_reranker()` picks the local `CrossEncoderReranker` or the API-backed `CohereReranker`.

**Each spec records the `score_scale` it expects, and that field is the whole point.** ms-marco emits unbounded logits needing a sigmoid; Cohere returns a 0-1 relevance already, and squashing that again would drag every score toward 0.5 and silently change what the gate means. `python -m rag_app models` warns when `rerank_score_scale` disagrees with the active model.

Swapping a reranker needs **no re-ingest** — it never touches stored vectors — but `score_threshold` is calibrated per model and must be re-checked.

`CohereReranker` restores input order from the API's index-keyed results. Getting that wrong would put scores on the wrong chunks, which looks like "the reranker is bad" rather than a bug.

### Embedding models are asymmetric

`Embedder` exposes `encode_queries` and `encode_documents` separately. E5 needs `query: `/`passage: `; BGE needs an instruction on the query only; MiniLM needs neither. Using the wrong one degrades retrieval **with no error**. `MODEL_REGISTRY` in [embed.py](src/rag_app/embed.py) holds the per-family rules and infers from the checkpoint name for unknown models.

**This is live now:** the active model is `BAAI/bge-small-en-v1.5`, which is asymmetric. Encoding the same string both ways gives cosine 0.94, not 1.0 — the prefix is genuinely being applied. The UI's *② Query encoding* panel prints the exact prepared string for this reason.

Swapping the model is a `config.yaml` edit plus a re-ingest. `StoreMeta.assert_compatible_with` refuses the stale index rather than returning garbage: *"Store was built with embedding model X but config asks for Y. Vectors from different models are not comparable."*

### Vectors are pre-normalized

`Embedder` passes `normalize_embeddings=True`. Qdrant's collections are created with `Distance.COSINE`, so this is what makes that distance metric meaningful rather than a numerically-valid-but-meaningless computation. Any code writing vectors into a store directly (tests included) must L2-normalize first or the ranking is silently wrong.

### Stores carry provenance — `provenance.json`, deliberately not `meta.json`

[qdrant_store.py](src/rag_app/qdrant_store.py)'s `QdrantStore` writes a provenance sidecar (embedding model, dim, chunk_size, overlap, chunk count) next to every collection. `open_store()` reads it back via `load_meta()` and refuses to proceed on a mismatch — otherwise a model swap either fails with a dimension-mismatch error from Qdrant's own local matching engine, or — when dims happen to match, common among MiniLM-class models — silently returns garbage.

**The file is named `provenance.json`, not `meta.json`, for a real reason, found the hard way.** In embedded mode, `meta_dir` is the *same directory* Qdrant's own local storage uses, and Qdrant already keeps its own `meta.json` there — a `{"collections": {...}}` registry it needs to reopen the collection. A same-named sidecar silently clobbers it. The result isn't an immediate crash: `build()` still succeeds, the first `search()` in the same process still works, tests still pass if they never reopen — the corruption only surfaces on the **next fresh `QdrantClient(path=...)`**, which fails with `KeyError: 'collections'` reading its own overwritten registry. That's exactly the ingest-then-later-ask pattern every real invocation of this app uses, so it would have broken *every* real usage while every naive test still passed. Caught only by writing a test that explicitly closes and reopens a store — if you add any new local sidecar file to a Qdrant storage directory, reopening after close is the test that would have caught this, not a fresh-process test with no reopen.

### Qdrant embedded mode is NOT HNSW

`QdrantClient(path=...)` is a pure-Python prototype implementation: it accepts `hnsw_config`, ignores it, and does exact brute-force search — internally via numpy, which is why a dimension mismatch surfaces as a raw numpy `ValueError` ("shapes (3,3) and (2,) not aligned...") rather than a Qdrant-specific error. It also warns that payload indexes have no effect. Real HNSW requires the server (`docker run -p 6333:6333 qdrant/qdrant`, then set `qdrant.url`). Do not quote HNSW performance numbers from embedded mode — which is the default, so this matters even though numpy itself is gone.

### The test suite pays a real, measured cost for testing the real backend

Tests build genuine embedded `QdrantStore` instances (`tests/conftest.py::make_qdrant_store`), not a numpy-shaped stand-in — there is no lighter-weight fake backend to fall back to since numpy was removed everywhere, including tests. Slower than a numpy-backed suite's ~0.6s, still fast enough to run on every change: **510 passed, 1 skipped in ~5s** (measured, Python 3.14.7, with both optional
extras installed; the skip is the without-langgraph branch).

Still fully offline (no network, no model download) and fast enough to run on every change, just no longer near-instant. If a test needs to reopen a store it just built (proving a fix like the one above), it must explicitly `.close()` the first handle first — embedded mode holds a real file lock.

### The corpus ships empty, and one chunking config

There is no sample data. `data/tickets/` holds only `.gitkeep`; `data/store/` is empty. A fresh checkout must add documents and build the index before `ask` works — `open_store()` raises `FileNotFoundError` pointing at the UI, and `run_ingest()` raises `ValueError` naming the accepted extensions.

`data/tickets/*` is gitignored (with `!.gitkeep`) because the UI writes uploads there and a real corpus is usually not something to commit. The directory keeps its historical name; it holds `.md`/`.txt`/`.pdf`, not tickets.

**Measurement lives in a gold file, not in code.** [evaluate.py](src/rag_app/evaluate.py) is corpus-agnostic; the questions are data at `data/gold.yaml` (gitignored — it names your documents). `gold.example.yaml` at the repo root is the format.

The key departure from the textbook definition: **relevance is matched on CONTENT, not document id.** Hit-rate@k is normally "did the relevant doc id appear in the top k", which is useless on a corpus of one PDF — it would be 1.0 for every question including nonsense. So a gold entry names a *snippet* that must appear in a retrieved chunk. That works at any corpus size and is checkable by a human reading the document.

- `must_contain` — must appear in the **answer** (generation ground truth)
- `expect_in_chunk` — must appear in a **retrieved chunk** (retrieval ground truth); defaults to `must_contain`
- `unanswerable: true` — the app must refuse. Include several, or you cannot distinguish a calibrated gate from one that never refuses.

`recall@k` differs from `hit-rate@k` only when an entry lists several snippets — which is exactly when hit-rate flatters a partial retrieval.

**`config.yaml` accepts two shapes and the rest of the app cannot tell them apart:**

```yaml
chunking: {chunk_size: 500, overlap: 100}       # one config, named "default"
# or
chunk_presets: {A: {...}, C: {...}}                            # several
default_preset: C
```

The single form becomes a one-entry `chunk_presets` map keyed `config.DEFAULT_PRESET` (`"default"`, which is also the store suffix — `data/store/qdrant_default/`). Everything downstream iterates `sorted(cfg.chunk_presets)`, so **restoring side-by-side comparison is a YAML-only edit**: swap the block and `chunks --all` plus the UI's preset selector start working again with no code change, each preset getting its own index directory. The UI hides its preset dropdown when there is only one — a one-item dropdown is not a decision the user has.

`docs_dir` is now optional (`raw.get`), so the dead key no longer has to sit in `config.yaml`. It is still on `AppConfig` and still read by nothing.

### Config and presets

`load_config()` reads `config.yaml` and `.env` from the repo root, resolved as `parents[2]` of [config.py](src/rag_app/config.py) — moving that file breaks path resolution. `AppConfig` and friends are frozen dataclasses, so tests build variants by constructing a whole new `AppConfig` rather than mutating (see `tests/conftest.py::make_config`).

`load_config` validates: `default_preset` exists, `rerank_n <= retrieve_k`, threshold matches the score scale, `retrieval.mode` is known.

The UI additionally builds session-only variants with `dataclasses.replace` (its K/N/threshold/mode controls). **Those bypass `load_config`'s validation entirely**, so any invariant that matters must be re-checked at the widget — which is why `rerank_n`'s slider is capped at `retrieve_k`, and why the threshold control switches to a `number_input` when `rerank_score_scale` is `raw` (a 0–1 slider cannot express a logit threshold).

API key resolution is `OPENROUTER_API_KEY`, falling back to `LLM_API_KEY`.

### Windows console encoding

`cli.main()` reconfigures stdout/stderr to UTF-8. Windows consoles default to cp1252, which cannot encode the `→` in the summary line or the em dash in `DONT_KNOW` — printing either raises `UnicodeEncodeError` and takes down an otherwise correct answer.

## Known cruft

- `docs_dir` on `AppConfig` is dead — nothing reads it, and it is no longer required in `config.yaml`. `tickets_dir` is the real (badly named) corpus directory.
- The Qdrant collection is named `documents`; stores built before that rename hold a `tickets` collection and will read as "no index — go ingest". Rebuilding takes about a second.
- Chunking is character-based, not token-based, despite `chunk_size` reading like tokens.
- The OpenAI client block used to be duplicated in `generate.py` and `rewrite.py`; it now lives once in [llm.py](src/rag_app/llm.py). If you add a seventh caller, use `chat_once`/`chat_messages` rather than a third copy.
- `data/traces/sample_seed42_n20*.{jsonl,md}` are **stale**: they were generated against the deleted `.jsonl` ticket corpus (`TIC-1001`..`TIC-1035`) and a `sample` command that no longer exists. The current batch is `sample_current_n20*`. The old files are kept only as a format reference.
- The honest limits on the evaluation numbers here: the gold set is 13 questions (9 answerable), the human label set is 20 traces, G-Eval has no logprobs, RAGAS absolute values do not transfer across corpora, and the judge is validated against **one person's** labels — so "agreement" means agreement with you.
