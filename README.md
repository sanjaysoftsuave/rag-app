# Ask My Documents

A plain-Python RAG app over your own documents. Ask a question, get an answer
built only from what you uploaded, with the source it came from — and an
explicit "I don't know" when the answer isn't there.

No LangChain, no LlamaIndex. Every retrieval step is a function you can read and
print, and a browser UI that shows you each one for a given question rather than
only the answer.

```
data/tickets/*.md, *.txt, *.pdf   (each file windowed on its own)
    -> chunk -> embed -> Qdrant -> top-K -> cross-encoder rerank -> score gate -> LLM
                                      |            |                      |
                             embedded or server   reranks near-dups    refuses instead
                              + metadata filter                         of guessing
```

## What documents are accepted

Drop files into `data/tickets/` — or upload them in the browser, which puts them
there for you.

| Format | Loader | Citation label |
|---|---|---|
| `*.md`, `*.markdown`, `*.txt` | [docs.py](src/rag_app/docs.py) | The filename, e.g. `[refund-policy.md]` |
| `*.pdf` | [pdfs.py](src/rag_app/pdfs.py) | The filename, e.g. `[handbook.pdf]` |

All three go through the same path. A PDF's pages are joined into one block of
text before windowing, so it is chunked exactly like a `.md` file — which means
a fact spanning a page break stays whole.

Every source is windowed **on its own**, never concatenated with another file.
That single rule is what makes a citation trustworthy: because there is never a
second document in a window, a chunk can never be attributed to a source that
supplied only part of its text.

**A scanned PDF contributes nothing.** Page images carry no text layer, so
extraction returns empty and the file yields zero chunks. Ingest and the UI say
so explicitly rather than reporting a successful ingest of nothing. There is no
OCR in this app.

## The browser UI

```bash
pip install -e ".[ui]"
python -m rag_app ui            # or: streamlit run app.py
```

Three tabs: **Ask**, **Corpus & chunking**, **Add documents** (drag in PDFs and
re-ingest without touching a terminal).

The Ask tab is the point. For every question it shows the config in effect, the
exact string handed to the bi-encoder *including the prefix an asymmetric model
requires*, the top-K with scores, the full rerank table — **including the
candidates that were scored and cut**, with how far each moved and the raw logit
behind each sigmoid score — which of the four gate paths fired and why, the
literal prompt sent to the model, the grounded-vs-invented citation split, and a
per-stage timing breakdown.

It runs the same `ask()` the CLI does; nothing is re-derived for display. Timing
comes from wrapping the injected embedder/store/reranker in stopwatches, so the
pipeline itself carries no instrumentation.

## Quick start

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev,ui]"
copy .env.example .env          # then set OPENROUTER_API_KEY

python -m rag_app ui            # add documents, build the index, ask
```

Then in the browser: **Add documents** → drop your files → **Run ingest** →
**Ask**. First launch downloads ~150 MB of models and takes a minute; after that
everything but generation is local.

## Commands

The UI is the primary surface. What survives on the command line is what is
genuinely easier without a browser:

| Command | What it does |
|---|---|
| `ui [--port N] [--headless]` | Launch the app. Or `streamlit run app.py`. |
| `ask "..." [--filter k=v] [--quiet]` | Answer one question and print the full retrieval trace. |
| `chunks [--all] [--show N]` | Preview how the corpus splits. Loads no models, so it is instant. |
| `models` | The embedding-model registry and each family's prefix rules. |

**There is no `ingest` command.** Building the index is a UI action, deliberately
in one place so the two surfaces cannot drift apart.

---

# The topics, and where each one lives in this repo

## 1. Why RAG

A language model answers from its training data. Ask it about *your* documents
and it will produce something fluent and invented, because "I don't know" is a
rare token in its training distribution. RAG replaces recall with reading: find
the right text first, then let the model write only from that text.

Two things make it *grounded* rather than just "context-stuffed": a citation the
user can check ([generate.py](src/rag_app/generate.py)), and a refusal path that
runs before the model does ([pipeline.py](src/rag_app/pipeline.py)).

## 2. Embeddings & dense retrieval

An embedding maps text to a vector positioned by *meaning*, so "how do I stop
getting 429" lands near "rate limit exceeded" despite sharing no keywords. That
is the whole advantage over keyword search — and also the weakness: dense
retrieval will happily return something topically adjacent but factually wrong.

[embed.py](src/rag_app/embed.py) normalizes every vector to unit length, so
cosine similarity is a plain dot product and the whole search is one matrix
multiply ([store.py](src/rag_app/store.py)).

> Any code writing vectors into a store directly — tests included — must
> L2-normalize first, or the ranking is silently wrong.

## 3. Embedding models (MTEB, BGE, E5)

`python -m rag_app models`

The trap is **asymmetry**. A question and the passage answering it are different
kinds of text, and the leading open models bake that into a required prefix:

| Model | Params | Dim | Max tokens | Query prefix | Passage prefix |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 384 | 256 | — | — |
| `bge-small-en-v1.5` ← active | 33M | 384 | 512 | `Represent this sentence for searching relevant passages: ` | — |
| `e5-small-v2` | 33M | 384 | 512 | `query: ` | `passage: ` |

All three are 384-dimensional because they are the same size class — `dim` is an
output of the architecture, not a setting you pick. **`max_tokens` is the number
that actually constrains the app**: text past it is dropped before embedding, so
it bounds `chunk_size`. A test fails the build if a preset crosses it.

Omit the prefix on E5 or BGE and nothing errors. You get slightly worse vectors,
slightly worse retrieval, and no signal that anything is wrong. That is why
`Embedder` exposes `encode_queries` and `encode_documents` separately instead of
one `encode`.

**MTEB** (the Massive Text Embedding Benchmark) is how you choose between them —
but read the **Retrieval** column, not the headline average. A model can top the
average on classification and clustering while being mediocre at the one job
here. Switch models in `config.yaml` and re-ingest; the `provenance.json` sidecar
refuses to load a store built with a different model rather than returning garbage.

## 4. Bi-encoder vs cross-encoder

See the module docstring in [rerank.py](src/rag_app/rerank.py).

- **Bi-encoder** — encodes query and passage *separately*, compares with cosine.
  Passages are embedded once at ingest, so querying is a matrix multiply. Fast
  and indexable. It never sees query and passage together.
- **Cross-encoder** — feeds `[query, passage]` through the transformer as one
  input and emits a relevance score. Models word-level interaction the
  bi-encoder structurally cannot. Costs a full forward pass *per pair*, so it
  can't be pre-indexed.

Hence retrieve-then-rerank: the bi-encoder narrows millions to K cheaply, the
cross-encoder sorts those K correctly.

This matters most on near-duplicate passages — three pages describing three
plan tiers, say. They sit almost on top of each other in embedding space, and
only a model reading the query *against* each passage separates them. The UI's
**④ Reranking** panel shows exactly this: which candidates the cross-encoder
promoted, which it demoted out of context, and by how many places.

## 5. Vector databases (HNSW)

Read the header of [qdrant_store.py](src/rag_app/qdrant_store.py) before quoting
any HNSW numbers, because there's a real caveat in this setup.

Brute force is O(N) per query. Fine for a few hundred chunks, fatal at 36 million. **HNSW**
(Hierarchical Navigable Small World) builds a layered proximity graph and walks
it greedily: roughly O(log N) with ~95–99% recall. The knobs:

| Parameter | Effect |
|---|---|
| `m` | Edges per node. Higher = better recall, more memory, slower build. |
| `ef_construct` | Candidate list while building. Higher = better graph, slower build. |
| `hnsw_ef` | Candidate list at query time. Higher = better recall, slower query. |

You are trading **recall for latency** — an ANN index can miss a true neighbour,
and these parameters set how often.

> **Important caveat:** `QdrantClient(path=...)` runs Qdrant *embedded* — a
> pure-Python implementation for prototyping. It accepts `hnsw_config` and
> ignores it; search is exact brute force. To actually exercise HNSW you need
> the server:
> ```bash
> docker run -p 6333:6333 qdrant/qdrant
> ```
> then set `qdrant.url: http://localhost:6333` in `config.yaml`. The application
> code is identical either way.

## 6. Similarity search & top-K

`retrieve_k: 10` then `rerank_n: 3`.

K is a recall/precision budget. Too small and the right chunk never enters the
funnel — no reranker can recover it, which is why the UI shows the retrieval
list separately from the rerank list. Too large and you feed the cross-encoder junk and pay for
it linearly.

`config.py` rejects `rerank_n > retrieve_k` at load time, because the extra
slots are unreachable and the misconfiguration is otherwise invisible.

Cosine similarity is the metric (`Distance.COSINE` on Qdrant's collection);
vectors are pre-normalized so it's equivalent to a plain dot product.

## 7. Qdrant / Chroma / pgvector

Qdrant is the only vector store this app has — no in-process numpy fallback.
That wasn't the original design: a hand-rolled numpy store (exact brute force,
zero dependencies, fully inspectable — you could open `vectors.npy` and see
exactly what was stored) shipped first, specifically so the vector-DB concepts
had something honest to be measured against. It was deliberately removed once
Qdrant alone was judged sufficient for every environment this app runs in —
the numpy code, the `backend: numpy|qdrant` config toggle, and the two-backend
test suite are gone, not just unused. `SearchBackend`
([store.py](src/rag_app/store.py)) stays as a Protocol — every caller depends
on a small explicit shape, not a `QdrantStore` import — so a second backend
still has a contract to implement if one is ever needed again.

| | Strength | Cost |
|---|---|---|
| **Qdrant** (here) | Best-in-class filtered ANN, Rust core, embedded *or* server behind the identical API | Another service to run for real HNSW |
| **Chroma** | Easiest start, embedded, hnswlib under the hood | Weaker filtering, less operable |
| **pgvector** | It's just Postgres — joins, transactions, one backup story | ANN weaker than dedicated engines at scale |

Embedded mode (the default here — see §7) means "no server to run" and "no
approximate index" simultaneously; that tradeoff doesn't change by removing
numpy. The reason to point `qdrant.url` at a real server is filtering *at
scale* with real HNSW, not raw speed on a small corpus.

## 8. Metadata filtering

Every chunk carries what its source knows about itself: `source_type`
(`doc` or `pdf`), and for a PDF page the `pdf_file` it came from and its `page`
number.

```bash
python -m rag_app ask "refund timing" --filter source_type=pdf
python -m rag_app ask "refund timing" --filter pdf_file=handbook.pdf
```

Filtering is the clean fix for a question that is genuinely ambiguous — one
whose missing information isn't in the question at all, so no reranker can
recover it. Filtering supplies it from outside.

**Pre-filter vs post-filter** is the subtle part. Exact search can pre-filter for
free: shrink the candidate set, then rank ([store.py](src/rag_app/store.py)).
A graph index can't — post-filtering (search, then discard) can return fewer
than K results or none when the filter is selective, while pre-filtering breaks
the graph's connectivity assumptions. Qdrant's answer is a *filterable* HNSW
that consults payload indexes during traversal, which is why
`create_payload_index` in [qdrant_store.py](src/rag_app/qdrant_store.py) is not
optional bookkeeping.

## 9. Grounded generation & citations

Grounding is enforced in **four** places. All four must survive any refactor.

1. **Score gate** — [pipeline.py](src/rag_app/pipeline.py). If the best
   cross-encoder score is below `score_threshold`, return `DONT_KNOW` and never
   call the LLM. `used_llm=False` is the observable proof.
2. **Restrictive prompt** — [generate.py](src/rag_app/generate.py). Excerpts
   only, exact-id citations required, explicit refusal string otherwise.
3. **Refusal detection** — a refusal that slips past the gate is stripped of its
   sources. Attaching sources to a non-answer credits documents for something
   they didn't say. Matching is punctuation-normalized, because models rewrite
   the em dash in `DONT_KNOW` constantly and an `==` check would miss it.
4. **Citation verification** — `cited_sources()` splits the model's citations
   into grounded and invented. A citation naming a source that was never in the
   context window is the clearest possible grounding failure, and it's invisible
   unless you check. It's reported as `!! HALLUCINATED CITATIONS`.

### The score gate is on a calibrated scale

The cross-encoder emits **unbounded logits** (roughly −11…+11), not a 0–1
similarity. Thresholding those directly means picking an arbitrary point on an
uninterpretable axis — and it breaks silently if you swap in a reranker that
emits probabilities.

`rerank_score_scale: sigmoid` squashes them so `score_threshold: 0.5` means "the
model considers this more relevant than not". Sigmoid is monotonic, so scaling
never reorders results — it only makes the threshold mean something. Set
`rerank_score_scale: raw` to work in logits, and `config.py` will then allow a
threshold outside 0–1.

---

## Testing

```bash
python -m pytest -q              # 161 tests, ~2s
python -m pytest tests/test_grounding.py -q
```

The suite never downloads a model, calls the LLM API, or reads a real PDF —
every test injects a fake (`FakeEmbedder`, `FakeReranker`, a stub `generate_fn`,
a stub PDF reader). `ask()` takes `embedder`, `reranker`, `generate_fn` and
`store` overrides, `run_ingest()` takes an `embedder`, and `read_pdf()` takes a
`reader_factory`, all for exactly this reason — so no binary fixtures are
checked in either.

Vector storage is the one thing tests do NOT fake: Qdrant is the only backend
the app has, so tests build real embedded `QdrantStore` instances
(`tests/conftest.py`) rather than a stand-in. It costs real time, and it means
these tests exercise the actual storage/retrieval code path rather than an
approximation of it.

## Known limitations

- Chunking is character-based, not token-based, despite `chunk_size` reading like
  tokens — and `all-MiniLM-L6-v2` truncates at 256 word-pieces (~1000 characters),
  so a larger `chunk_size` silently discards the tail of every chunk before it is
  ever embedded.
- Chunk boundaries fall at exact character counts, mid-word and mid-sentence.
  Separator-aware splitting (paragraph, then line, then sentence) would be a
  straightforward improvement.
- A PDF is cited by filename only. On a long PDF that means searching the file to
  check a claim; page-level citation was tried and traded away because it split
  facts across page breaks.
- Embedded Qdrant is brute force, not HNSW (see topic 5).
- `score_threshold` ships at a value measured on a different corpus and will not
  transfer to yours. Tune it in the UI: drag the slider and watch which questions
  flip between answered and refused.
- There is no automated evaluation harness. An earlier version scored hit@K, MRR,
  rerank lift and refusal accuracy against a gold set, but that was built around a
  structured ticket corpus and was removed with it (`git log` has it).
- `data/store/` is gitignored, so a fresh clone must `ingest` before `ask`.
