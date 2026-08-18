# Ask My Support Tickets

A plain-Python RAG app over a customer-support help-centre drop. Ask a question,
get an answer built only from the tickets, with the ticket id it came from — and
an explicit "I don't know" when the answer isn't there.

No LangChain, no LlamaIndex, no web UI. Every retrieval step is a function you
can read and print.

```
tickets.jsonl → chunk → embed → vector store → top-K → cross-encoder rerank → score gate → LLM
                  ↑                    ↑                        ↑                  ↑
             3 strategies      numpy or Qdrant         reranks near-dups     refuses instead
                                + metadata filter                            of guessing
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env          # then set OPENROUTER_API_KEY

python -m rag_app chunks --all              # see chunking differences, no models needed
python -m rag_app ingest --all              # build all four stores
python -m rag_app ask "What is the rate limit on the Free plan?"
python -m rag_app eval --all                # score retrieval + the refusal gate
```

## Commands

| Command | What it does |
|---|---|
| `chunks --all [--show N]` | Chunk statistics per preset, including boundary bleed. No embedding, so it's instant. |
| `ingest [--preset X \| --all]` | Load → chunk → embed → persist. Reports bleed for `flat` presets. |
| `ask "..." [--preset X] [--filter k=v] [--quiet]` | Answer a question. Repeat `--filter` to narrow by metadata. |
| `compare "..." [--generate]` | Same question across every preset, side by side, with a summary table. Retrieval-only unless `--generate`. |
| `eval [--all]` | hit@K, MRR, rerank lift, refusal accuracy against the gold set. |
| `models` | The embedding-model registry and each family's prefix rules. |

## The four presets

`A` and `B` differ **only** in chunk size, so the size effect is isolated.
`C` and `D` change the strategy itself.

| Preset | Strategy | Size | Overlap | What it demonstrates |
|---|---|---|---|---|
| A | `flat` | 500 | 50 | Naive character windows over the whole corpus |
| B | `flat` | 1000 | 100 | Same, larger windows |
| C | `ticket` | 2000 | 200 | One chunk per ticket — the right default here |
| D | `section` | 700 | 0 | Turns packed under a repeated ticket header |

---

# The topics, and where each one lives in this repo

## 1. Why RAG

A language model answers from its training data. Ask it about *your* help centre
and it will produce something fluent and invented, because "I don't know" is a
rare token in its training distribution. RAG replaces recall with reading: find
the right text first, then let the model write only from that text.

Two things make it *grounded* rather than just "context-stuffed": a citation the
user can check ([generate.py](src/rag_app/generate.py)), and a refusal path that
runs before the model does ([pipeline.py](src/rag_app/pipeline.py)).

## 2. Chunking strategies

Read [chunking.py](src/rag_app/chunking.py). Three strategies, and the choice
matters more than any parameter in this repo.

Support tickets have hard semantic boundaries. A fixed character window does not
know that, so it produces chunks that begin inside one ticket and end inside the
next. The chunk is then cited as whichever ticket it *starts* in — a citation
that is wrong for half its own content.

Measured on the shipped 36-ticket corpus:

```
preset  strategy  size    overlap   chunks   avg chars   boundary bleed
A       flat      500     50        60       492         42/60 (70%)
B       flat      1000    100       30       984         29/30 (97%)
C       ticket    2000    200       36       732         0/36 (0%)
D       section   700     0         62       476         0/62 (0%)
```

Run `python -m rag_app chunks --all --show 3` to see the bleeding chunks
verbatim.

## 3. Chunk size & overlap

The counterintuitive result above is the thing worth understanding: **going from
500 to 1000 characters made bleeding worse, not better** — 70% → 97%.

Larger windows span more ticket boundaries, not fewer. The instinct "bigger
chunks preserve more context" is right for prose with no hard boundaries and
exactly wrong for a corpus of discrete records. Structure beats size: preset C
uses the *largest* chunks of all and bleeds 0%, because its boundaries are the
document's own.

Overlap exists to stop a fact being split across two chunks so neither contains
it whole. It costs storage and duplicates content into your top-K (two
overlapping chunks can both surface, wasting a slot). With per-ticket chunking
overlap is nearly redundant — the boundary is already in the right place.

## 4. Embeddings & dense retrieval

An embedding maps text to a vector positioned by *meaning*, so "how do I stop
getting 429" lands near "rate limit exceeded" despite sharing no keywords. That
is the whole advantage over keyword search — and also the weakness: dense
retrieval will happily return something topically adjacent but factually wrong.

[embed.py](src/rag_app/embed.py) normalizes every vector to unit length, so
cosine similarity is a plain dot product and the whole search is one matrix
multiply ([store.py](src/rag_app/store.py)).

> Any code writing vectors into a store directly — tests included — must
> L2-normalize first, or the ranking is silently wrong.

## 5. Embedding models (MTEB, BGE, E5)

`python -m rag_app models`

The trap is **asymmetry**. A question and the passage answering it are different
kinds of text, and the leading open models bake that into a required prefix:

| Model | Params | Query prefix | Passage prefix |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | — | — |
| `bge-small-en-v1.5` | 33M | `Represent this sentence for searching relevant passages: ` | — |
| `e5-small-v2` | 33M | `query: ` | `passage: ` |

Omit the prefix on E5 or BGE and nothing errors. You get slightly worse vectors,
slightly worse retrieval, and no signal that anything is wrong. That is why
`Embedder` exposes `encode_queries` and `encode_documents` separately instead of
one `encode`.

**MTEB** (the Massive Text Embedding Benchmark) is how you choose between them —
but read the **Retrieval** column, not the headline average. A model can top the
average on classification and clustering while being mediocre at the one job
here. Switch models in `config.yaml` and re-ingest; `meta.json` refuses to load
a store built with a different model rather than returning garbage.

## 6. Bi-encoder vs cross-encoder

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

The corpus contains three near-identical rate-limit tickets (Free 60/min, Pro
600/min, Enterprise 6000/min). They sit almost on top of each other in embedding
space. Only a model reading "Free plan" *against* the question separates them —
which is exactly what `eval` measures as **rerank lift**.

## 7. Vector databases (HNSW)

Read the header of [qdrant_store.py](src/rag_app/qdrant_store.py) before quoting
any HNSW numbers, because there's a real caveat in this setup.

Brute force is O(N) per query. Fine for 36 tickets, fatal at 36 million. **HNSW**
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

## 8. Similarity search & top-K

`retrieve_k: 10` then `rerank_n: 3`.

K is a recall/precision budget. Too small and the right chunk never enters the
funnel — no reranker can recover it, which is why `eval` reports **hit@K**
separately from top-1. Too large and you feed the cross-encoder junk and pay for
it linearly.

`config.py` rejects `rerank_n > retrieve_k` at load time, because the extra
slots are unreachable and the misconfiguration is otherwise invisible.

Cosine similarity is the metric; vectors are pre-normalized so it's a dot
product.

## 9. Qdrant / Chroma / pgvector

This repo implements **numpy** and **Qdrant** behind one `SearchBackend`
protocol ([store.py](src/rag_app/store.py)), switchable with `backend:` in
`config.yaml`. Same filters, same pipeline, same tests.

| | Strength | Cost |
|---|---|---|
| **numpy** (here) | Exact, zero deps, fully inspectable | O(N) — dies at scale |
| **Qdrant** | Best-in-class filtered ANN, Rust, embedded or server | Another service to run |
| **Chroma** | Easiest start, embedded, hnswlib under the hood | Weaker filtering, less operable |
| **pgvector** | It's just Postgres — joins, transactions, one backup story | ANN weaker than dedicated engines at scale |

The honest default for a project this size is the numpy store. The reason to
reach for a real vector DB is filtering at scale, not raw speed.

## 10. Metadata filtering

Every chunk carries its ticket's metadata — `product`, `category`, `status`,
`priority`, `customer_tier`, `created_at`, `tags`
([tickets.py](src/rag_app/tickets.py)).

```bash
python -m rag_app ask "what is my rate limit?" --filter customer_tier=free
python -m rag_app ask "refund timing" --filter product=Billing --filter status=resolved
```

This is the clean fix for the near-duplicate problem: "what's my rate limit"
is genuinely ambiguous across plans, and no reranker can resolve it, because the
missing information isn't in the question. Filtering supplies it.

**Pre-filter vs post-filter** is the subtle part. Exact search can pre-filter for
free: shrink the candidate set, then rank ([store.py](src/rag_app/store.py)).
A graph index can't — post-filtering (search, then discard) can return fewer
than K results or none when the filter is selective, while pre-filtering breaks
the graph's connectivity assumptions. Qdrant's answer is a *filterable* HNSW
that consults payload indexes during traversal, which is why
`create_payload_index` in [qdrant_store.py](src/rag_app/qdrant_store.py) is not
optional bookkeeping.

## 11. Grounded generation & citations

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
   into grounded and invented. A citation naming a ticket that was never in the
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

## Evaluation

`python -m rag_app eval --all` scores 16 answerable questions (each with a known
correct ticket) and 4 that are deliberately absent from the corpus.

- **hit@K** — did the right ticket enter the funnel at all? A ceiling on
  everything downstream.
- **top-1 bi-encoder vs top-1 reranked** — the difference is the rerank lift,
  the number that justifies the cross-encoder's cost.
- **MRR** — how high the right ticket ranked, on average.
- **refusal accuracy** — of the questions with no answer in the corpus, how many
  were refused.
- **false refusals** — answerable questions wrongly gated out. Read this
  *together* with refusal accuracy: a gate that refuses everything scores 100%
  on refusals and is useless.

Add cases in `GOLD` in [evaluate.py](src/rag_app/evaluate.py). A test asserts
every gold question points at a ticket that actually exists, so the set can't
rot silently.

## Testing

```bash
pytest -q                        # 67 tests, ~1s
pytest tests/test_grounding.py -q
```

The suite never downloads a model or calls an API — every test injects a fake.
That's a hard constraint, not a convenience: it's the only reason it's fast
enough to run on every change. `ask()` takes `embedder`, `reranker`,
`generate_fn` and `store` overrides, and `run_ingest()` takes an `embedder`, for
exactly this reason.

## Known limitations

- Chunking is character-based, not token-based, despite `chunk_size` reading
  like tokens.
- Embedded Qdrant is brute force, not HNSW (see topic 7).
- The corpus is 36 synthetic tickets. Thresholds tuned on it won't transfer to a
  real drop — re-run `eval` after swapping in real data.
- `data/store/` is gitignored, so a fresh clone must `ingest` before `ask`.
