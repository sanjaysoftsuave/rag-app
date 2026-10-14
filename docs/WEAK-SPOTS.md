# Weak spots

What this app does **not** prove, in one place. Each entry says what the
consequence is, because a limit with no consequence attached is a disclaimer, not
information. The prompt-injection limits have their own, longer file:
[data/redteam/RESIDUAL-RISK.md](../data/redteam/RESIDUAL-RISK.md).

Numbers quoted elsewhere are rendered from snapshot files by
`python -m rag_app report` into [EVAL-REPORT.md](EVAL-REPORT.md), with a label
saying whether each came from a real model. If a number in prose disagrees with
that file, the file is right.

## The measurements rest on a small, unaudited base

| Limit | Consequence |
|---|---|
| The gold set is **13 questions, 9 answerable**. | One answerable question flipping moves any answerable-only rate by ~11 points. A difference smaller than that between two runs is noise, and `compare` says so. |
| `data/gold.yaml` and `data/agent_tasks.yaml` are **DRAFT**: written from the extracted text, not checked by a human against the source. | Accuracy currently means agreement with whoever wrote the gold set. A wrong `must_contain` makes a correct answer look like a failure. |
| The agent race is **8 tasks, 5 answerable**. | `single-v1` 80% vs `team-v1` 60% is one task of difference. The team being worse is consistent with the cost numbers (2x tokens, 2x dollars) but its quality gap alone is not a trend. |
| The 20-trace labelling sheet for judge validation is **empty**. | The LLM judge has never been checked against a human. `--judge`/`--geval` numbers should not be quoted as accuracy until it has been. |
| When it is filled in, it will be **one person's** labels. | "Judge agreement" will mean agreement with that person, not with ground truth. |
| The gold set is **too easy**. With the model on, `debug --generate` found zero retrieval failures and zero generation failures across all 9 answerable questions. Gate scores split into 0.0000 and 0.94+, with nothing in between. | A test set that never fails can't rank two changes. The one real failure on the main path (below) came from an *unanswerable* question. |
| The sample corpus and its gold set were written by the **same author**. | `quickstart` and `--sample` prove the pipeline works end to end. They do not prove it finds things a stranger would ask about. |

## The score gate can be fooled by question shape

"What did ticket CS-9999 conclude?" names a ticket that doesn't exist, and its
best excerpt scores **0.9755**. That is above the 0.20 gate, and above a real
answerable question (0.9447). So the gate lets it through, and only the model's
own refusal stops it (`gate=model-refused`, request `rq-20261014T105846-4b30b6`).
No threshold separates the two questions. Consequence: questions about
nonexistent entities cost a model call, and the app depends on the model
refusing. A cheap fix, not yet built: refuse before the gate when the question
names an identifier that appears nowhere in the corpus. The full story is in
[DEMO.md](DEMO.md), part 2.

## Numbers that came from a stand-in, not a model

| Limit | Consequence |
|---|---|
| The red-team result, injection success **71.4% -> 42.9%**, came from the scripted `obedient_llm` stand-in. Both snapshots say `generated: false`. | It shows the defences *engage* on a model that obeys anything legible in its prompt. It does not show a real model would have followed the payloads that got through, or ignored the ones that did not. |
| The injection detector is regex-based. | It catches marked imperatives and forged headers. It cannot catch paraphrase, encoding, other languages, or an instruction split across a chunk boundary. |
| RAGAS absolute values do not transfer across corpora. Answer relevancy's floor is ~0.3-0.6, not 0. | Only the same metric on the same questions before and after a change means anything. |
| G-Eval samples 5 times instead of reading logprobs, because OpenRouter does not reliably forward them. | It estimates the same quantity, at 5x the calls, with a reported standard deviation. |

## Things the app simply does not do

| Limit | Consequence |
|---|---|
| **No OCR.** | A scanned PDF yields zero chunks. Ingest says so loudly rather than succeeding at nothing. |
| Chunking counts **characters**, not tokens, and cuts at exact counts, mid-sentence. | `chunk_size` reads like tokens and is not. A sentence split across two chunks can be half-retrieved. |
| A PDF is cited by **filename**, not page. | Checking a claim on a long PDF means searching the file. Page citations were tried and traded away: they split facts across page breaks. |
| Embedded Qdrant (the default) is **brute force, not HNSW**. | Fine at this size. Any latency number from embedded mode says nothing about a real vector database. |
| `score_threshold` (0.20) was calibrated on a different corpus. | On your documents it may refuse real answers or let weak excerpts through. Tune it in the UI. |
| The answer cache is **exact-match only**, in one JSON file, single writer. | A reworded question is a miss. Two processes writing at once can lose an entry. Re-ingest clears it. |
| The MCP HTTP transport has **no auth**. | Run it only on a network path you already trust. |
| Team specialists run **sequentially**. | The team's latency is a worst case. Parallel specialists would close some of the time gap, not the token gap. |
| **One model for everything** (`gpt-4o-mini`), except the judge. | No routing to a cheaper tier, because there is no honest cheaper tier to route to. |
| The agent's prompt budget counts characters (~4 per token). | It is a ceiling, not a token count. Real token counts come from the API's `usage` field and are reported separately. |
| Agent memory rebuilds its index on every write. | Invisible at hundreds of turns. It would need a real upsert path at tens of thousands. |
| mem0 cannot run offline. | Its tests cover the interface shape and the import guard only. |

## What would move the most, in order

1. **Refuse unknown identifiers before the gate**: the CS-9999 case above.
   Deterministic, offline-testable, and it removes a model call the gate should
   have prevented.
2. **Fill in the 20 labels** (`python -m rag_app judge --init`, then edit the sheet),
   then run `judge`. Until then every LLM-judged number is unvalidated.
3. **Audit `data/gold.yaml`** against the PDF, then grow it. Thirty answerable
   questions would bring one question's worth of noise down to ~3 points.
   Add questions that are *hard*: near-miss ids, facts split across two
   tickets, paraphrases with no shared words.
4. **Run the red-team suite against a real model** (`redteam --generate`) so the
   injection numbers stop being a statement about a stand-in.
