# Demo script

About ten minutes, four parts. Every command below has been run, and every
number is copied from a file on disk named next to it. Nothing here is the
planted drill entry (`logs --plant-bad-answer`), which is fabricated on purpose
and labelled as such in its own record.

**Before you start:** the real index is built (UI → *Run ingest*), `.env` has a
key, and `python -m rag_app doctor` is all PASS. Run `python -m rag_app report`
so [EVAL-REPORT.md](EVAL-REPORT.md) is current, and keep it open in a tab.

---

## 1. A real run (2 min)

```bash
python -m rag_app ask "Why was the MFA code never delivered by SMS?"
```

Point at, in order: the answer and its `[RAG-Test-Document-1.pdf]` citation; the
retrieval list with scores; the rerank table; `gate=answered`; the request id;
the per-stage times; tokens and cost.

If you have already run the cache pair (part 3), this question is served from
cache and the line says `served from cache — $0`. Say so; that is the feature
working. To show a fresh model call, reword the question.

Then show it was recorded: `python -m rag_app logs "MFA"`.

## 2. A real failure, and how it is handled (4 min)

**The question:** *"What did ticket CS-9999 conclude?"* There is no ticket
CS-9999; the document's tickets run CS-1001 to CS-1009. The app should say it
doesn't know.

**What happened** (request `rq-20261014T105846-4b30b6`, from the real
`cache-cold` run):

```bash
python -m rag_app logs --id rq-20261014T105846-4b30b6
```

```
Q: What did ticket CS-9999 conclude?
A: I don't know — that information is not in the provided documents.
sources: (none)   gate=model-refused   cache_hit=False
usage: 680 tokens, cost=$0.000108
```

The right answer, but look at `gate=model-refused`. **The first line of
defence failed.** The score gate is supposed to stop an unanswerable question
*before* the model is called. Here the cross-encoder gave the best excerpt
**0.9755** (the gate is 0.20). That excerpt is the *Resolution:* line of a
different ticket. The reranker judged "what did X conclude" against text that
looks like a conclusion, and the ticket id barely counted.

**Why a higher threshold can't fix it.** These are the gate scores of every
question in `data/gold.yaml`, measured with the model switched off:

| Best rerank score | Question | Should |
|---|---|---|
| 0.0000 | weekend support hours / Enterprise price / weather | refuse |
| **0.9447** | What should a CSV file look like to import successfully? | **answer** |
| **0.9755** | **What did ticket CS-9999 conclude?** | **refuse** |
| 0.9799 – 0.9999 | the other seven answerable questions | answer |

Any threshold that refuses CS-9999 also refuses the CSV question. The gate
cannot separate them, because a relevance score measures how much a passage is
*about* the question, and a nonexistent ticket's question is about tickets.

**What caught it** was the third guarantee: the prompt restricts the model to the
excerpts, the model said it didn't know, and `is_refusal()` stripped the
sources. Defence in depth worked, at a cost of one model call that should never
have happened.

**It is already a permanent test.** It sits in `data/gold.yaml` as
`unanswerable: true`, and the free eval shows the gate letting it through on
every run:

```bash
python -m rag_app compare main-v1 main-v1-fold
#   refusal_accuracy   1 -> 0.75    (generated True -> False)
```

With the model, refusal accuracy is 100%. With the gate alone it is 75%: one of
the four unanswerable questions gets past the gate, and it is this one. The
fix this points at (in *What's next*) is a cheap check before the gate. If the
question names an identifier like `CS-9999` that appears nowhere in the corpus,
refuse without calling the model.

**To show the loop on a new failure:** find it with `logs "<vague words>"`,
then
`logs --id ID --promote-gold --must-contain "<the right fact>"` (or
`--unanswerable`), then `eval --generate`. The entry is graded from then on.

### Backup failures, with evidence on disk

Use one of these if there's time or if asked for another.

1. **A metric that was wrong, found today.** `main-v1` reported recall@3 =
   90.9%, i.e. one expected snippet ("18 widgets") never reached the model.
   It had: PDF extraction stores it as `18\n\nwidgets`, and the matcher didn't
   fold whitespace. The fix and its regression test are
   `evaluate.fold`; `compare main-v1 main-v1-fold` shows recall 0.909 → 1.0. The
   question had still been answered correctly, which is what prompted the check.
2. **A number in the docs with no run behind it.** In Week 11 the docs said the
   workflow path costs $0.0021 per request. It had never been measured. The
   figure was the *agent's* cost from a different snapshot, and the command it
   was attributed to (`logs --replay-gold`) crashed every time it ran. The real
   cold-cache figure is **$0.000101** (`cache-cold`), about 20× lower. The
   structural fix: every snapshot now records `generated`, and `report` renders
   the numbers from files with a real/stand-in label, so prose no longer types them.
3. **The multi-agent team lost.** `compare single-v1 team-v1`: task success
   80% → 60%, mean cost $0.00212 → $0.00443, mean tokens 11.4k → 23.4k. That
   is one task of difference out of five answerable, so the quality gap alone
   is noise; the 2× cost is not. Verdict: keep the single agent.

## 3. A number that improved (2 min)

```bash
python -m rag_app compare cache-cold cache-warm
```

Same 13 gold questions, real model, run twice:

| | cache-cold | cache-warm |
|---|---|---|
| cache hit rate | 0% | 76.9% |
| mean cost per request | $0.000101 | $0 |
| mean tokens per request | 557 | 0 |
| mean latency | 2.14 s | 0.11 s |

Why 76.9% and not 100%: 10 of 13 hit. The other 3 are the questions the gate
refuses, which never call the model and so never get cached. The cache key
is (question, retrieved chunk ids, model, preset), so a re-ingest that changes
what is retrieved can't serve a stale answer, and the UI clears the cache on
every ingest.

Be straight about the size of the win. The cache only pays when the *exact*
question repeats. On a real support desk that is the FAQ fraction of traffic,
not 77%. Latency on the warm run is a laptop measurement, not a claim.

## 4. What's next (1 min)

From [WEAK-SPOTS.md](WEAK-SPOTS.md), in order of what would move most:

1. **An unknown-identifier check before the gate** — the CS-9999 fix above.
   Cheap, deterministic, and testable offline.
2. **Fill in the 20 judge labels** and run `judge`. Until then no LLM-judged
   number is validated.
3. **Audit and grow the gold set.** It is DRAFT, and on the main path it found
   **zero** retrieval or generation failures across 9 answerable questions
   (`debug --generate`). A test set that never fails is too easy to tell you anything.

---

## Tough questions

**"What does a request cost?"** $0.000101 mean, $0.000168 p99, on
`gpt-4o-mini`, over the 13 gold questions with an empty cache (`cache-cold`).
A refused-by-gate question costs $0. Token counts come from the API's own
`usage` field; dollars come from a dated price table in `config.yaml`, so
re-check the rates before quoting them as current.

**"How accurate is it?"** On the 9 answerable gold questions: 100% answer
accuracy (substring match), 100% recall@3, 4/4 refusals (`main-v1`). Then say
the caveats in the same breath. One question is worth 11 points, so this can't
tell 100% from 90%. The gold set is a DRAFT nobody has audited. And substring
matching is crude: it can pass an answer that negates the fact.

**"So why not use the LLM judge?"** It exists (`eval --judge`), but it has never
been validated. The 20-trace label sheet it would be checked against is empty.
An unvalidated judge is a second opinion from a model, not a measurement.

**"What happens when it's wrong?"** Four layers, in order. (1) The score gate
refuses before calling the model. (2) The prompt limits the model to the
excerpts and requires a citation. (3) A refusal gets its sources stripped.
(4) Any citation naming a source the model wasn't shown is flagged as
hallucinated. Every request is logged with an id, and a bad one becomes a gold
test with `logs --promote-gold`. Part 2 is a real case where layer 1 failed and
layer 2 caught it.

**"Is it safe against prompt injection?"** Partly, and the numbers are not from
a real model. Injection success went 71.4% → 42.9% with the defences on, but
measured against a scripted stand-in that obeys anything left in its prompt
(both snapshots say `generated: false`). Three attacks still land; they are
entries 2–4 in [RESIDUAL-RISK.md](../data/redteam/RESIDUAL-RISK.md).

**"Why should I believe any number in your docs?"** Because they aren't typed.
`report` renders them from snapshot files, and each snapshot says whether a
real model produced it. That rule exists because a typed number was wrong
once (backup failure 2).

**"Why not a framework?"** Each stage is a function you can print, the suite
runs offline in ~17 s, and the one place a framework was tried (the LangGraph
agent) is kept beside the plain one so its cost could be measured rather than
argued about.
