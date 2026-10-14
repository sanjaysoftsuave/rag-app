# Ask My Documents

Ask a question about your own documents and get an answer built only from them,
with the file it came from in brackets — or an explicit "I don't know" when the
answer isn't there. Plain Python, no LangChain/LlamaIndex on the answering path,
and a browser UI that shows every step for a given question, not just the answer.

```
your .md / .txt / .pdf  ->  chunk -> embed -> Qdrant -> top 10 -> rerank -> top 3
                                                                        |
                         score too low? refuse, never call the model <--+--> LLM writes a cited answer
```

## What you need

- **Python 3.11 or newer** and `git`.
- **About 2 GB of disk** for PyTorch and two small local models. The models
  (~150 MB) download on first run; after that, everything except writing the
  answer runs on your machine.
- **An OpenRouter API key** for the answer itself. Sign up at
  [openrouter.ai](https://openrouter.ai), add a few dollars of credit, and create
  a key under *Keys*. A question costs a fraction of a cent on the default model
  (`openai/gpt-4o-mini`). Without a key the app still runs retrieval and shows
  you what it found; it just doesn't write an answer.

## Quick start

**macOS / Linux**

```bash
git clone https://github.com/sanjaysoftsuave/rag-app.git
cd rag-app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ui,dev]"
cp .env.example .env              # then open .env and paste your key after OPENROUTER_API_KEY=
python -m rag_app doctor
python -m rag_app quickstart
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/sanjaysoftsuave/rag-app.git
cd rag-app
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,dev]"
copy .env.example .env            # then open .env and paste your key after OPENROUTER_API_KEY=
python -m rag_app doctor
python -m rag_app quickstart
```

`doctor` checks your setup without loading a model or touching the network, and
ends with the exact command to run next. `quickstart` builds a small index over
six **synthetic** support documents bundled in [data/sample/](data/sample/) and
answers one question. Your own documents are never touched by it. The output
has this shape (the wording, timing and token count will be your own run's):

```
Q: How long do I have to request a refund after cancelling?

Answer: <a sentence stating the refund window, ending in [refund-policy.md]>
Sources: refund-policy.md
(K=10 → N=3; best_score=0.9998; threshold=0.2; used_llm=True; gate=answered)

[rq-<timestamp>-<id>  <seconds>  <tokens> tokens]
```

Every request gets an id like that and one line in a log you can search later.
If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation
and call `.venv\Scripts\python.exe` directly.

The first run takes a minute or two while the models download. Then try your own
question against the sample: `python -m rag_app ask --sample "When does a password reset link expire?"`

## Use your own documents

```bash
python -m rag_app ui
```

In the browser: **Add documents** → drop in `.md`, `.txt` or `.pdf` files →
**Run ingest** → **Ask**. Files land in `data/tickets/` (gitignored). Each answer
cites a filename, e.g. `[handbook.pdf]`. A scanned PDF with no text layer yields
nothing — there is no OCR, and ingest tells you so.

After that the terminal works too: `python -m rag_app ask "your question"`.

## Measure it

Commands with **`--generate` call the model and cost money**; each prints how many
calls it will make before the first one. Everything else is free.

| Command | What it tells you | Cost |
|---|---|---|
| `eval` | Did retrieval find the right text? (questions in `data/gold.yaml`, format in [gold.example.yaml](gold.example.yaml)) | free |
| `eval --generate --snapshot NAME` | ...and did the answer contain the fact? Saved for comparison. | 1 call / question |
| `debug --generate` | For each miss: wrong text retrieved, or right text misused? | 1 call / question |
| `logs [WORDS]` / `logs --id ID` | Find any past request by a vague term, replay it in full | free |
| `logs --id ID --promote-gold --must-contain TEXT` | Turn a bad answer into a permanent test | free |
| `logs --cost-report` | Mean/p99 latency, tokens, cost, cache hit rate | free |
| `compare BEFORE AFTER` | What changed between two snapshots, and whether it is beyond noise | free |
| `report` | Render every saved snapshot into [docs/EVAL-REPORT.md](docs/EVAL-REPORT.md), labelled real model or stand-in | free |

Add `--sample` to `ask`, `eval`, `debug`, `logs` or `chunks` to run against the
bundled sample instead of your own corpus. `python -m rag_app --help` lists the
rest (agents, MCP, red-teaming).

## Troubleshooting

| You see | Do this |
|---|---|
| `No index ...` | Build one: `quickstart` for the sample, or the UI's **Run ingest** for your files. |
| `No OPENROUTER_API_KEY in .env` | `.env` must sit in the repo root, next to `config.yaml`, with `OPENROUTER_API_KEY=sk-or-...` on one line. |
| `The request failed: AuthenticationError` | The key is wrong or revoked. Create a new one on OpenRouter. |
| `The request failed: APIStatusError ... 402` | The key works but the account has no credit. |
| Everything answers "I don't know" | `score_threshold` in `config.yaml` is tuned for another corpus. Drag the slider in the UI's sidebar and watch which questions flip. |
| `Fatal error in launcher` running `pytest` or `pip` | Always use `python -m pytest`, `python -m pip`. |
| First run is slow | It is downloading the models. Later runs load them from cache in seconds. |
| Garbled characters on Windows | Use Windows Terminal or PowerShell; the CLI already switches its own output to UTF-8. |

## Tests

```bash
python -m pytest -q        # ~800 tests, under 20 s, fully offline: no model downloads, no API calls
```

## Read next

- [docs/WEAK-SPOTS.md](docs/WEAK-SPOTS.md) — what the numbers do *not* prove. Read this before quoting any of them.
- [docs/EVAL-REPORT.md](docs/EVAL-REPORT.md) — every measured number, generated from the snapshot files.
- [docs/DEMO.md](docs/DEMO.md) — a ten-minute walkthrough, including a real failure.
- [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) — the 18-topic curriculum: why each stage exists and what was measured.
- [data/redteam/RESIDUAL-RISK.md](data/redteam/RESIDUAL-RISK.md) — prompt injection: what still gets through.
- [CLAUDE.md](CLAUDE.md) — architecture and design decisions, in depth.
