# The bundled sample corpus

`corpus/` holds six short, **synthetic** support documents for a made-up
product, Northwind Cloud. They exist so someone who has just cloned the repo
can get a real, cited answer in a few minutes without bringing documents of
their own:

```bash
python -m rag_app quickstart
python -m rag_app ask --sample "When does a password reset link expire?"
```

Nothing here is real policy. Each document says so in its second line.

**Isolated structurally.** The sample builds into its own index
(`data/store/sample/`), and its gold set, request log, snapshots and answer
cache all live under this folder — `sample_config()` redirects both the corpus
and the store, exactly as the red-team corpus does. Using `--sample` can never
touch your real index or your real logs.

`gold.yaml` is the sample's own gold set; see its header for what it can and
cannot tell you.
