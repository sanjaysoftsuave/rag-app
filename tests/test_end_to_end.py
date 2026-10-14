"""The whole app, through its one real entry point.

Every other test exercises a module. This one drives `cli.main([...])` —
real argparse, real dispatch, real embedded Qdrant, real files on disk — with
only the models and the LLM replaced by the suite's existing fakes. It exists
because the capstone's question is whether the pieces work TOGETHER, and the
seams between them are exactly where Week 11's `ask` lost its error handling
without any module-level test noticing.

The flow is the one a stranger and the demo both follow:

    quickstart -> ask --sample -> logs --sample -> --promote-gold -> eval -> report
"""

from __future__ import annotations

from conftest import FakeEmbedder, FakeReranker, make_config

from rag_app import cli
from rag_app.evaluate import gold_path, load_gold
from rag_app.reqlog import load_log, log_path
from rag_app.sample import sample_config

REFUND_DOC = (
    "# Refund policy\n\n"
    "You can request a refund within 14 days of cancelling a subscription.\n"
)
GOLD_HEADER = "# DRAFT sample gold — the header promote-gold must not delete.\n"


def make_app(tmp_path, monkeypatch):
    """A complete, isolated app: config, sample corpus, sample gold, fakes."""
    cfg = make_config(tmp_path)
    scfg = sample_config(cfg)
    scfg.tickets_dir.mkdir(parents=True)
    (scfg.tickets_dir / "refund-policy.md").write_text(REFUND_DOC, encoding="utf-8")
    gold_path(scfg).write_text(
        GOLD_HEADER + "\n- question: What is the refund window?\n  must_contain: 14 days\n",
        encoding="utf-8",
    )

    calls = {"llm": 0}

    def fake_generate(question, contexts, cfg_, client=None, usage=None):
        calls["llm"] += 1
        if usage is not None:
            usage.add(cfg_.llm.model, 100, 20)
        return "You have 14 days after cancelling [refund-policy.md]."

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "Embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(cli, "build_reranker", lambda name: FakeReranker(0.9))
    monkeypatch.setattr("rag_app.reqlog.generate_answer", fake_generate)
    return cfg, scfg, calls


def test_the_whole_flow_from_one_entry_point(tmp_path, monkeypatch, capsys):
    cfg, scfg, calls = make_app(tmp_path, monkeypatch)

    # 1. A stranger's first command: builds the sample index and answers.
    assert cli.main(["quickstart"]) == 0
    out = capsys.readouterr().out
    assert "14 days" in out and "rq-" in out
    assert calls["llm"] == 1

    # 2. The same question again is served from the durable cache — no call.
    assert cli.main(["quickstart"]) == 0
    assert "served from cache" in capsys.readouterr().out
    assert calls["llm"] == 1

    # 3. Ask through the ordinary command, isolated to the sample.
    assert cli.main(["ask", "--sample", "--quiet", "How long for a refund?"]) == 0
    records = load_log(log_path(scfg))
    assert len(records) == 3
    assert not log_path(cfg).exists(), "--sample must never write to the real log"

    # 4. Find a request from a vague term (the support drill), then promote it.
    assert cli.main(["logs", "--sample", "refund"]) == 0
    assert records[-1].request_id in capsys.readouterr().out
    rid = records[-1].request_id
    assert cli.main(
        ["logs", "--sample", "--id", rid, "--promote-gold", "--must-contain", "14 days"]
    ) == 0
    assert gold_path(scfg).read_text(encoding="utf-8").startswith(GOLD_HEADER)
    assert len(load_gold(scfg)) == 2

    # 5. The promoted test is now part of the gold set eval grades (free run).
    assert cli.main(["eval", "--sample"]) == 0

    # 6. The report renders from whatever snapshots exist.
    out_path = tmp_path / "report.md"
    assert cli.main(["report", "--out", str(out_path)]) == 0
    assert "## Not yet measured" in out_path.read_text(encoding="utf-8")


def test_ask_with_no_index_is_one_line_and_exit_1_not_a_traceback(tmp_path, monkeypatch, capsys):
    make_app(tmp_path, monkeypatch)
    """The Week 11 seam: `ask` had lost its error handling, so a stranger's
    most likely first mistake printed a stack trace."""
    assert cli.main(["ask", "anything"]) == 1
    err = capsys.readouterr().err
    assert "No index" in err
    assert "Traceback" not in err


def test_ask_with_no_key_says_so_instead_of_a_silent_refusal(tmp_path, monkeypatch, capsys):
    cfg, scfg, calls = make_app(tmp_path, monkeypatch)
    assert cli.main(["quickstart"]) == 0
    capsys.readouterr()

    from dataclasses import replace

    keyless = replace(cfg, llm_api_key=None)
    monkeypatch.setattr(cli, "load_config", lambda: keyless)
    before = calls["llm"]
    assert cli.main(["ask", "--sample", "--quiet", "How long for a refund?"]) == 0
    captured = capsys.readouterr()
    assert "No OPENROUTER_API_KEY" in captured.err
    assert "generation skipped" in captured.out
    assert calls["llm"] == before  # nothing was sent
    # Found running quickstart keyless for real: the summary said used_llm=True
    # and warned that "the model" produced no citations — about a model that
    # was never called. A stranger reads that as the app being broken.
    assert "used_llm=skipped" in captured.out
    assert "no citations" not in captured.out


def test_a_dry_replay_never_writes_to_the_request_log(tmp_path, monkeypatch, capsys):
    """A dry run measures shape; it must not leave '(dry run)' answers in the
    real log for every later --cost-report to average in."""
    cfg, scfg, calls = make_app(tmp_path, monkeypatch)
    assert cli.main(["quickstart"]) == 0
    n_before = len(load_log(log_path(scfg)))
    assert cli.main(["logs", "--sample", "--replay-gold"]) == 0
    assert len(load_log(log_path(scfg))) == n_before


def test_doctor_runs_offline_and_names_a_next_step(tmp_path, monkeypatch, capsys):
    make_app(tmp_path, monkeypatch)
    cli.main(["doctor"])
    assert "Next:" in capsys.readouterr().out
