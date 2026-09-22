import pytest

from rag_app.cli import build_parser
from rag_app.filters import MetaFilter


def test_all_commands_parse():
    parser = build_parser()
    for cmd in ("ui", "ask", "chunks", "models", "eval", "debug", "codes"):
        argv = [cmd] + (["q"] if cmd == "ask" else [])
        assert parser.parse_args(argv).command == cmd


def test_every_subcommand_has_a_handler():
    """A subparser with no entry in main()'s dispatch dict fails only at runtime."""
    import argparse

    from rag_app import cli

    parser = build_parser()
    subparsers = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ][0]
    registered = set(subparsers.choices)
    handled = set(cli._HANDLERS)
    assert registered == handled, (
        f"parser and dispatch disagree: only in parser {sorted(registered - handled)}, "
        f"only in dispatch {sorted(handled - registered)}"
    )


def test_ingest_has_no_cli_command():
    """Building the index is a UI action, deliberately in one place only."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest"])


def test_ui_flags_and_defaults():
    defaults = build_parser().parse_args(["ui"])
    assert defaults.port == 8501
    assert defaults.headless is False
    args = build_parser().parse_args(["ui", "--port", "9000", "--headless"])
    assert args.port == 9000
    assert args.headless is True


def test_chunks_flags():
    assert build_parser().parse_args(["chunks", "--show", "3"]).show == 3
    assert build_parser().parse_args(["chunks", "--all"]).all is True
    assert build_parser().parse_args(["chunks"]).show == 0


def test_ask_accepts_repeated_filters():
    args = build_parser().parse_args(
        ["ask", "what is the refund window?", "--filter", "source_type=pdf",
         "--filter", "pdf_file=handbook.pdf"]
    )
    flt = MetaFilter.parse(args.filter)
    assert flt.must == {"source_type": "pdf", "pdf_file": "handbook.pdf"}


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask"])


def test_codes_flags_and_defaults():
    defaults = build_parser().parse_args(["codes"])
    assert defaults.init is False
    assert defaults.json is False
    assert defaults.traces is None
    assert defaults.labels is None


def test_codes_init_is_explicit():
    """Writing the blank sheet must be asked for; a bare `codes` only reads."""
    assert build_parser().parse_args(["codes", "--init"]).init is True


# ---------------------------------------------------------------------------
# Week 6 surface: eval scoring flags, judge, compare
# ---------------------------------------------------------------------------


def test_eval_scoring_flags_default_off():
    """`eval` with no flags makes zero LLM calls, forever."""
    a = build_parser().parse_args(["eval"])
    assert a.judge is False and a.geval is False and a.ragas is False
    assert a.generate is False
    assert a.snapshot is None
    assert a.limit == 0


def test_judge_requires_generate_and_the_message_says_why():
    from rag_app.cli import check_eval_args

    args = build_parser().parse_args(["eval", "--judge"])
    msg = check_eval_args(args)
    assert msg is not None
    assert "--generate" in msg
    assert "no answer to score" in msg


def test_every_scoring_flag_requires_generate():
    from rag_app.cli import check_eval_args

    for flag in ("--judge", "--geval", "--ragas"):
        assert check_eval_args(build_parser().parse_args(["eval", flag])) is not None


def test_scoring_with_generate_is_accepted():
    from rag_app.cli import check_eval_args

    assert check_eval_args(
        build_parser().parse_args(["eval", "--generate", "--judge", "--ragas"])
    ) is None


def test_a_negative_limit_is_rejected():
    from rag_app.cli import check_eval_args

    assert "cannot be negative" in check_eval_args(
        build_parser().parse_args(["eval", "--limit", "-1"])
    )


def test_snapshot_takes_a_label():
    assert build_parser().parse_args(["eval", "--snapshot", "before"]).snapshot == "before"


def test_compare_requires_two_snapshots():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["compare", "only-one"])
    a = build_parser().parse_args(["compare", "before", "after"])
    assert (a.before, a.after) == ("before", "after")


def test_judge_flags_and_defaults():
    a = build_parser().parse_args(["judge"])
    assert a.init is False and a.json is False
    assert a.traces is None and a.labels is None


def test_atasks_snapshot_takes_a_label_and_a_note():
    a = build_parser().parse_args(["atasks", "--snapshot", "before", "--note", "why"])
    assert a.snapshot == "before"
    assert a.note == "why"
    assert build_parser().parse_args(["atasks"]).snapshot is None


def test_atasks_snapshot_requires_generate_and_says_why():
    """A dry run stops every task 'unparseable', so snapshotting it would record
    the harness's shape as if it were the agent's behaviour."""
    from rag_app.cli import check_atasks_args

    msg = check_atasks_args(build_parser().parse_args(["atasks", "--snapshot", "x"]))
    assert msg is not None
    assert "--generate" in msg
    assert "harness's shape" in msg


def test_atasks_snapshot_with_generate_is_accepted():
    from rag_app.cli import check_atasks_args

    assert check_atasks_args(
        build_parser().parse_args(["atasks", "--snapshot", "x", "--generate"])
    ) is None


def test_a_bare_atasks_run_needs_no_generate():
    from rag_app.cli import check_atasks_args

    assert check_atasks_args(build_parser().parse_args(["atasks"])) is None


def test_eval_snapshot_without_generate_is_still_legitimate():
    """Unlike atasks: the retrieval metrics cost nothing and are real."""
    from rag_app.cli import check_eval_args

    assert check_eval_args(build_parser().parse_args(["eval", "--snapshot", "x"])) is None
