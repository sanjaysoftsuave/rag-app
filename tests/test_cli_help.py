import pytest
from conftest import make_config

from rag_app.cli import build_parser
from rag_app.evaluate import GoldQuestion
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


# ---------------------------------------------------------------------------
# Week 9: MCP discovery and this app's own server
# ---------------------------------------------------------------------------


def test_agent_mcp_flags_default_off():
    a = build_parser().parse_args(["agent", "q"])
    assert a.mcp_stdio is False
    assert a.mcp_command is None
    assert a.mcp_url is None
    assert a.mcp_allow is False


def test_agent_mcp_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["agent", "q", "--mcp-stdio", "--mcp-url", "http://x"])


def test_agent_mcp_command_takes_a_command_line():
    a = build_parser().parse_args(["agent", "q", "--mcp-command", "python other_server.py"])
    assert a.mcp_command == "python other_server.py"


def test_mcp_serve_defaults_to_stdio():
    a = build_parser().parse_args(["mcp-serve"])
    assert a.transport == "stdio"
    assert a.preset is None
    assert a.port == 8765


def test_mcp_serve_rejects_an_unknown_transport():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["mcp-serve", "--transport", "carrier-pigeon"])


# ---------------------------------------------------------------------------
# Week 10: the team arm, and the arena --generate bug it exposed
# ---------------------------------------------------------------------------


def test_arena_selects_the_real_callables_when_generating():
    """Pins the fix: arena --generate used to pass None, which compare_arms'
    own `or` fallback silently read as 'use the dry stand-in' — so --generate
    cost nothing and measured nothing, for both arms, with no error anywhere."""
    from rag_app.agent import _call as agent_call
    from rag_app.cli import _arena_callables
    from rag_app.generate import generate_answer

    assert _arena_callables(True) == (generate_answer, agent_call)


def test_arena_selects_safe_dry_standins_without_generate():
    from rag_app.cli import _arena_callables

    generate_fn, llm_fn = _arena_callables(False)
    assert generate_fn("q", None, 3) == "(dry run)"
    assert "dry run" in llm_fn([], None).lower()


def test_agent_impl_choices_include_team():
    a = build_parser().parse_args(["agent", "q", "--impl", "team"])
    assert a.impl == "team"


def test_atasks_impl_choices_include_team():
    a = build_parser().parse_args(["atasks", "--impl", "team"])
    assert a.impl == "team"


def test_arena_arms_help_mentions_team():
    help_text = build_parser().format_help()
    # Parsing a comma-joined --arms string is untyped by argparse (free text),
    # so there is no choices list to assert against; this just confirms a
    # value of "team" is accepted at all.
    a = build_parser().parse_args(["arena", "--arms", "agent,team"])
    assert a.arms == "agent,team"


def test_cfg_for_impl_overrides_the_snapshotted_implementation(tmp_path):
    """Pins the atasks --snapshot fix directly."""
    from rag_app.cli import _cfg_for_impl

    cfg = make_config(tmp_path)
    assert cfg.agent.implementation == "plain"
    assert _cfg_for_impl(cfg, "team").agent.implementation == "team"
    assert _cfg_for_impl(cfg, None) is cfg  # no override named -> unchanged


def test_atasks_snapshots_show_the_implementation_that_actually_ran(tmp_path):
    """End to end through make_snapshot/compare: the exact scenario the
    mentor checklist asks for - racing plain vs team on the same tasks and
    being able to tell, from the snapshots alone, which arm is which."""
    from rag_app.before_after import compare, make_snapshot
    from rag_app.cli import _cfg_for_impl

    cfg = make_config(tmp_path)
    gold = [GoldQuestion(question="q", must_contain=["x"])]

    before = make_snapshot("single-v1", _cfg_for_impl(cfg, "plain"), {"task_success": 1.0}, gold)
    after = make_snapshot("team-v1", _cfg_for_impl(cfg, "team"), {"task_success": 1.0}, gold)

    assert before.settings["agent_implementation"] == "plain"
    assert after.settings["agent_implementation"] == "team"
    result = compare(before, after)
    assert ("agent_implementation", "plain", "team") in result.changed_settings
    assert not any("no configuration difference" in w for w in result.warnings)


def test_attach_mcp_tools_is_a_noop_without_any_mcp_flag():
    from rag_app.cli import _attach_mcp_tools
    from rag_app.tools import ToolRegistry

    args = build_parser().parse_args(["agent", "q"])
    tools = ToolRegistry()
    result, conn = _attach_mcp_tools(args, None, tools)
    assert result is tools
    assert conn is None


def test_attach_mcp_tools_explains_the_missing_extra(monkeypatch):
    import rag_app.mcp_client as mcp_client
    from rag_app.cli import _attach_mcp_tools

    monkeypatch.setattr(mcp_client, "available", lambda: False)
    args = build_parser().parse_args(["agent", "q", "--mcp-stdio"])
    with pytest.raises(RuntimeError, match="pip install"):
        _attach_mcp_tools(args, None, None)
