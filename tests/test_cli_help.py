import pytest

from rag_app.cli import build_parser
from rag_app.filters import MetaFilter


def test_all_commands_parse():
    parser = build_parser()
    needs_question = {"ask", "compare"}
    for cmd in ("ingest", "ask", "compare", "eval", "chunks", "models", "chat", "debug"):
        argv = [cmd] + (["q"] if cmd in needs_question else [])
        assert parser.parse_args(argv).command == cmd


def test_eval_compare_retrieval_flags():
    args = build_parser().parse_args(["eval", "--compare-retrieval", "--k", "5"])
    assert args.compare_retrieval is True
    assert args.k == 5
    assert build_parser().parse_args(["eval"]).compare_retrieval is False


def test_debug_flags():
    args = build_parser().parse_args(["debug", "--generate", "--show-pass", "--preset", "A"])
    assert args.generate is True
    assert args.show_pass is True
    assert args.preset == "A"
    defaults = build_parser().parse_args(["debug"])
    assert defaults.generate is False
    assert defaults.show_pass is False


def test_ask_accepts_repeated_filters():
    args = build_parser().parse_args(
        ["ask", "why 429?", "--filter", "product=API", "--filter", "customer_tier=free"]
    )
    flt = MetaFilter.parse(args.filter)
    assert flt.must == {"product": "API", "customer_tier": "free"}
    assert flt.describe() == "customer_tier=free AND product=API"


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask"])


def test_sweep_flags_exist():
    parser = build_parser()
    assert parser.parse_args(["ingest", "--all"]).all is True
    assert parser.parse_args(["eval", "--all"]).all is True
    assert parser.parse_args(["eval", "--sweep"]).sweep is True
    assert parser.parse_args(["chunks", "--show", "3"]).show == 3
