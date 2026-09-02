import pytest

from rag_app.cli import build_parser
from rag_app.filters import MetaFilter


def test_all_commands_parse():
    parser = build_parser()
    for cmd in ("ui", "ask", "chunks", "models"):
        argv = [cmd] + (["q"] if cmd == "ask" else [])
        assert parser.parse_args(argv).command == cmd


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
