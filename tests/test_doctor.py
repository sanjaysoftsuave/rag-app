"""`doctor`: free, offline, never prints the key, always names the next step."""

from __future__ import annotations

from conftest import make_config

from rag_app.doctor import FAIL, PASS, WARN, _key, _packages, _python, describe, next_step, run_checks


def test_an_old_python_fails_with_a_fix():
    c = _python((3, 9))
    assert c.status == FAIL and c.fix


def test_a_current_python_passes():
    assert _python((3, 14)).status == PASS


def test_a_missing_core_package_fails_and_names_it():
    checks = _packages(spec=lambda name: None if name == "torch" else object())
    core = next(c for c in checks if c.name == "packages")
    assert core.status == FAIL and "torch" in core.detail


def test_missing_streamlit_only_warns():
    """The CLI works without the UI; that is a WARN, never a FAIL."""
    checks = _packages(spec=lambda name: None if name == "streamlit" else object())
    ui = next(c for c in checks if c.name == "ui")
    assert ui.status == WARN


def test_the_key_is_checked_for_presence_and_never_printed(tmp_path):
    secret = "sk-or-v1-SECRET-DO-NOT-PRINT"
    cfg = make_config(tmp_path, llm_api_key=secret)
    assert _key(cfg).status == PASS
    out = describe(run_checks(cfg))
    assert secret not in out


def test_a_missing_key_warns_with_where_to_get_one(tmp_path):
    c = _key(make_config(tmp_path, llm_api_key=None))
    assert c.status == WARN
    assert ".env" in c.fix


def test_a_fresh_checkout_is_told_to_run_quickstart(tmp_path):
    """No sample index yet is the first thing a stranger hits."""
    checks = run_checks(make_config(tmp_path))
    by_name = {c.name: c for c in checks}
    assert by_name["sample index"].status == WARN
    assert by_name["your index"].status == WARN
    assert next_step(checks) == "python -m rag_app quickstart"
    assert "Next: python -m rag_app quickstart" in describe(checks)
