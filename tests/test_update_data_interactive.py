"""Tests for `update-data --interactive`.

The prompting used to live in ``update_data.bat``, where an empty answer made
cmd parse ``for %%s in () do ...`` inside an ``if (...)`` block — a parse-time
failure that aborted the script and closed the window with no visible error.
Prompting now happens in Python, so both the empty-input and scoped-input
paths are testable here.
"""
from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from stockpredict.cli import cli


def _patch_fetch(monkeypatch):
    """Stub out everything that would touch the network / disk, and capture the
    symbol list `update_many` is called with."""
    captured: dict = {}

    def fake_update_many(syms, full=False):
        captured["symbols"] = list(syms)
        captured["full"] = full
        return {s: 1 for s in syms}

    monkeypatch.setattr("stockpredict.data.fetcher.update_many", fake_update_many)
    monkeypatch.setattr("stockpredict.data.intro.introduce", lambda *a, **k: None)
    monkeypatch.setattr("stockpredict.data.universe.load_universe",
                        lambda *a, **k: pd.DataFrame({"symbol": ["AAA", "BBB"],
                                                      "exchange": ["HSX", "HSX"]}))
    monkeypatch.setattr("stockpredict.data.universe.filter_exchanges",
                        lambda u, exchanges: u)
    return captured


def test_interactive_scoped_symbols(monkeypatch):
    """A comma-separated answer fetches exactly those symbols, upper-cased."""
    captured = _patch_fetch(monkeypatch)
    result = CliRunner().invoke(cli, ["update-data", "--interactive"],
                                input="acb,hpg\nn\n")
    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ["ACB", "HPG"]
    assert captured["full"] is False


def test_interactive_empty_input_means_full_universe(monkeypatch):
    """Plain Enter at the symbols prompt falls through to the whole universe.

    This is the exact input that used to kill the batch script's window.
    """
    captured = _patch_fetch(monkeypatch)
    result = CliRunner().invoke(cli, ["update-data", "--interactive"],
                                input="\nn\n")
    assert result.exit_code == 0, result.output
    # Universe stub (AAA/BBB) plus the CURATED top-up — the point is that it
    # did NOT scope to an empty list and did not crash.
    assert "AAA" in captured["symbols"]
    assert len(captured["symbols"]) > 1


def test_interactive_full_refetch_confirm(monkeypatch):
    """Answering yes to the second prompt passes full=True through."""
    captured = _patch_fetch(monkeypatch)
    result = CliRunner().invoke(cli, ["update-data", "--interactive"],
                                input="acb\ny\n")
    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ["ACB"]
    assert captured["full"] is True


def test_non_interactive_flags_still_bind(monkeypatch):
    """The flag path is unregressed: no prompting, flags used as given."""
    captured = _patch_fetch(monkeypatch)
    result = CliRunner().invoke(cli, ["update-data", "-s", "acb", "--full"])
    assert result.exit_code == 0, result.output
    assert captured["symbols"] == ["ACB"]
    assert captured["full"] is True
