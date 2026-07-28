"""Tests for cache.log_corp_action_event, the durable record of
SuspectedCorporateActionArtifact guard trips (see test_cache_merge_guard.py
for the guard itself)."""
from __future__ import annotations

import csv

import pytest

from stockpredict.data import cache


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path)
    yield tmp_path


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_log_creates_header_on_first_write(isolated_log):
    cache.log_corp_action_event("ABB", "ABB 2026-07-08: 15.0% move ...", healed=True)

    path = cache._corp_action_log_path()
    assert path.exists()
    rows = _read_rows(path)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "ABB"
    assert rows[0]["healed"] == "True"


def test_log_appends_without_overwriting(isolated_log):
    cache.log_corp_action_event("ABB", "first violation", healed=True)
    cache.log_corp_action_event("XYZ", "second violation", healed=False)

    rows = _read_rows(cache._corp_action_log_path())
    assert len(rows) == 2
    assert [r["symbol"] for r in rows] == ["ABB", "XYZ"]
    assert rows[1]["healed"] == "False"


def test_log_records_unhealed_events(isolated_log):
    cache.log_corp_action_event("ABB", "still broken after full re-fetch", healed=False)

    rows = _read_rows(cache._corp_action_log_path())
    assert rows[0]["healed"] == "False"
