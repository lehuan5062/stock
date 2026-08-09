"""Verify per-dimension hit-rate tracking: tags get extracted from Step 4
of the plan markdown, ride through finalize/record into the ledger, and
get aggregated by recent_performance + rendered by feedback_block."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredict import tracking
from stockpredict.news import llm_plan_runner


# ---------------------------------------------------------------------------
# 1. _extract_dimension_tags — the tag-parsing primitive
# ---------------------------------------------------------------------------

def test_extract_tags_kebab_case():
    findings = [
        "[insider-action] Deputy GM registered to buy 1M shares",
        "[sector-flow] Oil & gas led the market on May 5",
        "[governance] HoSE warning since 22/9/2025",
    ]
    tags = llm_plan_runner._extract_dimension_tags(findings)
    assert tags == ["insider-action", "sector-flow", "governance"]


def test_extract_tags_dedupes_keeping_first_seen_order():
    """If Claude tags two findings with the same dimension, the ledger
    should still get one entry — and the tag list is order-stable."""
    findings = [
        "[macro-VN] VN-Index breakout",
        "[macro-VN] foreign flows reversing",
        "[insider-action] insider buy",
        "[macro-VN] another macro point",
    ]
    tags = llm_plan_runner._extract_dimension_tags(findings)
    assert tags == ["macro-VN".lower(), "insider-action"]


def test_extract_tags_skips_pure_numeric_and_sentinels():
    """Citation markers like [1] [2] aren't dimensions. The "Net: +1"
    summary bullet shouldn't pollute either."""
    findings = [
        "[1] some footnote",
        "[insider-action] real dimension",
        "**Net: +1** — pooled commentary",
    ]
    tags = llm_plan_runner._extract_dimension_tags(findings)
    assert tags == ["insider-action"]


def test_extract_tags_empty_input():
    assert llm_plan_runner._extract_dimension_tags([]) == []
    assert llm_plan_runner._extract_dimension_tags(["", "no tags here"]) == []


def test_extract_tags_handles_multiple_per_bullet():
    """A bullet that cites two dimensions still feeds them both into
    the ledger — rare but documented behavior."""
    findings = [
        "[dim-a] [dim-b] cross-cutting finding tagged twice",
    ]
    tags = llm_plan_runner._extract_dimension_tags(findings)
    assert set(tags) == {"dim-a", "dim-b"}


# ---------------------------------------------------------------------------
# 2. parse_llm_plan emits a dimensions_cited column
# ---------------------------------------------------------------------------

def test_parse_plan_populates_dimensions_cited(tmp_path):
    plan = tmp_path / "p.md"
    plan.write_text(
        "# Test plan\n\n"
        "## Universe (UNRANKED — the full mechanically-gated set)\n\n"
        "### AAA  —  Test Co\n\n"
        "**Step 1 — Business**: \n- Plastic widget maker.\n\n"
        "**Step 2 — Research dimensions**:\n- one\n- two\n\n"
        "**Step 4 — Findings**:\n"
        "- [insider-action] CEO bought 500k shares\n"
        "- [sector-flow] Plastics sector tailwind\n\n"
        "## Results — fill this with your chosen picks\n\n"
        "| rank | symbol | N_days | P |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | AAA | 3 | 0.05 |\n",
        encoding="utf-8",
    )
    df = llm_plan_runner.parse_llm_plan(plan)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["dimensions_cited"] == "insider-action,sector-flow"


def test_parse_plan_dimensions_cited_empty_when_no_tags(tmp_path):
    """A plan whose Step 4 has bullets but no [tags] yields an empty
    dimensions_cited string — not NaN — so the parquet column stays
    string-typed."""
    plan = tmp_path / "p.md"
    plan.write_text(
        "# Test plan\n\n"
        "## Universe (UNRANKED — the full mechanically-gated set)\n\n"
        "### AAA  —  Test Co\n\n"
        "**Step 4 — Findings**:\n- something happened, no tag\n\n"
        "## Results — fill this with your chosen picks\n\n"
        "| rank | symbol | N_days | P |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | AAA | 3 | 0.05 |\n",
        encoding="utf-8",
    )
    df = llm_plan_runner.parse_llm_plan(plan)
    assert df.iloc[0]["dimensions_cited"] == ""


# ---------------------------------------------------------------------------
# 3. _normalize_dimensions and ledger storage
# ---------------------------------------------------------------------------

def test_normalize_accepts_string_and_list():
    assert tracking._normalize_dimensions("a,b,c") == "a,b,c"
    assert tracking._normalize_dimensions(["a", "b"]) == "a,b"
    assert tracking._normalize_dimensions(("a", "b")) == "a,b"


def test_normalize_strips_whitespace_and_lowercases():
    assert tracking._normalize_dimensions(" InsiderAction , Sector-Flow ") \
        == "insideraction,sector-flow"


def test_normalize_handles_none_nan_empty():
    assert tracking._normalize_dimensions(None) == ""
    assert tracking._normalize_dimensions(float("nan")) == ""
    assert tracking._normalize_dimensions("") == ""
    assert tracking._normalize_dimensions("   ") == ""


def test_record_stores_dimensions_cited(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")
    picks = pd.DataFrame([{
        "symbol": "AAA", "pred_mean": 0.05, "rank": 1,
        "close": 10.0, "actionable": True, "news_score": 1, "adjusted": 0.0525,
        "dimensions_cited": "insider-action,sector-flow",
    }])
    tracking.record(picks, mode="claude", as_of=pd.Timestamp("2026-05-05"))
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert df.iloc[0]["dimensions_cited"] == "insider-action,sector-flow"


# ---------------------------------------------------------------------------
# 4. _read backfill
# ---------------------------------------------------------------------------

def test_read_backfills_missing_dimensions_cited(monkeypatch, tmp_path):
    """An old ledger written before this release loads cleanly with
    dimensions_cited = '' for every legacy row."""
    monkeypatch.setattr(tracking, "ledger_path",
                        lambda: tmp_path / "predictions.parquet")
    # Exclude every later-added column so the legacy frame represents a
    # ledger written BEFORE the dimensions_cited release (and any later
    # release like low-prediction). _read should backfill them all.
    # Derived from the module's own backfill tuples rather than hardcoded, so
    # adding a ledger column can't silently break this test again.
    later_added = (set(tracking._NEW_STRING_COLUMNS)
                   | set(tracking._NEW_BOOL_COLUMNS)
                   | set(tracking._NEW_FLOAT_COLUMNS)
                   | {"t0_evaluated", "actual_exit_date"})
    legacy_cols = [c for c in tracking._LEDGER_COLUMNS if c not in later_added]
    today = pd.Timestamp.today().normalize()
    legacy = pd.DataFrame([{
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5, "realized_return": 0.05,
        "evaluated": True, "t0_open": np.nan, "t0_low": np.nan,
        "t0_close": np.nan, "entry_slippage": np.nan,
    }])[legacy_cols]
    legacy.to_parquet(tmp_path / "predictions.parquet", index=False)

    df = tracking._read()
    assert "dimensions_cited" in df.columns
    assert df.iloc[0]["dimensions_cited"] == ""


# ---------------------------------------------------------------------------
# 5. _by_dimension aggregator
# ---------------------------------------------------------------------------

def _row(dimensions_cited: str, realized_return: float, **kw):
    today = pd.Timestamp.today().normalize()
    base = {
        "run_id": "20260420_claude_d2_u100",
        "signature": "claude_d2_u100",
        "as_of": today - pd.Timedelta(days=10),
        "target_date": today - pd.Timedelta(days=8),
        "exit_offset_days": 2, "mode": "claude", "symbol": "AAA", "rank": 1,
        "pred_mean": 0.01, "news_score": 0, "adjusted": 0.01,
        "entry_price": 10.0, "actual_exit": 10.5,
        "realized_return": realized_return, "evaluated": True,
        "t0_open": np.nan, "t0_low": np.nan, "t0_close": np.nan,
        "entry_slippage": np.nan, "dimensions_cited": dimensions_cited,
    }
    base.update(kw)
    return base


def test_by_dimension_groups_correctly(monkeypatch):
    """Each cited tag contributes the row's realized_return to that bucket;
    a row with two tags contributes to both."""
    df = pd.DataFrame([
        _row("insider-action,sector-flow", 0.05, symbol="AAA"),
        _row("insider-action", 0.03, symbol="BBB"),
        _row("sector-flow,governance", -0.02, symbol="CCC"),
        _row("governance", -0.04, symbol="DDD"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode="claude")
    by_dim = perf["by_dimension"]
    # insider-action: rows AAA(+5%), BBB(+3%) → mean 0.04, both positive → 100% hit
    assert by_dim["insider-action"]["n"] == 2
    assert abs(by_dim["insider-action"]["mean_return"] - 0.04) < 1e-9
    assert by_dim["insider-action"]["hit_rate"] == 1.0
    # sector-flow: AAA(+5%), CCC(-2%) → mean 0.015, 1 of 2 positive → 50%
    assert by_dim["sector-flow"]["n"] == 2
    assert abs(by_dim["sector-flow"]["mean_return"] - 0.015) < 1e-9
    assert by_dim["sector-flow"]["hit_rate"] == 0.5
    # governance: CCC(-2%), DDD(-4%) → mean -0.03, 0 of 2 → 0%
    assert by_dim["governance"]["n"] == 2
    assert by_dim["governance"]["hit_rate"] == 0.0


def test_by_dimension_skips_rows_with_no_tags(monkeypatch):
    """Base-mode picks have empty dimensions_cited; they shouldn't pollute
    the aggregation — only Claude rows with actual tags contribute."""
    df = pd.DataFrame([
        _row("insider-action", 0.05, mode="claude", symbol="AAA"),
        _row("", 0.01, mode="base", symbol="BBB"),
        _row("", -0.02, mode="claude", symbol="CCC"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    perf = tracking.recent_performance(window_days=90, mode=None)  # all modes
    by_dim = perf["by_dimension"]
    assert "insider-action" in by_dim
    assert by_dim["insider-action"]["n"] == 1
    assert len(by_dim) == 1  # untagged rows didn't create any bucket


def test_by_dimension_empty_when_no_tagged_rows(monkeypatch):
    """All rows untagged → empty by_dimension dict (gracefully handled)."""
    df = pd.DataFrame([
        _row("", 0.05),
        _row("", -0.02),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)
    perf = tracking.recent_performance(window_days=90, mode="claude")
    assert perf["by_dimension"] == {}


# ---------------------------------------------------------------------------
# 6. feedback_block rendering
# ---------------------------------------------------------------------------

def test_feedback_block_renders_dimension_table(monkeypatch):
    """When ≥1 tag has n ≥ 2 evaluated rows, the dimension table appears."""
    df = pd.DataFrame([
        _row("insider-action", 0.05, symbol="AAA"),
        _row("insider-action", 0.03, symbol="BBB"),
        _row("sector-flow", 0.02, symbol="CCC"),
        _row("sector-flow", -0.01, symbol="DDD"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)

    block = tracking.feedback_block(window_days=90, mode="claude")
    assert "By dimension category cited" in block
    assert "`insider-action`" in block
    assert "`sector-flow`" in block
    # insider-action has higher mean → should sort above sector-flow
    insider_idx = block.index("`insider-action`")
    sector_idx = block.index("`sector-flow`")
    assert insider_idx < sector_idx


def test_feedback_block_omits_dimension_table_when_singleton_tags(monkeypatch):
    """Single-observation buckets are too noisy to surface; the table
    should be skipped if no tag has n ≥ 2."""
    df = pd.DataFrame([
        _row("only-once", 0.05),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)
    block = tracking.feedback_block(window_days=90, mode="claude")
    assert "By dimension category cited" not in block


def test_feedback_block_omits_dimension_table_when_no_data(monkeypatch):
    """Legacy ledger with all empty dimensions_cited → no dimension section."""
    df = pd.DataFrame([
        _row("", 0.05, symbol="AAA"),
        _row("", -0.02, symbol="BBB"),
    ], columns=tracking._LEDGER_COLUMNS)
    monkeypatch.setattr(tracking, "_read", lambda: df)
    block = tracking.feedback_block(window_days=90, mode="claude")
    assert "By dimension category cited" not in block
