"""Dividend mode: forward-dividend prediction + the cache-schema adapter.

Two things are covered here, and the first one is a real bug this replaced:

1. **The adapter.** ``read_dividend_history`` used to reject any cache file whose
   columns didn't match one exact set, returning an empty frame. 148 of 150 real
   cache files were in an older shape, so the mode saw dividend history for
   exactly TWO symbols — VCB, GAS, SAB and every other major payer read as
   "no dividend history". The normalizer must accept all known shapes.

2. **Ranking on the forecast.** The mode ranks on the agent's predicted
   next-12-month cash dividend, not on the trailing yield. A fat trailing yield
   with a predicted cut must lose to a modest trailing yield that holds up.
"""
import datetime as dt
import json

import pandas as pd
import pytest

from stockpredict.data import dividends as div_data
from stockpredict.modes import dividend as dividend_mode


# ---------------------------------------------------------------------------
# 1. The cache-schema adapter
# ---------------------------------------------------------------------------

def _write(tmp_path, monkeypatch, symbol, frame):
    monkeypatch.setattr(div_data, "dividends_cache_dir", lambda: tmp_path)
    frame.to_parquet(tmp_path / f"{symbol}.parquet", index=False)


def test_reads_canonical_shape(tmp_path, monkeypatch):
    frame = pd.DataFrame([{
        "kind": "cash", "ex_date": pd.Timestamp("2026-03-01"),
        "record_date": pd.Timestamp("2026-03-02"), "pay_date": pd.Timestamp("2026-03-20"),
        "announce_date": pd.Timestamp("2026-02-10"), "cash_per_share_vnd": 1000.0,
        "ratio": float("nan"), "subscription_vnd": float("nan"),
        "title": "Cash Dividend", "source": "VCI", "parsed_from": "DIV/x",
        "ex_date_estimated": False,
    }])
    _write(tmp_path, monkeypatch, "AAA", frame)
    out = div_data.read_dividend_history("AAA")
    assert list(out.columns) == div_data.CANONICAL_COLUMNS
    assert len(out) == 1 and out.iloc[0]["kind"] == "cash"


def test_reads_legacy_wide_shape(tmp_path, monkeypatch):
    """THE regression: the 2026-06-30 on-disk shape (148 of 150 real files).
    Before the adapter this returned empty and the symbol looked dividend-less."""
    frame = pd.DataFrame([
        {"symbol": "BBB", "cutoff_date": pd.Timestamp("2026-02-26"),
         "record_date": pd.Timestamp("2026-02-27"), "pay_date": pd.Timestamp("2026-04-03"),
         "kind": "cash", "cash_vnd": 1000.0, "ratio": float("nan"),
         "subscription_vnd": float("nan"), "source": "VCI",
         "parsed_from": "DIV/value_per_share[vnd]",
         "public_date": pd.Timestamp("2026-02-06"), "fetched_at": pd.Timestamp("2026-06-30")},
        {"symbol": "BBB", "cutoff_date": pd.Timestamp("2026-05-18"),
         "record_date": pd.Timestamp("2026-05-19"), "pay_date": pd.NaT,
         "kind": "stock", "cash_vnd": float("nan"), "ratio": 0.15,
         "subscription_vnd": float("nan"), "source": "VCI",
         "parsed_from": "ISS/stock/exercise_ratio",
         "public_date": pd.Timestamp("2026-05-08"), "fetched_at": pd.Timestamp("2026-06-30")},
    ])
    _write(tmp_path, monkeypatch, "BBB", frame)

    out = div_data.read_dividend_history("BBB")
    assert len(out) == 2, "legacy-wide rows must not be silently discarded"
    assert list(out.columns) == div_data.CANONICAL_COLUMNS
    assert out["kind"].tolist() == ["cash", "stock"]
    # cutoff_date -> ex_date, cash_vnd -> cash_per_share_vnd, public_date -> announce_date
    assert out.iloc[0]["ex_date"] == pd.Timestamp("2026-02-26")
    assert out.iloc[0]["cash_per_share_vnd"] == 1000.0
    assert out.iloc[0]["announce_date"] == pd.Timestamp("2026-02-06")


def test_reads_div_only_shape(tmp_path, monkeypatch):
    """The immediately-previous schema. Every row was a DIV event, so every row
    is a cash dividend."""
    frame = pd.DataFrame([{
        "ex_date": pd.Timestamp("2026-05-28"), "record_date": pd.Timestamp("2026-05-29"),
        "payout_date": pd.Timestamp("2026-06-10"), "cash_per_share_vnd": 1000.0,
        "exercise_ratio": 0.1, "title": "Cash Dividend - Interim 2 2025",
    }])
    _write(tmp_path, monkeypatch, "CCC", frame)
    out = div_data.read_dividend_history("CCC")
    assert len(out) == 1
    assert out.iloc[0]["kind"] == "cash"
    assert out.iloc[0]["pay_date"] == pd.Timestamp("2026-06-10")
    assert out.iloc[0]["ratio"] == 0.1


def test_unknown_shape_reads_empty_not_raises(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "DDD",
           pd.DataFrame([{"something": 1, "else": 2}]))
    out = div_data.read_dividend_history("DDD")
    assert out.empty
    assert list(out.columns) == div_data.CANONICAL_COLUMNS


def test_missing_ex_date_falls_back_to_announce_date(tmp_path, monkeypatch):
    """172 real rows have no ex-date and 164 of them are issuance events — the
    dilution signal. They must be anchored to the announcement date and
    flagged, not dropped."""
    frame = pd.DataFrame([{
        "symbol": "EEE", "cutoff_date": pd.NaT, "record_date": pd.NaT,
        "pay_date": pd.NaT, "kind": "stock", "cash_vnd": float("nan"),
        "ratio": 0.5, "subscription_vnd": float("nan"), "source": "VCI",
        "parsed_from": "ISS/stock/exercise_ratio",
        "public_date": pd.Timestamp("2026-06-09"), "fetched_at": pd.NaT,
    }])
    _write(tmp_path, monkeypatch, "EEE", frame)
    out = div_data.read_dividend_history("EEE")
    assert len(out) == 1, "an issuance with only an announce date must survive"
    assert out.iloc[0]["ex_date"] == pd.Timestamp("2026-06-09")
    assert bool(out.iloc[0]["ex_date_estimated"]) is True


def test_legacy_rights_label_is_downgraded_to_placement(tmp_path, monkeypatch):
    """The old writer's `rights` label was its GUESS bucket — every such row
    carries `ISS/rights/par-fallback` and it swept ESOP grants in too. We must
    not restate it as a rights offering we can't evidence."""
    frame = pd.DataFrame([{
        "symbol": "GGG", "cutoff_date": pd.Timestamp("2026-06-09"),
        "record_date": pd.NaT, "pay_date": pd.NaT, "kind": "rights",
        "cash_vnd": float("nan"), "ratio": 0.0008, "subscription_vnd": 10000.0,
        "source": "VCI", "parsed_from": "ISS/rights/par-fallback",
        "public_date": pd.Timestamp("2026-06-01"), "fetched_at": pd.NaT,
    }])
    _write(tmp_path, monkeypatch, "GGG", frame)
    out = div_data.read_dividend_history("GGG")
    assert out.iloc[0]["kind"] == "placement"


def test_row_with_no_usable_date_is_dropped(tmp_path, monkeypatch):
    frame = pd.DataFrame([{
        "symbol": "FFF", "cutoff_date": pd.NaT, "kind": "stock",
        "cash_vnd": float("nan"), "ratio": 0.1, "public_date": pd.NaT,
        "record_date": pd.NaT, "pay_date": pd.NaT, "subscription_vnd": float("nan"),
        "source": "VCI", "parsed_from": "x", "fetched_at": pd.NaT,
    }])
    _write(tmp_path, monkeypatch, "FFF", frame)
    assert div_data.read_dividend_history("FFF").empty


# ---------------------------------------------------------------------------
# 2. dividend_summary — forecast inputs + dilution, cash-only income metrics
# ---------------------------------------------------------------------------

def _mixed_history():
    """Two cash dividends a year for 2025-2026, plus a stock dividend and a
    rights issue."""
    return pd.DataFrame([
        {"kind": "cash", "ex_date": pd.Timestamp("2025-03-01"),
         "cash_per_share_vnd": 500.0, "ratio": float("nan"),
         "announce_date": pd.Timestamp("2025-02-01"), "ex_date_estimated": False},
        {"kind": "cash", "ex_date": pd.Timestamp("2025-09-01"),
         "cash_per_share_vnd": 500.0, "ratio": float("nan"),
         "announce_date": pd.Timestamp("2025-08-02"), "ex_date_estimated": False},
        {"kind": "cash", "ex_date": pd.Timestamp("2026-03-01"),
         "cash_per_share_vnd": 800.0, "ratio": float("nan"),
         "announce_date": pd.Timestamp("2026-02-01"), "ex_date_estimated": False},
        {"kind": "cash", "ex_date": pd.Timestamp("2026-06-01"),
         "cash_per_share_vnd": 700.0, "ratio": float("nan"),
         "announce_date": pd.Timestamp("2026-05-02"), "ex_date_estimated": False},
        {"kind": "stock", "ex_date": pd.Timestamp("2026-04-01"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.20,
         "announce_date": pd.Timestamp("2026-03-01"), "ex_date_estimated": False},
        {"kind": "rights", "ex_date": pd.Timestamp("2026-05-01"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.10,
         "announce_date": pd.Timestamp("2026-04-01"), "ex_date_estimated": False},
        {"kind": "placement", "ex_date": pd.Timestamp("2026-05-15"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.30,
         "announce_date": pd.Timestamp("2026-04-15"), "ex_date_estimated": False},
    ])


def test_summary_income_metrics_are_cash_only(monkeypatch):
    """Stock dividends and rights issues must NOT count as "a dividend was
    paid" — that would inflate years_paid_consecutive and corrupt payout_trend."""
    monkeypatch.setattr(div_data, "read_dividend_history", lambda s: _mixed_history())
    s = div_data.dividend_summary("XXX", close_vnd_thousand=30.0,
                                  as_of=dt.date(2026, 8, 4), years=3)

    # TTM (2025-08-04..2026-08-04) cash = 500 + 800 + 700 = 2000
    assert s["cash_paid_ttm_vnd"] == 2000.0
    # yield is rounded to 4dp on the way out
    assert s["dividend_yield_ttm"] == pytest.approx(2000.0 / 30000.0, abs=5e-5)
    # n_dividend_events counts CASH events only (4), not all 6 rows.
    assert s["n_dividend_events"] == 4
    # last_ex_date is the last CASH ex-date (2026-06-01), not the stock/rights one.
    assert s["last_ex_date"] == "2026-06-01"
    # 2025 total 1000 -> 2026 total 1500 = rising
    assert s["payout_trend"] == "rising"


def test_summary_exposes_forecast_inputs(monkeypatch):
    monkeypatch.setattr(div_data, "read_dividend_history", lambda s: _mixed_history())
    s = div_data.dividend_summary("XXX", close_vnd_thousand=30.0,
                                  as_of=dt.date(2026, 8, 4), years=3)
    # 4 cash events inside a 3-year window -> 1.33 per year
    assert s["payouts_per_year"] == pytest.approx(4 / 3, abs=0.01)
    # every cash event announced ~28-30 days ahead
    assert 25 <= s["announce_lead_days"] <= 35


def test_summary_exposes_dilution(monkeypatch):
    monkeypatch.setattr(div_data, "read_dividend_history", lambda s: _mixed_history())
    s = div_data.dividend_summary("XXX", close_vnd_thousand=30.0,
                                  as_of=dt.date(2026, 8, 4), years=3)
    assert s["stock_div_ratio_recent"] == pytest.approx(0.20)
    assert s["rights_recent"] == 1
    # A placement is NOT summed into the stock-dividend ratio: being paid in
    # shares and the company selling shares to someone else are different facts.
    assert s["placement_ratio_recent"] == pytest.approx(0.30)


def test_placement_is_not_counted_as_a_stock_dividend(monkeypatch):
    """The bug this guards: VCB's legacy 0.6879 "stock dividend" ratio actually
    included a 6.5% private placement."""
    hist = pd.DataFrame([
        {"kind": "stock", "ex_date": pd.Timestamp("2026-04-01"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.1279,
         "announce_date": pd.Timestamp("2026-03-01"), "ex_date_estimated": False},
        {"kind": "placement", "ex_date": pd.Timestamp("2026-04-01"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.0650,
         "announce_date": pd.Timestamp("2026-03-01"), "ex_date_estimated": False},
    ])
    monkeypatch.setattr(div_data, "read_dividend_history", lambda s: hist)
    s = div_data.dividend_summary("VCBLIKE", close_vnd_thousand=56.5,
                                  as_of=dt.date(2026, 8, 4), years=3)
    assert s["stock_div_ratio_recent"] == pytest.approx(0.1279)
    assert s["placement_ratio_recent"] == pytest.approx(0.0650)


def test_summary_handles_stock_only_payer(monkeypatch):
    """A company that only ever issued stock has NO cash yield, but its dilution
    must still be reported — this is the VCB/BID/VIC pattern."""
    hist = pd.DataFrame([
        {"kind": "stock", "ex_date": pd.Timestamp("2026-04-01"),
         "cash_per_share_vnd": float("nan"), "ratio": 0.5,
         "announce_date": pd.Timestamp("2026-03-01"), "ex_date_estimated": False},
    ])
    monkeypatch.setattr(div_data, "read_dividend_history", lambda s: hist)
    s = div_data.dividend_summary("YYY", close_vnd_thousand=30.0,
                                  as_of=dt.date(2026, 8, 4), years=3)
    assert pd.isna(s["dividend_yield_ttm"])
    assert s["n_dividend_events"] == 0
    assert s["years_paid_consecutive"] == 0
    assert s["last_ex_date"] is None
    assert s["stock_div_ratio_recent"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. The mode ranks on the forecast
# ---------------------------------------------------------------------------

_CANDIDATES = [
    # AAA: fat trailing yield (3000/11000 = 27%), cheap price.
    {"symbol": "AAA", "close": 11.0, "dividend_yield_ttm": 0.2727,
     "cash_paid_ttm_vnd": 3000.0, "payouts_per_year": 1.7,
     "years_paid_consecutive": 3, "payout_trend": "declining",
     "last_ex_date": "2026-06-18", "stock_div_ratio_recent": 0.0,
     "placement_ratio_recent": 0.0},
    # BBB: modest trailing yield (3000/50000 = 6%).
    {"symbol": "BBB", "close": 50.0, "dividend_yield_ttm": 0.0595,
     "cash_paid_ttm_vnd": 3000.0, "payouts_per_year": 1.0,
     "years_paid_consecutive": 10, "payout_trend": "flat",
     "last_ex_date": "2026-03-17", "stock_div_ratio_recent": 0.0,
     "placement_ratio_recent": 0.0},
]

_HEADER = ("| rank | symbol | fwd_dps_vnd | expected_hold_years | confidence |\n"
           "| --- | --- | --- | --- | --- |\n")


def _setup(tmp_path, monkeypatch, rows, header=_HEADER):
    plan = tmp_path / "dividend_plan_2026-08-04_dividend.md"
    plan.write_text(
        "# plan\n\n"
        "### AAA  —  Alpha\n**Step 4 — Findings**\n- [payout] covered\n\n"
        "### BBB  —  Beta\n**Step 4 — Findings**\n- [payout] covered\n\n"
        "## Results\n\n" + header + rows,
        encoding="utf-8")
    pd.DataFrame(_CANDIDATES).to_parquet(
        plan.with_suffix(".candidates.parquet"), index=False)
    plan.with_suffix(".meta.json").write_text(json.dumps({
        "method": "llm_only", "n_picks": 2, "hose_only": False,
        "include_etfs": True, "exclude": [], "run_signature": "dividend",
        "mode": "dividend",
    }), encoding="utf-8")
    import stockpredict.modes.common as common_mod
    monkeypatch.setattr(common_mod, "reports_dir", lambda: tmp_path)
    return plan


def test_forecast_beats_trailing_yield(tmp_path, monkeypatch):
    """The headline behaviour: AAA has a 27% TRAILING yield but a predicted cut,
    BBB has a 6% trailing yield that holds. BBB must rank first."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 500 | 3 | high |\n"     # 500/11000  = 4.5%
                  "| 2 | BBB | 3000 | 5 | high |\n")   # 3000/50000 = 6.0%

    merged, _ = dividend_mode.finalize(plan)

    assert list(merged["symbol"]) == ["BBB", "AAA"]
    assert merged.iloc[0]["rank"] == 1
    assert merged.iloc[0]["pred_forward_yield"] == pytest.approx(0.06, abs=1e-4)
    assert merged.iloc[1]["pred_forward_yield"] == pytest.approx(0.045455, abs=1e-4)
    # Trailing yield is preserved as a reference column, unchanged.
    aaa = merged[merged.symbol == "AAA"].iloc[0]
    assert aaa["dividend_yield_ttm"] == pytest.approx(0.2727)


def test_score_is_forward_yield_times_confidence(tmp_path, monkeypatch):
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 1100 | 3 | med |\n"     # 0.10 * 0.66 = 0.066
                  "| 2 | BBB | 2500 | 5 | high |\n")   # 0.05 * 1.00 = 0.05
    merged, _ = dividend_mode.finalize(plan)
    by = {r["symbol"]: r for _, r in merged.iterrows()}
    assert by["AAA"]["score"] == pytest.approx(0.066, abs=1e-6)
    assert by["BBB"]["score"] == pytest.approx(0.05, abs=1e-6)
    assert list(merged["symbol"]) == ["AAA", "BBB"]


@pytest.mark.parametrize("cell", ["0", "-100", ""])
def test_non_positive_forecast_is_dropped(tmp_path, monkeypatch, cell):
    """A predicted zero/negative dividend is not a pick. A blank cell can't be
    ranked either — the agent must DROP instead."""
    plan = _setup(tmp_path, monkeypatch,
                  f"| 1 | AAA | {cell} | 3 | high |\n"
                  "| 2 | BBB | 3000 | 5 | high |\n")
    merged, _ = dividend_mode.finalize(plan)
    assert list(merged["symbol"]) == ["BBB"]


def test_drop_sentinel_on_forecast_cell(tmp_path, monkeypatch):
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | DROP | 3 | high |\n"
                  "| 2 | BBB | 3000 | 5 | high |\n")
    merged, _ = dividend_mode.finalize(plan)
    assert list(merged["symbol"]) == ["BBB"]


def test_plan_predating_forward_column_fails_loudly(tmp_path, monkeypatch):
    """A plan filled against the OLD 4-column template must not be silently
    reinterpreted under the new formula — it must raise."""
    old_header = ("| rank | symbol | expected_hold_years | confidence |\n"
                  "| --- | --- | --- | --- |\n")
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | high |\n"
                  "| 2 | BBB | 5 | high |\n",
                  header=old_header)
    with pytest.raises(RuntimeError, match="fwd_dps_vnd"):
        dividend_mode.finalize(plan)


def test_forecast_reaches_picks_json_and_ledger(tmp_path, monkeypatch):
    plan = _setup(tmp_path, monkeypatch, "| 1 | BBB | 3000 | 5 | high |\n")
    merged, out = dividend_mode.finalize(plan)

    assert (merged["score_formula"] == dividend_mode.SCORE_FORMULA).all()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["score_formula"] == dividend_mode.SCORE_FORMULA
    assert payload["picks"][0]["pred_fwd_dps_vnd"] == 3000.0

    from stockpredict import tracking
    ledger = pd.read_parquet(tracking.ledger_path())
    assert ledger.iloc[0]["pred_fwd_dps_vnd"] == 3000.0
    assert ledger.iloc[0]["score_formula"] == dividend_mode.SCORE_FORMULA


def test_plan_table_is_income_only_no_price_technicals(tmp_path, monkeypatch):
    """This mode is about the dividend, not about timing an entry: no rsi /
    momentum columns belong in its universe table."""
    monkeypatch.setattr(dividend_mode, "reports_dir", lambda: tmp_path)
    universe = pd.DataFrame(_CANDIDATES)
    universe["rsi_14"] = 70.0
    universe["mom_20"] = 0.2
    path = dividend_mode._write_dividend_plan(
        universe, on=dt.date(2026, 8, 4), run_signature="sig", n_picks=2)
    text = path.read_text(encoding="utf-8")

    header = next(ln for ln in text.splitlines() if ln.startswith("| symbol |"))
    for col in ("rsi_14", "mom_20", "high_prox_20"):
        assert col not in header, f"{col} must not be in the dividend table"
    for col in ("yield_ttm", "cash_ttm_vnd", "payouts_per_yr",
                "stock_div_ratio", "placement_ratio"):
        assert col in header, f"{col} missing from the dividend table"
    assert "fwd_dps_vnd" in text
    assert "| rank | symbol | fwd_dps_vnd | expected_hold_years | confidence |" in text


# ---------------------------------------------------------------------------
# 4. ISS classification against the real VCI payload shape
#
# Captured from the live endpoint (VCB / REE, 2026-08-10). The payload carries
# NO subscription/issue-price field for ISS events, so the event title is the
# only discriminator — an earlier version of this code looked for a price field
# and therefore never classified anything correctly.
# ---------------------------------------------------------------------------

def _iss(title_en, ratio):
    return {"event_code": "ISS", "event_title_en": title_en,
            "event_title_vi": "", "exright_date": "2026-04-28",
            "public_date": "2026-04-01", "record_date": None,
            "issue_date": None, "exercise_ratio": ratio,
            "value_per_share": None}


@pytest.mark.parametrize("title,ratio,expected_kind", [
    ("Share Issue - Stock dividend ratio 49.5%", 0.495, "stock"),
    ("Share Issue - Bonus Issue ratio 12.8%", 0.1279, "stock"),
    ("Share Issue - Private Placements ratio 6.5%", 0.065, "placement"),
    ("Share Issue - ESOP ratio 0.1%", 0.0008, "placement"),
    ("Share Issue - Rights Issue ratio 50.0%", 0.5, "rights"),
    ("Share Issue - Something Novel", 0.1, "placement"),
])
def test_iss_subtypes_from_real_titles(title, ratio, expected_kind):
    out = div_data._parse_events(pd.DataFrame([_iss(title, ratio)]))
    assert len(out) == 1
    assert out.iloc[0]["kind"] == expected_kind
    assert out.iloc[0]["ratio"] == pytest.approx(ratio)


def test_div_and_iss_both_parsed():
    """An earlier version kept only DIV, discarding every stock dividend and
    rights issue — i.e. the whole dilution picture."""
    raw = pd.DataFrame([
        {"event_code": "DIV", "event_title_en": "Cash Dividend - 450 VND",
         "event_title_vi": "", "exright_date": "2026-07-23",
         "public_date": "2026-07-01", "record_date": "2026-07-24",
         "payout_date": "2026-08-10", "exercise_ratio": 0.045,
         "value_per_share": 450.0, "issue_date": None},
        _iss("Share Issue - Stock dividend ratio 49.5%", 0.495),
        # Non-payout event types must be ignored entirely.
        {"event_code": "AGME", "event_title_en": "AGM", "event_title_vi": "",
         "exright_date": "2026-03-01", "public_date": "2026-02-01",
         "record_date": None, "issue_date": None, "exercise_ratio": None,
         "value_per_share": None},
    ])
    out = div_data._parse_events(raw)
    assert out["kind"].tolist() == ["stock", "cash"]  # sorted by ex_date
    cash = out[out.kind == "cash"].iloc[0]
    assert cash["cash_per_share_vnd"] == 450.0
    assert cash["pay_date"] == pd.Timestamp("2026-08-10")
