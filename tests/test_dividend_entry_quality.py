"""Dividend mode's entry-quality column.

The dividend score used to be ``dividend_yield_ttm × confidence``; it is now
additionally scaled by an agent-supplied ``entry_quality`` judgment, because a
sustainable payer bought at a stretched price is a worse hold than the same
payer bought well (buy-at-close, no stop — a bad entry can't be undone).

The load-bearing property tested here is BACKWARD COMPATIBILITY: dividend plans
emitted before the column existed have a 4-cell results row, and must still
finalize to exactly the score and rank they produced before. The ledger keeps
``rank`` but not ``score``, so a silent change here would be unrecoverable.
"""
import json

import pandas as pd

from stockpredict.modes import dividend as dividend_mode

_CANDIDATES = [
    # AAA: fat yield. BBB: half the yield, same confidence.
    {"symbol": "AAA", "close": 20.0, "dividend_yield_ttm": 0.08,
     "years_paid_consecutive": 7, "payout_trend": "rising",
     "last_ex_date": "2026-03-15", "rsi_14": 75.0, "mom_20": 0.15,
     "high_prox_20": -0.002, "adv_vnd_20": 2e9},
    {"symbol": "BBB", "close": 30.0, "dividend_yield_ttm": 0.04,
     "years_paid_consecutive": 5, "payout_trend": "flat",
     "last_ex_date": "2026-01-10", "rsi_14": 42.0, "mom_20": -0.03,
     "high_prox_20": -0.12, "adv_vnd_20": 3e9},
]


def _setup(tmp_path, monkeypatch, results_rows: str, header: str) -> "pathlib.Path":
    plan = tmp_path / "dividend_plan_2026-08-04_dividend.md"
    plan.write_text(
        "# plan\n\n"
        "### AAA  —  Alpha\n**Step 4 — Findings**\n- [payout] covered\n\n"
        "### BBB  —  Beta\n**Step 4 — Findings**\n- [payout] covered\n\n"
        "## Results\n\n"
        f"{header}"
        f"{results_rows}",
        encoding="utf-8",
    )
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


_OLD_HEADER = ("| rank | symbol | expected_hold_years | confidence |\n"
               "| --- | --- | --- | --- |\n")
_NEW_HEADER = ("| rank | symbol | expected_hold_years | confidence | entry_quality |\n"
               "| --- | --- | --- | --- | --- |\n")


def test_legacy_four_column_plan_still_finalizes(tmp_path, monkeypatch):
    """A plan predating entry_quality must finalize, and score exactly
    yield × confidence (entry treated as neutral, NOT as "good")."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | high |\n"
                  "| 2 | BBB | 5 | high |\n",
                  _OLD_HEADER)

    merged, out = dividend_mode.finalize(plan)

    assert list(merged["symbol"]) == ["AAA", "BBB"]
    # 0.08 * 1.0 (high) — unchanged from the pre-entry_quality formula.
    assert abs(float(merged.iloc[0]["score"]) - 0.08) < 1e-9
    assert abs(float(merged.iloc[1]["score"]) - 0.04) < 1e-9
    assert merged.iloc[0]["score_income"] == merged.iloc[0]["score"]
    # Not assessed, and honestly reported as such rather than as "good".
    assert list(merged["entry_quality"]) == ["", ""]
    assert merged["entry_factor"].isna().all()


def test_poor_entry_demotes_a_higher_yield_pick(tmp_path, monkeypatch):
    """The whole point: a fat yield at a stretched price can rank BELOW a
    thinner yield at a good price."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | high | poor |\n"
                  "| 2 | BBB | 5 | high | good |\n",
                  _NEW_HEADER)

    merged, out = dividend_mode.finalize(plan)

    # AAA income 0.08 * 0.6 = 0.048; BBB 0.04 * 1.0 = 0.04 -> AAA still wins.
    assert abs(float(merged[merged["symbol"] == "AAA"].iloc[0]["score"]) - 0.048) < 1e-9
    assert abs(float(merged[merged["symbol"] == "BBB"].iloc[0]["score"]) - 0.04) < 1e-9
    assert list(merged["symbol"]) == ["AAA", "BBB"]
    # score_income is preserved unscaled so the two effects stay separable.
    assert abs(float(merged[merged["symbol"] == "AAA"].iloc[0]["score_income"]) - 0.08) < 1e-9
    assert list(merged["entry_quality"]) == ["poor", "good"]


def test_poor_entry_can_flip_the_ranking(tmp_path, monkeypatch):
    """With a wide enough entry gap the order actually changes — proving
    entry_quality is a real ranking input, not decoration."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | med | poor |\n"     # 0.08*0.66*0.6 = 0.03168
                  "| 2 | BBB | 5 | high | good |\n",   # 0.04*1.00*1.0 = 0.04
                  _NEW_HEADER)

    merged, _ = dividend_mode.finalize(plan)

    assert list(merged["symbol"]) == ["BBB", "AAA"]
    assert merged.iloc[0]["rank"] == 1


def test_score_formula_is_stamped(tmp_path, monkeypatch):
    """The version stamp must reach both the picks JSON and the ledger, so
    `rank` rows from the old and new formulas stay distinguishable."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | high | good |\n",
                  _NEW_HEADER)

    merged, out = dividend_mode.finalize(plan)

    assert (merged["score_formula"] == dividend_mode.SCORE_FORMULA).all()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["score_formula"] == dividend_mode.SCORE_FORMULA
    assert payload["mode"] == "dividend"

    from stockpredict import tracking
    ledger = pd.read_parquet(tracking.ledger_path())
    assert (ledger["score_formula"] == dividend_mode.SCORE_FORMULA).all()


def test_unrecognised_entry_quality_is_neutral_not_fatal(tmp_path, monkeypatch):
    """A garbled cell must never discard a pick — entry_quality is not a drop
    condition."""
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | 5 | high | excellent?? |\n"
                  "| 2 | BBB | 5 | high |  |\n",
                  _NEW_HEADER)

    merged, _ = dividend_mode.finalize(plan)

    assert list(merged["symbol"]) == ["AAA", "BBB"]
    assert abs(float(merged.iloc[0]["score"]) - 0.08) < 1e-9
    assert list(merged["entry_quality"]) == ["", ""]


def test_drop_still_works_with_the_extra_column(tmp_path, monkeypatch):
    plan = _setup(tmp_path, monkeypatch,
                  "| 1 | AAA | DROP | high | good |\n"
                  "| 2 | BBB | 5 | high | good |\n",
                  _NEW_HEADER)

    merged, _ = dividend_mode.finalize(plan)

    assert list(merged["symbol"]) == ["BBB"]


def test_universe_table_shows_entry_columns(tmp_path, monkeypatch):
    """The agent can only judge entry quality if the price columns are actually
    rendered — they were absent before this change."""
    monkeypatch.setattr(dividend_mode, "reports_dir", lambda: tmp_path)
    import datetime as dt

    universe = pd.DataFrame(_CANDIDATES)
    path = dividend_mode._write_dividend_plan(
        universe, on=dt.date(2026, 8, 4), run_signature="sig", n_picks=2)
    text = path.read_text(encoding="utf-8")

    header = next(ln for ln in text.splitlines()
                  if ln.startswith("| symbol |"))
    for col in ("rsi_14", "mom_20", "high_prox_20", "adv_vnd_20"):
        assert col in header, f"{col} missing from the dividend universe table"
    assert "entry_quality" in text
    # The results template must offer the 5th cell to fill.
    assert "| rank | symbol | expected_hold_years | confidence | entry_quality |" in text
