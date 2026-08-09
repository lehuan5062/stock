"""Dividend (long-term hold) mode: 100% LLM-agent-driven.

Unlike momentum/rebound, dividend numbers are NOT delegated to the agent's
web search — ``data.dividends`` is a deterministic fetcher (KBS/VCI, same
vnai-bypass technique as OHLCV) that computes real yield/payout-history
columns. The agent's job is purely to VET the sustainability of those real
numbers (earnings coverage, governance, dilution risk, sector stability) and
predict an ``expected_hold_years`` + confidence. There is no N/P/target — a
dividend pick is a hold, not a swing trade, so pricing is buy-at-close only
(see ``pricing.add_dividend_price_suggestions``).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..data.dividends import dividend_cache_path, dividend_summary
from ..news.llm_plan_runner import (Column, ResultField, fmt_int, fmt_num1,
                                    fmt_pct2, fmt_ratio3, fmt_symbol, fmt_text,
                                    fmt_type, parse_num_cell,
                                    parse_results_table, render_universe_table)
from ..news.sources import global_urls, vn_urls
from ..pricing import add_dividend_price_suggestions
from ..selector import eligible_universe
from ..tracking import run_signature
from .common import (default_n_picks, emit_universe_meta, read_candidates_sidecar,
                     read_meta, resolve_on_date, write_picks_json)

MODE = "dividend"

# The dividend universe reference table — every column is DETERMINISTIC data
# from ``data.dividends``, not a price/technical read. This mode is about the
# dividend, not about timing an entry: no rsi/momentum columns belong here.
#
# `cash_ttm_vnd` + `payouts_per_year` are the base rates the agent extrapolates
# from when predicting the forward dividend; `stock_div_ratio_3y` / `rights_3y`
# are the dilution signal (a "dividend" paid in new shares, or a rights issue
# funding the payout) that the prompt used to ask the agent to go web-search.
_UNIVERSE_COLUMNS = (
    Column("symbol", "symbol", fmt_symbol),
    Column("organ_name", "company", fmt_text),
    Column("close", "close", fmt_int),
    Column("dividend_yield_ttm", "yield_ttm", fmt_pct2),
    Column("cash_paid_ttm_vnd", "cash_ttm_vnd", fmt_int),
    Column("payouts_per_year", "payouts_per_yr", fmt_num1),
    Column("years_paid_consecutive", "years_paid", fmt_text),
    Column("payout_trend", "payout_trend", fmt_text),
    Column("last_ex_date", "last_ex_date", fmt_text),
    Column("stock_div_ratio_recent", "stock_div_ratio", fmt_ratio3),
    Column("placement_ratio_recent", "placement_ratio", fmt_ratio3),
    Column("instrument_type", "type", fmt_type),
)

# Mirrors ``data.dividends.dividend_summary``'s empty return, used to
# short-circuit symbols with no cached dividend file at all.
_NO_DIVIDEND_SUMMARY = {
    "dividend_yield_ttm": float("nan"),
    "cash_paid_ttm_vnd": float("nan"),
    "payouts_per_year": float("nan"),
    "years_paid_consecutive": 0,
    "last_ex_date": None,
    "payout_trend": "unknown",
    "n_dividend_events": 0,
    "stock_div_ratio_recent": float("nan"),
    "rights_recent": 0,
    "placement_ratio_recent": float("nan"),
    "announce_lead_days": float("nan"),
}


def _enrich_with_dividend_data(universe: pd.DataFrame, on_date: dt.date) -> pd.DataFrame:
    """Merge in the deterministic dividend-history columns for every symbol in
    the eligible universe: ``dividend_yield_ttm``, ``years_paid_consecutive``,
    ``last_ex_date``, ``payout_trend``, ``n_dividend_events``.

    Reads the LOCAL parquet cache (``cache/dividends/<SYM>.parquet``) — no
    network. Symbols with no cached file at all are short-circuited to the empty
    summary instead of paying for a miss inside ``dividend_summary``, which is
    most of the universe on a cold dividend cache.
    """
    symbols = universe["symbol"].astype(str).str.upper()
    if "close" in universe.columns:
        closes = list(universe["close"])
    else:
        closes = [None] * len(symbols)

    rows = []
    for sym, close in zip(symbols, closes):
        if not dividend_cache_path(sym).exists():
            rows.append({"symbol": sym, **_NO_DIVIDEND_SUMMARY})
            continue
        rows.append({"symbol": sym,
                     **dividend_summary(sym, close_vnd_thousand=close,
                                        as_of=on_date)})
    return universe.merge(pd.DataFrame(rows), on="symbol", how="left")


def run(on: str | None = None, n_picks: int | None = None,
       symbols: list[str] | None = None, hose_only: bool = False,
       include_etfs: bool = True, exclude: list[str] | None = None
       ) -> tuple[pd.DataFrame, Path]:
    """Fetch the eligible universe + dividend history, emit the dividend plan
    markdown. Returns (universe_df, plan_path)."""
    requested_n = default_n_picks(n_picks)
    universe = eligible_universe(on=on, symbols=symbols)
    on_date = resolve_on_date(on)
    universe = _enrich_with_dividend_data(universe, on_date)

    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode=MODE, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)
    plan_path = _write_dividend_plan(universe, on=on_date, run_signature=sig,
                                     n_picks=requested_n)
    emit_universe_meta(plan_path, universe, method="llm_only",
                       n_picks=requested_n, hose_only=hose_only,
                       include_etfs=include_etfs, exclude=excl_list, sig=sig,
                       mode=MODE)
    return universe, plan_path


def _write_dividend_plan(universe: pd.DataFrame, on: dt.date, run_signature: str,
                         n_picks: int) -> Path:
    out_dir = reports_dir()
    path = out_dir / f"dividend_plan_{on.isoformat()}_{run_signature}.md"

    from ..news.company_info import enrich
    universe = enrich(universe)

    div_cfg = dict(getattr(load_config(), "strategy", {}) or {}).get("dividend", {}) or {}
    lookback_years = int(div_cfg.get("trend_lookback_years", 3))

    lines = [
        f"# Dividend pick plan — {on.isoformat()}",
        "",
        "## Method — PREDICT next year's dividend (no ML ranking)",
        "",
        "This is the **dividend** (long-term hold) strategy: a HOLD, not a",
        "swing trade — buy at close, no profit target, no stop-loss, no fixed",
        "exit day. The payout numbers below come from a DETERMINISTIC fetcher",
        "(real VCI corporate-events data, not your web search).",
        "",
        "Your job is to **predict the dividend**, not to look up the last one.",
        "Every column below is BACKWARD-looking — it tells you what was paid,",
        "which is an INPUT to your forecast, not the answer. The pick is ranked",
        "on the dividend you predict for the NEXT 12 months.",
        "",
        f"1. **Select** the best **{int(n_picks)}** name(s) from the universe",
        "   table below. Use `yield_ttm` / `cash_ttm_vnd` / `payouts_per_yr` /",
        "   `years_paid` / `payout_trend` as your starting point.",
        "2. **Research sustainability**: earnings coverage (can the company",
        "   afford this payout from FCF/earnings, not debt?), governance/audit",
        "   flags, sector stability, and any sign the payout is about to be cut.",
        "   For **dilution**, check the data first — both columns cover the last",
        f"   {lookback_years} year(s):",
        "   - `stock_div_ratio` — new shares handed to EXISTING holders for free",
        "     (stock dividend / bonus issue). A company 'paying' its dividend in",
        "     shares rather than cash shows up here, not in `cash_ttm_vnd`.",
        "   - `placement_ratio` — new shares issued to THIRD PARTIES or staff",
        "     (private placement, ESOP). Dilutive, but not a distribution to you.",
        "   Treat a large `stock_div_ratio` with a thin cash yield as a warning:",
        "   the 'dividend' is share issuance. Rights offerings (which take cash",
        "   OUT of you) can't be reliably distinguished from placements in older",
        "   cached data, so verify any large `placement_ratio` with a search.",
        "3. **Predict `fwd_dps_vnd`** — the total CASH dividend per share, in",
        "   absolute VND, you expect over the next 12 months. Ground it in:",
        "   - `payouts_per_yr` — the cadence (how many payments a year), and",
        "     `cash_ttm_vnd` — the recent per-share level.",
        "   - any dividend already ANNOUNCED but not yet ex (search for board",
        "     resolutions / AGM minutes — these are near-certain income).",
        "   - management guidance, the declared payout ratio, and the earnings",
        "     trajectory. If you expect a CUT, predict the lower number and say so.",
        "   **Do not extrapolate a one-off.** A fat `yield_ttm` driven by a",
        "   single special dividend is not a repeating yield — e.g. a name",
        "   showing ~11% trailing off one 5,000 VND special payout will not pay",
        "   that again next year. Predict the RECURRING dividend; call out the",
        "   special separately in your findings if you expect another.",
        "4. **Predict** `expected_hold_years` (how many years you'd expect to",
        "   hold this for the dividend thesis to play out) and a `confidence`",
        "   (`low` / `med` / `high`) per pick.",
        "5. For each chosen name, write a `### TICKER — Company` section",
        "   documenting the business, dimensions researched, and findings —",
        "   then fill the results table at the bottom.",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing, or you expect the dividend to be suspended outright, do NOT",
        "pick the name (or write `DROP` in its `fwd_dps_vnd` cell). A predicted",
        "CUT is still a pick — forecast the lower amount. A predicted ZERO is",
        "not; `DROP` it.",
        "",
        "## Global / macro context (read once)",
        "",
        "Scan for major global shocks (wars, sanctions/tariffs, sharp oil/gold/",
        "USD-VND moves) and note the VN-Index's broad trend — a dividend hold",
        "cares less about short-term index moves than momentum/rebound do, but",
        "a sector-wide shock (e.g. a rate move hitting bank dividends) still",
        "matters.",
        "",
    ]
    for name, url in global_urls().items():
        lines.append(f"- [{name}]({url})")
    lines += [
        "",
        "## Universe (UNRANKED — the full mechanically-gated set + real dividend data)",
        "",
        *render_universe_table(universe, _UNIVERSE_COLUMNS),
    ]

    lines += [
        "",
        "## Per-pick research sections",
        "",
        "For EACH name you choose, add a section in this exact format. Use",
        "WebFetch / WebSearch, cross-check at least 2 sources, tag each finding",
        "with `[dimension]`.",
        "",
        "Seed sources you can reuse per ticker (replace TICKER):",
        "",
    ]
    for name, url in vn_urls("TICKER").items():
        lines.append(f"- [{name}]({url})")

    lines += [
        "",
        "```",
        "### TICKER  —  Company name",
        "",
        "**Step 1 — Business**: one line on what the company does.",
        "- ",
        "",
        "**Step 2 — Research dimensions**: the 3-7 sustainability drivers you",
        "judged matter for THIS ticker's payout (earnings coverage, governance,",
        "dilution, sector cycle, ...).",
        "- ",
        "",
        "**Step 3 — Forward dividend**: how you got to `fwd_dps_vnd`. State the",
        "cadence you assumed, the per-payment amount, anything already announced,",
        "and whether you stripped out a one-off special.",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "## Results — fill this with your chosen picks",
        "",
        "`fwd_dps_vnd` = predicted CASH dividend per share over the next 12",
        "months, in absolute VND (e.g. `2500`, not `2.5%` and not `2.5`) — it",
        "must be > 0. `expected_hold_years` >= 0.5; `confidence` in {low, med,",
        "high}. Finalize computes `pred_forward_yield = fwd_dps_vnd / close` and",
        "ranks by `pred_forward_yield × confidence`, so your FORECAST decides the",
        "ranking, not the trailing yield. Write `DROP` in `fwd_dps_vnd` to",
        "exclude a row you listed (including any name you expect to pay nothing).",
        "",
        "| rank | symbol | fwd_dps_vnd | expected_hold_years | confidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i in range(int(n_picks)):
        lines.append(f"| {i + 1} |  |  |  |  |")
    lines += [
        "",
        "When done, run:",
        f"  `python -m stockpredict.cli finalize reports/{path.name}`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_CONF_MAP = {"low": 0.33, "med": 0.66, "medium": 0.66, "high": 1.0}

# Version stamp for the ranking formula. The ledger persists ``rank`` but NOT
# ``score``, so a formula change is otherwise invisible in
# ``predictions.parquet`` — rows from both formulas claim ``rank: 1`` with
# nothing to tell them apart, and ``analyze.mode_compare`` would average two
# different strategies under one mode label. This id is written into the meta
# sidecar, the picks JSON and the ledger so history stays segmentable.
#
# Lineage: v1 (unstamped) ranked on the TRAILING yield × confidence. v2
# ("dividend_v2_entry") scaled that by an entry-quality judgment — reverted as
# the wrong direction for this mode, and it never reached the ledger. v3 ranks
# on the agent's FORWARD dividend forecast, which is the point of the mode.
SCORE_FORMULA = "dividend_v3_forward_yield"


def _parse_confidence(cell: str) -> float:
    return _CONF_MAP.get((cell or "").strip().lower(), float("nan"))


_RESULT_FIELDS = (
    # fwd_dps_vnd is the gate/DROP field: it is the mode's core prediction, so a
    # row without a usable forecast is not a pick. Note this MOVED the DROP cell
    # (it used to be expected_hold_years) — a plan filled against the older
    # template therefore fails validation loudly rather than being silently
    # reinterpreted, which is the intent.
    ResultField("fwd_dps_vnd", "pred_fwd_dps_vnd", parse_num_cell,
                drop_sentinel=True),
    ResultField("expected_hold_years", "expected_hold_years", parse_num_cell),
    ResultField("confidence", "confidence", _parse_confidence),
)


def parse_dividend_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled dividend plan and return DataFrame[symbol, dropped,
    pred_fwd_dps_vnd, expected_hold_years, confidence, business, dimensions,
    key_news, dimensions_cited]."""
    return parse_results_table(path, _RESULT_FIELDS)


def finalize(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    plan_path = Path(plan_path)
    scored = parse_dividend_plan(plan_path)
    if scored.empty:
        # A plan filled against the pre-forecast template has only 4 cells per
        # row and parses to nothing here. Say so — otherwise this reads as
        # "you left the table blank", which sends the user looking in the
        # wrong place.
        raise RuntimeError(
            f"no picks parsed from {plan_path} — fill the Results table. If the "
            f"table has no `fwd_dps_vnd` column, it predates the forward-dividend "
            f"forecast: re-run `predict --mode dividend` and fill the new table "
            f"(rank | symbol | fwd_dps_vnd | expected_hold_years | confidence).")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[dividend] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    bad = scored[scored["pred_fwd_dps_vnd"].isna() | (scored["pred_fwd_dps_vnd"] <= 0)]
    if not bad.empty:
        print(f"[dividend] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"non-positive fwd_dps_vnd: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError(
            "no picks with a valid fwd_dps_vnd — this mode ranks on the "
            "PREDICTED next-12-month cash dividend per share (absolute VND). "
            "If you filled a plan emitted before this column existed, re-run "
            "`predict --mode dividend` to get the current Results table.")

    bad_hold = scored[scored["expected_hold_years"].isna()
                     | (scored["expected_hold_years"] <= 0)]
    if not bad_hold.empty:
        print(f"[dividend] WARNING: dropping {len(bad_hold)} pick(s) with a missing/"
              f"invalid expected_hold_years: {', '.join(bad_hold['symbol'].tolist())}")
    scored = scored.drop(bad_hold.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid expected_hold_years")

    universe = read_candidates_sidecar(plan_path)
    if universe is not None:
        ref_cols = [c for c in ["symbol", "close", "dividend_yield_ttm",
                                "cash_paid_ttm_vnd", "payouts_per_year",
                                "years_paid_consecutive", "payout_trend",
                                "last_ex_date", "stock_div_ratio_recent",
                                "rights_recent", "announce_lead_days",
                                "organ_name", "instrument_type"]
                   if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    merged = add_dividend_price_suggestions(merged)

    # The ranking objective is the agent's FORECAST, not the trailing yield:
    # pred_forward_yield = predicted next-12-month cash DPS / today's price.
    # `dividend_yield_ttm` is kept alongside purely as a reference column so
    # trailing vs predicted stay visible side by side.
    fwd_dps = merged["pred_fwd_dps_vnd"].astype(float)
    close_vnd = merged["close"].astype(float) * 1000.0
    merged["pred_forward_yield"] = (fwd_dps / close_vnd).round(6)

    conf = merged.get("confidence", pd.Series(float("nan"), index=merged.index)).astype(float)
    merged["score"] = (merged["pred_forward_yield"] * conf.fillna(0.5)).round(6)

    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1
    # No T+2/T+N target for a hold — the ledger's exit resolution doesn't
    # apply here; this flag is purely informational for downstream tools.
    merged["below_recovery_bar"] = False
    merged["score_formula"] = SCORE_FORMULA

    meta = read_meta(plan_path)
    out, sig, _ = write_picks_json(MODE, merged, plan_path, meta,
                                   extra={"weight": None,
                                          "score_formula": SCORE_FORMULA})
    return merged, out
