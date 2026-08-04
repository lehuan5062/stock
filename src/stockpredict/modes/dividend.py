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

from ..config import reports_dir
from ..data.dividends import dividend_cache_path, dividend_summary
from ..news.llm_plan_runner import (Column, ResultField, parse_num_cell,
                                    fmt_comma, fmt_int, fmt_pct2, fmt_signed3,
                                    fmt_symbol, fmt_text, fmt_type,
                                    parse_results_table, render_universe_table)
from ..news.sources import global_urls, vn_urls
from ..pricing import add_dividend_price_suggestions
from ..selector import eligible_universe
from ..tracking import run_signature
from .common import (default_n_picks, emit_universe_meta, read_candidates_sidecar,
                     read_meta, resolve_on_date, write_picks_json)

MODE = "dividend"

# The dividend universe reference table. The income columns come from the
# deterministic fetcher; the price/technical columns ride along on
# ``eligible_universe`` and are shown so the agent can judge ENTRY quality —
# a great payer bought at a stretched price is a poor hold. ``mom_5`` is
# deliberately omitted: a 5-day move is noise over a multi-year holding period.
_UNIVERSE_COLUMNS = (
    Column("symbol", "symbol", fmt_symbol),
    Column("organ_name", "company", fmt_text),
    Column("close", "close", fmt_int),
    Column("dividend_yield_ttm", "dividend_yield_ttm", fmt_pct2),
    Column("years_paid_consecutive", "years_paid_consecutive", fmt_text),
    Column("payout_trend", "payout_trend", fmt_text),
    Column("last_ex_date", "last_ex_date", fmt_text),
    Column("rsi_14", "rsi_14", fmt_int),
    Column("mom_20", "mom_20", fmt_signed3),
    Column("high_prox_20", "high_prox_20", fmt_signed3),
    Column("adv_vnd_20", "adv_vnd_20", fmt_comma),
    Column("instrument_type", "type", fmt_type),
)


_NO_DIVIDEND_SUMMARY = {
    "dividend_yield_ttm": float("nan"),
    "years_paid_consecutive": 0,
    "last_ex_date": None,
    "payout_trend": "unknown",
    "n_dividend_events": 0,
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

    lines = [
        f"# Dividend pick plan — {on.isoformat()}",
        "",
        "## Method — vet payout sustainability (no ML ranking)",
        "",
        "This is the **dividend** (long-term hold) strategy: a HOLD, not a",
        "swing trade — buy at close, no profit target, no stop-loss, no fixed",
        "exit day. The yield/payout numbers below come from a DETERMINISTIC",
        "fetcher (real VCI corporate-events data, not your web search) — your",
        "job is to VET whether the payout is sustainable, not to find the",
        "numbers yourself.",
        "",
        f"1. **Select** the best **{int(n_picks)}** name(s) from the universe",
        "   table below, using the real dividend_yield_ttm / "
        "years_paid_consecutive / payout_trend columns as your starting point.",
        "2. **Research sustainability**: earnings coverage (can the company",
        "   afford this payout from FCF/earnings, not debt?), governance/audit",
        "   flags, dilution risk (is it really a stock dividend disguised as a",
        "   cash one, or issuing new shares to fund the payout?), sector",
        "   stability, and any signs the payout is about to be cut.",
        "3. **Judge the entry price.** A sustainable payer bought at a",
        "   stretched price is still a poor hold: the yield you actually lock",
        "   in is the yield AT YOUR ENTRY, and this is a buy-at-close hold with",
        "   no stop — you cannot un-pay a bad entry. Use `rsi_14` / `mom_20` /",
        "   `high_prox_20` plus your own research to rate the entry `good` /",
        "   `fair` / `poor`, and watch for three distinct traps:",
        "   - **Stretched**: extended after a run-up, `rsi_14` overbought, or",
        "     pressed against the 20-day high (`high_prox_20` near 0).",
        "   - **Yield trap**: a fat `dividend_yield_ttm` that is only fat",
        "     because the price is collapsing on a real problem. Cross-check",
        "     `payout_trend` and your sustainability findings — a high yield on",
        "     a `declining` payout and a falling price is a warning, not a buy.",
        "   - **Ex-date position**: `last_ex_date` tells you where you are in",
        "     the payout cycle. If the ex-date just passed you have to wait a",
        "     full cycle for income, and the quoted TTM yield overstates what",
        "     you'll receive near-term.",
        "4. **Predict** `expected_hold_years` (how many years you'd expect to",
        "   hold this for the dividend thesis to play out) and a `confidence`",
        "   (`low` / `med` / `high`) per pick.",
        "5. For each chosen name, write a `### TICKER — Company` section",
        "   documenting the business, dimensions researched, and findings —",
        "   then fill the results table at the bottom.",
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing / an imminent dividend CUT, do NOT pick the name (or write",
        "`DROP` in its results row).",
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
        "**Step 3 — Entry quality**: is TODAY a good price to start this hold?",
        "State good / fair / poor and why (stretched? yield trap? where in the",
        "payout cycle?).",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "## Results — fill this with your chosen picks",
        "",
        "`expected_hold_years` >= 0.5; `confidence` in {low, med, high};",
        "`entry_quality` in {good, fair, poor} (blank = not assessed, scored as",
        "neutral). Finalize ranks by `dividend_yield_ttm × confidence ×",
        "entry_factor`, where entry_factor is good 1.0 / fair 0.85 / poor 0.6 —",
        "so a stretched entry demotes a pick rather than dropping it. Write",
        "`DROP` in `expected_hold_years` to exclude a row you listed.",
        "",
        "| rank | symbol | expected_hold_years | confidence | entry_quality |",
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

# Entry-quality → multiplier applied to the income score. A BLANK/unrecognised
# cell parses to NaN, which is treated as the NEUTRAL factor 1.0 when scoring —
# so a dividend plan emitted before this column existed still finalizes to
# exactly the score and rank it would have produced before. NaN is kept
# distinct from an explicit "good" (also 1.0) so the picks JSON can record
# "not assessed" honestly rather than claiming the agent vetted the entry.
_ENTRY_FACTOR_MAP = {"good": 1.0, "fair": 0.85, "poor": 0.6}
_NEUTRAL_ENTRY_FACTOR = 1.0
_ENTRY_LABEL_MAP = {v: k for k, v in _ENTRY_FACTOR_MAP.items()}

# Version stamp for the ranking formula. ``score`` used to be
# ``dividend_yield_ttm × confidence``; it is now additionally scaled by the
# agent's entry-quality judgment. Because the ledger persists ``rank`` but NOT
# ``score``, a formula change would otherwise be invisible in
# ``predictions.parquet`` — rows from both formulas would claim ``rank: 1``
# with nothing to tell them apart. This id is written into the meta sidecar,
# the picks JSON and the ledger so ``analyze.mode_compare`` can segment
# history instead of blending two different strategies under one label.
SCORE_FORMULA = "dividend_v2_entry"


def _parse_confidence(cell: str) -> float:
    return _CONF_MAP.get((cell or "").strip().lower(), float("nan"))


def _parse_entry_factor(cell: str) -> float:
    """Map the entry_quality cell to its multiplier. Unrecognised/blank -> NaN
    ("not assessed"), scored as neutral later. This field is never a
    ``drop_sentinel``, so a NaN here can't gate the row out."""
    return _ENTRY_FACTOR_MAP.get((cell or "").strip().lower(), float("nan"))


_RESULT_FIELDS = (
    ResultField("expected_hold_years", "expected_hold_years",
                parse_num_cell, drop_sentinel=True),
    ResultField("confidence", "confidence", _parse_confidence),
    # required=False: plans emitted before entry-quality existed have only 4
    # cells and must keep finalizing.
    ResultField("entry_quality", "entry_factor", _parse_entry_factor,
                required=False),
)


def parse_dividend_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled dividend plan and return DataFrame[symbol, dropped,
    expected_hold_years, confidence, entry_factor, business, dimensions,
    key_news, dimensions_cited]."""
    return parse_results_table(path, _RESULT_FIELDS)


def finalize(plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    plan_path = Path(plan_path)
    scored = parse_dividend_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no picks parsed from {plan_path} — fill the Results table")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[dividend] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    bad = scored[scored["expected_hold_years"].isna() | (scored["expected_hold_years"] <= 0)]
    if not bad.empty:
        print(f"[dividend] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"invalid expected_hold_years: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid expected_hold_years")

    universe = read_candidates_sidecar(plan_path)
    if universe is not None:
        ref_cols = [c for c in ["symbol", "close", "dividend_yield_ttm",
                                "years_paid_consecutive", "payout_trend",
                                "last_ex_date", "organ_name", "instrument_type",
                                # Recorded so the picks JSON preserves the price
                                # state the entry judgment was actually made on.
                                "rsi_14", "mom_20", "high_prox_20", "adv_vnd_20"]
                   if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    merged = add_dividend_price_suggestions(merged)

    # Income quality: yield weighted by the agent's sustainability confidence.
    yld = merged.get("dividend_yield_ttm", pd.Series(float("nan"), index=merged.index)).astype(float)
    conf = merged.get("confidence", pd.Series(float("nan"), index=merged.index)).astype(float)
    merged["score_income"] = (yld.fillna(0.0) * conf.fillna(0.5)).round(6)

    # Entry quality scales it: a sustainable payer bought at a stretched price
    # is a worse hold than the same payer bought well, because this is a
    # buy-at-close hold with no stop — you cannot un-pay the entry. An
    # unassessed entry (NaN) scores neutral, so pre-existing plans rank exactly
    # as they did before.
    entry_factor = merged.get(
        "entry_factor", pd.Series(float("nan"), index=merged.index)).astype(float)
    merged["entry_factor"] = entry_factor
    merged["entry_quality"] = entry_factor.map(
        lambda v: _ENTRY_LABEL_MAP.get(v, "") if pd.notna(v) else "")
    merged["score"] = (merged["score_income"]
                       * entry_factor.fillna(_NEUTRAL_ENTRY_FACTOR)).round(6)

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
