"""Command-line entry point for the predictor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from .config import load_config

# Force UTF-8 on stdout/stderr so Vietnamese text in plans/picks doesn't crash
# the default Windows cp1252 console codec.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

MODES = ("momentum", "rebound", "dividend")


@click.group()
def cli() -> None:
    """Vietnamese stock predictor — 100% LLM-agent-driven, 3 modes (momentum /
    rebound / dividend). No ML/DL model anywhere in the live path."""


def _mode_module(mode: str):
    from .modes import dividend, momentum, rebound
    return {"momentum": momentum, "rebound": rebound, "dividend": dividend}[mode]


def _format_picks(picks) -> str:
    """One-line-per-pick view with entry / target / fees / net (VND) sized
    for the configured position. Falls back to full dataframe if pricing
    columns aren't present."""
    if picks is None or len(picks) == 0:
        return "(no picks)"
    show_cols = [c for c in [
        "symbol", "close_vnd", "target_vnd",
        "fees_round_trip_vnd", "fees_buy_vnd", "net_reward_vnd",
        "score", "pred_days", "pred_profit",
        "below_recovery_bar", "dividend_yield_ttm", "pred_fwd_dps_vnd",
        "pred_forward_yield", "years_paid_consecutive",
        "payout_trend", "expected_hold_years", "confidence",
    ] if c in picks.columns]
    if not show_cols:
        return picks.to_string(index=False)
    fmt = picks[show_cols].copy()
    money_cols = ["close_vnd", "target_vnd", "fees_round_trip_vnd",
                 "fees_buy_vnd", "net_reward_vnd"]
    for c in money_cols:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(
                lambda v: f"{int(v):>+9,}" if pd.notna(v) and c in ("net_reward_vnd",)
                          else (f"{int(v):>9,}" if pd.notna(v) else "        -")
            )
    return fmt.to_string(index=False)


def _format_picks_explained(picks) -> str:
    """Verbose paragraph-per-pick view once the agent has produced business +
    key_news + dimensions per ticker."""
    if picks is None or len(picks) == 0:
        return "(no picks)"
    parts: list[str] = []
    for i, r in picks.reset_index(drop=True).iterrows():
        sym = r["symbol"]
        organ = r.get("organ_name", "") or ""
        business = r.get("business", "") or ""
        header = f"=== #{i+1} {sym}"
        if organ:
            header += f"  —  {organ}"
        elif business:
            header += f"  —  {business[:60]}"
        parts.append(header)

        is_hold = "expected_hold_years" in r and pd.notna(r.get("expected_hold_years"))
        if is_hold:
            entry = int(r["close_vnd"]) if pd.notna(r.get("close_vnd")) else None
            yld = r.get("dividend_yield_ttm")
            years = r.get("years_paid_consecutive")
            trend = r.get("payout_trend")
            hold = r.get("expected_hold_years")
            if entry is not None:
                parts.append(f"  Trade: buy @ {entry:,} VND  |  HOLD (no target/no stop)  "
                             f"|  expected hold ≈ {hold:.1f}y")
            yld_s = f"{yld:.2%}" if pd.notna(yld) else "n/a"
            parts.append(f"  Signal: yield_ttm={yld_s}  years_paid={years}  "
                         f"trend={trend}  score={r.get('score', float('nan')):.4f}")
            # The forecast is the ranking objective, so show it explicitly
            # next to the trailing number it must not be confused with.
            fwd_dps = r.get("pred_fwd_dps_vnd")
            fwd_yld = r.get("pred_forward_yield")
            if pd.notna(fwd_dps):
                fwd_yld_s = f"{fwd_yld:.2%}" if pd.notna(fwd_yld) else "n/a"
                parts.append(f"  Forecast: next-12m dividend ≈ {float(fwd_dps):,.0f} "
                             f"VND/share  ->  forward yield {fwd_yld_s} "
                             f"(vs trailing {yld_s})")
        elif "score" in r and pd.notna(r.get("score")):
            if "close_vnd" in r and pd.notna(r.get("close_vnd")):
                entry = int(r["close_vnd"]); tgt = int(r["target_vnd"])
                fees = int(r.get("fees_round_trip_vnd", 0))
                net = int(r.get("net_reward_vnd", 0))
                below = bool(r.get("below_recovery_bar", False))
                verdict = "BELOW BAR (weak)" if below else "OK"
                hold = r.get("pred_days")
                hold_s = f"{int(round(hold))}d" if pd.notna(hold) else "?"
                parts.append(f"  Trade: buy @ {entry:,} VND  |  target {tgt:,}  |  hold ≈ {hold_s} (sell at target)")
                parts.append(f"  P&L per share (after ACBS fees {fees:,}): net {net:+,}  -> {verdict}")
            parts.append(f"  Signal: score={r.get('score'):.4f}  "
                        f"N≈{r.get('pred_days', float('nan')):.0f}d  "
                        f"P≈{r.get('pred_profit', float('nan')):+.3f}")

        if business:
            parts.append(f"  Business: {business}")
        dims = r.get("dimensions", None)
        if isinstance(dims, str) and dims.strip():
            parts.append(f"  Research dimensions: {dims}")
        key_news = r.get("key_news", None)
        if isinstance(key_news, (list, tuple)) and len(key_news) > 0:
            parts.append("  News found:")
            for k in key_news:
                parts.append(f"    - {k}")
        parts.append("")
    return "\n".join(parts)


def _has_explanations(picks) -> bool:
    if picks is None or len(picks) == 0:
        return False
    return any(c in picks.columns for c in ("business", "key_news"))


def _print_pick_warnings(picks, requested_n: int) -> None:
    n = int(len(picks)) if picks is not None else 0
    if requested_n and n < int(requested_n):
        click.echo("")
        click.echo(f"==> SHORTFALL: only {n} of {requested_n} requested pick(s) "
                   f"available — the eligible universe is smaller than N "
                   f"(heavy --exclude / --hose-only / tiny cache).")
    bar_col = "below_recovery_bar"
    if picks is None or n == 0 or bar_col not in picks.columns:
        return
    k = int(picks[bar_col].fillna(True).astype(bool).sum())
    if k > 0:
        click.echo("")
        click.echo(f"==> QUALITY: {k} of {n} pick(s) are below the break-even bar "
                   f"(weak edge — forecast under round-trip cost). Shown to honor "
                   f"--picks; treat them with extra caution.")


def _print_sell_reminder(picks, *, mode: str) -> None:
    if picks is None or len(picks) == 0:
        return
    if mode == "dividend":
        click.echo("")
        click.echo("==> HOLD (no fixed sell day, no target) — this is a long hold, "
                   "not a swing trade. Do not schedule a sell reminder.")
        return
    click.echo("")
    click.echo("==> EXIT PLAN (flexible — sell when the price reaches the target):")
    for _, r in picks.reset_index(drop=True).iterrows():
        tgt = r.get("target_vnd")
        hold = r.get("pred_days")
        tgt_s = f"{int(tgt):,}" if pd.notna(tgt) else "?"
        hold_s = f", expected ≈ {int(round(hold))}d" if pd.notna(hold) else ""
        click.echo(f"    {r['symbol']}: target {tgt_s} VND{hold_s}")
    click.echo("    There is NO fixed sell day — do not schedule a hard sell "
               "alarm. Offer an optional check-in only if asked.")


# ---------------------------- data -----------------------------------------


@cli.command("update-data")
@click.option("--symbols", "-s", multiple=True, help="Specific tickers; default = full universe")
@click.option("--full", is_flag=True, help="Re-fetch full history instead of incremental")
@click.option("--limit", type=int, default=None, help="Cap symbol count (debug)")
@click.option("--interactive", "-i", is_flag=True,
              help="Prompt for symbol scope and full-refetch instead of reading flags.")
def update_data(symbols: tuple[str, ...], full: bool, limit: int | None,
                interactive: bool) -> None:
    """Refresh the OHLCV parquet cache from vnstock."""
    from .data.fetcher import update_many
    from .data.intro import introduce
    from .data.universe import filter_exchanges, load_universe

    if interactive:
        # Prompting lives here rather than in update_data.bat: batch's
        # `set /p` + delayed expansion + parenthesized blocks kill the whole
        # script (and its window) on a parse error, so an empty answer used to
        # close the terminal with no visible cause.
        raw = click.prompt("Symbols to fetch, comma-separated (empty = full universe)",
                           default="", show_default=False)
        symbols = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
        full = click.confirm("Force full history re-fetch instead of incremental?",
                             default=False)

    introduce()

    if symbols:
        syms = [s.upper() for s in symbols]
    else:
        cfg = load_config()
        u = load_universe(refresh=True)
        u = filter_exchanges(u, cfg.data["exchanges"])
        syms = u["symbol"].tolist()
        from .selector import CURATED
        seen = {s.upper() for s in syms}
        for s in CURATED:
            if s.upper() not in seen:
                syms.append(s.upper())
                seen.add(s.upper())
    if limit:
        syms = syms[:limit]
    click.echo(f"Updating {len(syms)} symbols (full={full})...")
    results = update_many(syms, full=full)
    ok = sum(1 for v in results.values() if isinstance(v, int))
    err = len(results) - ok
    click.echo(f"done. ok={ok} err={err}")
    if err:
        bad = [(k, v) for k, v in results.items() if not isinstance(v, int)][:10]
        for k, v in bad:
            click.echo(f"  {k}: {v}")


@cli.command("update-dividends")
@click.option("--symbols", "-s", multiple=True, help="Specific tickers; default = cached universe")
def update_dividends_cmd(symbols: tuple[str, ...]) -> None:
    """Refresh the dividend-history parquet cache (separable from OHLCV — a
    dividend-only refresh doesn't require a full data re-fetch)."""
    from .data.cache import cached_symbols
    from .data.dividends import update_dividends

    syms = [s.upper() for s in symbols] if symbols else cached_symbols()
    click.echo(f"Updating dividend history for {len(syms)} symbols...")
    results = update_dividends(syms)
    ok = sum(1 for v in results.values() if isinstance(v, int))
    err = len(results) - ok
    click.echo(f"done. ok={ok} err={err}")
    if err:
        bad = [(k, v) for k, v in results.items() if not isinstance(v, int)][:10]
        for k, v in bad:
            click.echo(f"  {k}: {v}")


@cli.command("compare-modes")
@click.option("--window", type=int, default=90, show_default=True,
              help="Look-back window in days to pool over.")
@click.option("--date", "as_of", default=None,
              help="YYYY-MM-DD: also show this single day's per-mode breakdown "
                   "as context (still pools the verdict over the window).")
@click.option("--modes", default=None,
              help="Comma-separated subset to compare (e.g. momentum,rebound). "
                   "Default: every mode found in the ledger.")
def compare_modes_cmd(window: int, as_of: str | None, modes: str | None) -> None:
    """Head-to-head realized performance of the 3 strategies, pooled over
    comparable runs — same day AND same params (picks/hose-only/etfs/exclude).
    Advisory: tells you which strategy has been winning, not a knob to tune."""
    from .analyze import mode_compare
    from .config import reports_dir
    mode_list = ([m.strip() for m in modes.split(",") if m.strip()]
                if modes else None)
    result = mode_compare.compare_modes(window_days=window, as_of=as_of,
                                       modes=mode_list)
    md = mode_compare.format_report(result)
    click.echo(md)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = reports_dir() / f"mode_comparison_{today}.md"
    out.write_text(md, encoding="utf-8")
    click.echo(f"\nreport -> {out}")


@cli.command("compare-picks")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD: compare picks reports for this day")
def compare_picks_cmd(as_of: str) -> None:
    """Compare the actual picks selected by different modes on the same day."""
    from .analyze import mode_compare
    result = mode_compare.compare_picks_same_day(as_of)
    click.echo(mode_compare.format_picks_comparison(result))


@cli.command("compare-picks-accountability")
@click.option("--date", "as_of", required=True, help="YYYY-MM-DD: mode accountability for this day")
def compare_picks_accountability_cmd(as_of: str) -> None:
    """Show mode accountability: for each resolved pick, which modes selected
    it and which avoided it."""
    from .analyze import mode_compare
    result = mode_compare.mode_accountability(as_of)
    click.echo(mode_compare.format_mode_accountability(result))


# ---------------------------- plan / finalize -------------------------------


@cli.command("predict")
@click.option("--mode", type=click.Choice(MODES), required=True)
@click.option("--picks", "-n", "n_picks", type=int, default=None,
              help="How many picks the agent should surface. Defaults to "
                   "pricing.default_picks in config.yaml.")
@click.option("--date", "on", default=None, help="YYYY-MM-DD; defaults to most recent cache date")
def predict_cmd(mode: str, n_picks: int | None, on: str | None) -> None:
    """Emit the plan markdown for one mode (alias for `run` without the data
    refresh — useful once the cache is already warm)."""
    if n_picks is not None and n_picks < 1:
        click.echo("ERROR: --picks must be >= 1.", err=True)
        sys.exit(2)
    mod = _mode_module(mode)
    universe, plan_path = mod.run(on=on, n_picks=n_picks)
    click.echo(f"eligible universe: {len(universe)} name(s) for the agent to pick from.")
    click.echo(f"\nsaved -> {plan_path}")
    click.echo(f"Next: research + fill the plan, then run "
              f"`finalize \"{plan_path}\"`.")


@cli.command("finalize")
@click.argument("plan_path", type=click.Path(exists=True))
def finalize_cmd(plan_path: str) -> None:
    """Finalize a filled plan markdown: auto-detects the mode from the
    ``.meta.json`` sidecar (falls back to the filename prefix), ranks,
    prices, writes ``picks_<mode>_<date>_<sig>.json``, and updates the
    ledger."""
    p = Path(plan_path)
    meta_path = p.with_suffix(".meta.json")
    mode = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mode = meta.get("mode")
        except Exception:
            mode = None
    if mode not in MODES:
        for m in MODES:
            if p.name.startswith(f"{m}_plan_"):
                mode = m
                break
    if mode not in MODES:
        click.echo(f"ERROR: could not detect mode for {plan_path} "
                   f"(expected a `.meta.json` sidecar or a "
                   f"`<mode>_plan_*` filename).", err=True)
        sys.exit(2)

    mod = _mode_module(mode)
    picks, out = mod.finalize(plan_path)
    click.echo(_format_picks(picks))
    if _has_explanations(picks):
        click.echo("")
        click.echo(_format_picks_explained(picks))
    click.echo(f"\nsaved -> {out}")
    try:
        payload = json.loads(Path(out).read_text(encoding="utf-8"))
        _print_pick_warnings(picks, payload.get("requested_picks") or len(picks))
        _print_sell_reminder(picks, mode=mode)
    except Exception:
        pass


# ---------------------------- one-shot run --------------------------------


@cli.command("run")
@click.option("--mode", type=click.Choice(MODES), required=True)
@click.option("--picks", "-n", "n_picks", type=int, default=None,
              help="How many picks the agent should surface. Defaults to "
                   "pricing.default_picks in config.yaml.")
@click.option("--hose-only/--no-hose-only", default=False, show_default=True,
              help="Restrict the universe to HOSE-listed tickers only.")
@click.option("--etfs/--no-etfs", "include_etfs", default=True, show_default=True,
              help="Include HOSE-listed ETFs in the universe.")
@click.option("--exclude", "exclude", multiple=True,
              help="Ticker(s) to exclude from this run. Repeatable "
                   "(--exclude ACB --exclude HPG) or comma-separated.")
@click.option("--warm-only", default="yes", show_default=True,
              type=click.Choice(["yes", "no", "always"], case_sensitive=False),
              help="Cache strategy. `yes` = smart lazy fetch (skip warm, fetch "
                   "only stale + cold). `always` = pure offline, run on "
                   "whatever parquet is on disk. `no` = force full re-fetch.")
def run_cmd(mode: str, n_picks: int | None,
           hose_only: bool, include_etfs: bool,
           exclude: tuple[str, ...], warm_only: str) -> None:
    """End-to-end: fetch data -> emit the LLM plan for one mode.

    Designed to be invoked from a double-click .bat. Always runs on the full
    universe (no time cap); lazy caching keeps repeat runs fast.
    """
    import time as _time

    from .data.cache import cached_symbols
    from .data.fetcher import update_many
    from .selector import select as select_symbols

    if n_picks is not None and n_picks < 1:
        click.echo("ERROR: --picks must be >= 1.", err=True)
        sys.exit(2)
    requested_n = int(n_picks) if n_picks else int(
        load_config().pricing.get("default_picks", 5))
    exclude_set: set[str] = set()
    for raw in exclude:
        for tok in str(raw).split(","):
            tok = tok.strip().upper()
            if tok:
                exclude_set.add(tok)
    exclude_list: list[str] = sorted(exclude_set)
    if exclude_list:
        click.echo(f"[note] excluding {len(exclude_list)} ticker(s): "
                  f"{', '.join(exclude_list)}")

    started = _time.time()
    click.echo(f"mode={mode}  universe=entire (no cap)  picks={requested_n}")
    click.echo("")

    syms = select_symbols(target=10_000, hose_only=hose_only,
                          include_etfs=include_etfs, exclude=exclude_list)
    cached = set(cached_symbols())
    n_warm = len(set(syms) & cached)
    n_cold = len(syms) - n_warm
    label = "HOSE-only" if hose_only else "all exchanges"
    if not include_etfs:
        label += ", stocks-only"
    if exclude_list:
        label += f", excl={len(exclude_list)}"
    click.echo(f"selected {len(syms)} tickers  (warm={n_warm}, cold={n_cold})  [{label}]")
    click.echo("")

    from .data.fetcher import _VALID_SOURCES, audit_cache, quiet_vnstock_logger
    quiet_vnstock_logger()
    from .tracking import latest_expected_bar_date
    _expected_pre = latest_expected_bar_date()

    warm, stale, cold = audit_cache(syms, expected_bar=_expected_pre)
    expected_str = (str(_expected_pre.date()) if _expected_pre is not None
                   else "(unknown)")
    click.echo(f"cache audit (expected bar = {expected_str}):")
    click.echo(f"  {len(warm):>5} cached and current  ->  no API call")
    click.echo(f"  {len(stale):>5} cached but stale    ->  incremental fetch")
    click.echo(f"  {len(cold):>5} missing             ->  full-history fetch")

    warm_mode = warm_only.lower()
    if warm_mode == "always":
        available = warm + stale
        if cold:
            click.echo(f"  --warm-only=always: dropping {len(cold)} cold "
                      f"ticker(s) (no parquet); running on {len(available)} "
                      f"cached symbols ({len(warm)} current + "
                      f"{len(stale)} stale)")
        syms = available
        if not syms:
            click.echo("ERROR: --warm-only=always but no cached symbols; "
                      "populate the cache first with --warm-only=yes.", err=True)
            sys.exit(1)
    elif warm_mode == "yes":
        n_to_fetch = len(stale) + len(cold)
        if n_to_fetch == 0:
            click.echo(f"  --warm-only=yes: all {len(warm)} symbols current, "
                      "no API calls needed")
        else:
            click.echo(f"  --warm-only=yes: {len(warm)} current + "
                      f"{n_to_fetch} need fetch ({len(stale)} stale + "
                      f"{len(cold)} cold)")
    else:
        click.echo(f"  --warm-only=no: forcing full re-fetch of all "
                  f"{len(syms)} symbols (rate-limited)")
    click.echo("")

    if warm_mode == "always":
        click.echo("skipping data refresh (--warm-only=always)")
        results = {s: 0 for s in syms}
        ok, err, no_data = len(results), 0, 0
    else:
        click.echo(f"updating data ({len(_VALID_SOURCES)} sources: "
                  f"{', '.join(_VALID_SOURCES)})...")
        results = update_many(syms, full=(warm_mode == "no"))
        ok = sum(1 for v in results.values() if isinstance(v, int))
        no_data = sum(1 for v in results.values()
                     if isinstance(v, str) and v.startswith("NODATA:"))
        err = len(results) - ok - no_data
    click.echo(f"  fetched: ok={ok} err={err} no-data={no_data}")
    click.echo("")

    cached_now = set(cached_symbols())
    syms_with_data = [s for s in syms if s in cached_now]
    if not syms_with_data:
        click.echo("ERROR: no cached data for any selected symbol — cannot proceed.",
                  err=True)
        sys.exit(1)
    if len(syms_with_data) < len(syms):
        click.echo(f"  proceeding with {len(syms_with_data)} cached symbols "
                  f"({len(syms) - len(syms_with_data)} have no cached data)")
    syms = syms_with_data

    from .tracking import evaluate_pending, recent_performance
    updated = evaluate_pending()
    if not updated.empty:
        click.echo(f"auto-evaluated {len(updated)} prior pick(s):")
        click.echo(updated[["as_of", "mode", "symbol", "realized_return"]].to_string(index=False))
        click.echo("")
    perf = recent_performance(window_days=90, mode=mode)
    if perf.get("n", 0) > 0:
        click.echo(f"recent {mode} performance: n={perf['n']}  "
                  f"hit={perf['hit_rate']:.1%}  mean_ret={perf['mean_return']:+.4f}")
        click.echo("")

    click.echo(f"emitting plan (mode={mode})...")
    mod = _mode_module(mode)
    universe, plan_path = mod.run(n_picks=requested_n, symbols=syms,
                                 hose_only=hose_only, include_etfs=include_etfs,
                                 exclude=exclude_list)
    click.echo("")
    click.echo(f"eligible universe: {len(universe)} name(s) for the agent to pick from "
              f"(target {requested_n}).")
    click.echo(f"\nsaved -> {plan_path}")
    click.echo("")
    click.echo("==> NEXT (run inside your LLM coding-agent session):")
    click.echo("    1. Research the universe, choose & price the picks, fill the Results table.")
    click.echo("    2. Then run:")
    click.echo(f"       python -m stockpredict.cli finalize \"{plan_path}\"")

    elapsed = (_time.time() - started) / 60.0
    click.echo(f"\nelapsed: {elapsed:.1f} min")


# ---------------------------- evaluate / track ----------------------------


@cli.command("evaluate-fills")
@click.option("--refresh-data/--no-refresh-data", default=True,
              help="Run incremental fetch on tickers with un-stamped T+0 fills first.")
def evaluate_fills_cmd(refresh_data: bool) -> None:
    """Stamp T+0 fill outcomes for picks whose buy day has closed."""
    from .data.fetcher import update_many
    from .tracking import _read, evaluate_pending

    if refresh_data:
        df = _read()
        if not df.empty:
            pending_syms = sorted(df[~df["t0_evaluated"]]["symbol"].unique().tolist())
            if pending_syms:
                click.echo(f"refreshing data for {len(pending_syms)} symbols "
                          f"with un-stamped T+0 fills...")
                update_many(pending_syms, full=False)

    updated = evaluate_pending()
    click.echo(f"newly stamped: {len(updated)}")
    if not updated.empty:
        cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                            "entry_price", "t0_low", "evaluated", "realized_return"]
               if c in updated.columns]
        click.echo(updated[cols].to_string(index=False))


@cli.command("evaluate")
@click.option("--refresh-data/--no-refresh-data", default=True,
              help="Run incremental fetch on tickers in pending evaluations first.")
def evaluate_cmd(refresh_data: bool) -> None:
    """Score past predictions whose exit has now resolved."""
    from .data.fetcher import update_many
    from .tracking import _read, evaluate_pending, recent_performance

    if refresh_data:
        df = _read()
        if not df.empty:
            pending_syms = sorted(df[~df["evaluated"]]["symbol"].unique().tolist())
            if pending_syms:
                click.echo(f"refreshing data for {len(pending_syms)} symbols...")
                update_many(pending_syms, full=False)

    updated = evaluate_pending()
    click.echo(f"newly evaluated: {len(updated)}")
    if not updated.empty:
        cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                            "entry_price", "actual_exit", "realized_return"]
               if c in updated.columns]
        click.echo(updated[cols].to_string(index=False))

    click.echo("\n=== recent performance (last 90 days) ===")
    for mode in (*MODES, None):
        label = mode or "ALL"
        perf = recent_performance(window_days=90, mode=mode)
        if perf.get("n", 0) == 0:
            click.echo(f"  {label}: {perf.get('note')}")
        else:
            click.echo(f"  {label}: n={perf['n']}  hit={perf['hit_rate']:.1%}  "
                      f"mean_ret={perf['mean_return']:+.4f}  "
                      f"med_ret={perf['median_return']:+.4f}")


@cli.command("track")
@click.option("--mode", type=click.Choice(MODES), default=None)
@click.option("--limit", type=int, default=20)
def track_cmd(mode: str | None, limit: int) -> None:
    """Print the most recent prediction ledger entries."""
    from .tracking import _read

    df = _read()
    if df.empty:
        click.echo("no predictions recorded yet.")
        return
    if mode:
        df = df[df["mode"] == mode]
    df = df.sort_values(["as_of", "rank"], ascending=[False, True]).head(limit)
    cols = [c for c in ["as_of", "target_date", "mode", "symbol", "rank",
                        "entry_price", "realized_return", "evaluated"]
           if c in df.columns]
    click.echo(df[cols].to_string(index=False))


# ---------------------------- diagnostics ---------------------------------


@cli.command("status")
def status_cmd() -> None:
    """Show what's cached on disk."""
    from .config import cache_dir
    from .data.cache import cached_symbols
    from .data.dividends import dividends_cache_dir

    syms = cached_symbols()
    click.echo(f"cached OHLCV symbols: {len(syms)}")
    if syms:
        click.echo(f"  example: {syms[:5]}")
    div_dir = dividends_cache_dir()
    n_div = len(list(div_dir.glob("*.parquet"))) if div_dir.exists() else 0
    click.echo(f"cached dividend-history symbols: {n_div}  ({div_dir})")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
