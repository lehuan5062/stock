"""Prediction ledger + retroactive scoring.

Every prediction run appends one row per pick to `cache/predictions.parquet`
with the fields needed to evaluate it later. When OHLCV for the target date
becomes available, `evaluate_pending` fills in the realized return so the
ledger always reflects the latest known truth.

Recent performance is then exposed to the LLM modes so Claude can
factor in "what has been working / failing" when re-ranking new candidates.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import cache_dir, load_config
from .data.cache import read_ohlcv
from .pricing import profit_threshold, resolve_exit, settle_days


_LEDGER_FILE = "predictions.parquet"


def ledger_path() -> Path:
    return cache_dir() / _LEDGER_FILE


_LEDGER_COLUMNS = [
    "run_id", "signature", "as_of", "target_date",
    "mode", "symbol", "rank",
    "pred_mean", "news_score", "adjusted", "entry_price",
    "actual_exit", "realized_return", "evaluated",
    # Execution-quality fields, stamped by evaluate_pending from the bar at
    # `as_of` itself (the buy day — `effective_today_for_trading` rolls
    # `as_of` forward past the 14:30 cutoff so the recorded date is always
    # the day the user is supposed to buy). entry_slippage measures
    # whether the predicted entry_price was achievable: negative means the
    # day's low was BELOW predicted entry (we could have bought cheaper),
    # positive means the day's low was ABOVE predicted entry (the entry
    # was unreachable — the realized_return calc is then fictional).
    "t0_open", "t0_low", "t0_close", "entry_slippage",
    # Comma-separated kebab-case dimension tags Claude actually CITED in
    # Step 4 of the news plan (the bracketed `[tag-name]` markers it puts
    # at the start of each finding bullet). Lets recent_performance aggregate
    # hit-rate by dimension so Claude can learn which research categories
    # have actually predicted returns vs. been noise. Empty string = no
    # dimensions cited (e.g. base mode picks, or a Claude pick with
    # no Step 4 findings at all).
    "dimensions_cited",
    # Low-prediction fields. ``pred_low`` is the model's predicted next-
    # day low return (negative typically); ``entry_limit_price`` is the
    # actual quoted limit-buy price (= close * (1 + pred_low)) the user
    # was supposed to place. ``entry_limit_filled`` is set by
    # evaluate_pending once the buy-day bar lands: True when t0_low <=
    # entry_limit_price (the limit would have filled), False otherwise.
    # ``t0_evaluated`` flips True once the buy-day bar is in cache and
    # the limit-fill outcome has been stamped — independent of T+N
    # evaluation, so a Claude self-correction can be triggered the very
    # next trading day instead of waiting for the realized return.
    "pred_low", "entry_limit_price", "entry_limit_filled", "t0_evaluated",
    # Rebound-strategy fields (strategy.mode: rebound). ``pred_days`` (N) and
    # ``pred_profit`` (P) are the KM head's estimates; ``pred_recovery_prob`` its
    # eventual-recovery probability; the pick's ``target_date`` is
    # ``as_of + round(pred_days)`` trading days (the PREDICTED sell day). The
    # flexible-exit evaluator scans forward from ``as_of`` for the first day the
    # close first clears the profit threshold and stamps ``actual_exit_date`` +
    # ``recovered_flag`` there (which may differ from the predicted target_date).
    # A pick that has not recovered yet stays pending (evaluated=False).
    "pred_days", "pred_profit", "pred_recovery_prob",
    "actual_exit_date", "recovered_flag",
    # How a rebound pick was closed: "recovery" | "" (open / not yet
    # resolved / legacy row). Filled by evaluate_pending.
    "exit_reason",
    # Which ranking formula produced this row's ``rank``. The ledger keeps
    # ``rank`` but NOT ``score``, so a mode that changes how it scores would
    # otherwise silently mix two rank populations in one append-only file with
    # nothing to tell them apart — and ``analyze.mode_compare`` would average
    # two different strategies under one mode label. Modes that have never
    # changed their formula leave this empty; empty also means "legacy row,
    # pre-dating the stamp". Set by the mode's finalize (see
    # ``modes.dividend.SCORE_FORMULA``).
    "score_formula",
]


# Columns that may be missing from old ledger files; _read backfills them
# with NaN so the file stays readable across schema changes.
_NEW_FLOAT_COLUMNS = ("t0_open", "t0_low", "t0_close", "entry_slippage",
                      "pred_low", "entry_limit_price",
                      "pred_days", "pred_profit", "pred_recovery_prob")
_NEW_STRING_COLUMNS = ("dimensions_cited", "exit_reason", "score_formula")
# Boolean columns added in the low-prediction release. Default for legacy
# rows: ``entry_limit_filled`` is False (no limit was placed); for
# ``t0_evaluated`` the backfill is "True if evaluated else False" because
# pre-existing evaluated rows already had their t0 bar stamped during
# the original evaluate_pending pass.
_NEW_BOOL_COLUMNS = ("entry_limit_filled", "recovered_flag")


def _normalize_dimensions(value) -> str:
    """Coerce a `dimensions_cited` cell value into a canonical
    comma-separated string. Accepts:
      * a string (returned trimmed and lower-cased)
      * a list/tuple/set of tag names (joined with commas)
      * anything else → empty string

    The downstream by-dimension aggregator splits on `,` so we strip out
    any whitespace around tags here too — saves the aggregator from doing
    it on every row.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        tags = [str(v).strip().lower() for v in value if str(v).strip()]
        return ",".join(tags)
    s = str(value).strip().lower()
    if not s or s == "nan":  # pandas turns missing data into the literal "nan" string sometimes
        return ""
    # Already comma-separated — re-split + re-join to normalize whitespace.
    return ",".join(t.strip() for t in s.split(",") if t.strip())


def _read() -> pd.DataFrame:
    p = ledger_path()
    if not p.exists():
        return pd.DataFrame(columns=_LEDGER_COLUMNS)
    df = pd.read_parquet(p)
    # Backfill signature for old ledger rows: derive from run_id by
    # stripping the YYYYMMDD_ date prefix (or full run_id if no underscore).
    if "signature" not in df.columns:
        def _derive(rid):
            try:
                rid = str(rid)
                first_underscore = rid.find("_")
                if first_underscore > 0 and rid[:first_underscore].isdigit():
                    return rid[first_underscore + 1:]
            except Exception:
                pass
            return ""
        df["signature"] = df["run_id"].map(_derive) if "run_id" in df.columns else ""
    # Backfill execution-quality columns added in the entry-slippage release.
    # Old rows get NaN — they'll be filled retroactively the next time
    # evaluate_pending revisits them (idempotent: it sets evaluated=True so
    # they aren't actually revisited unless the user resets that flag).
    for col in _NEW_FLOAT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Backfill string columns added later (dimensions_cited). Empty string
    # is the canonical "no data" value here, NOT NaN — this keeps the
    # column dtype consistent and makes the by-dimension aggregator's
    # split logic uniform.
    for col in _NEW_STRING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Backfill boolean columns added in the low-prediction release.
    for col in _NEW_BOOL_COLUMNS:
        if col not in df.columns:
            df[col] = False
    # ``t0_evaluated`` — separate fill rule: legacy rows that were already
    # evaluated for T+N had their buy-day bar stamped at that time, so we
    # treat them as "T+0 already done". Unevaluated legacy rows start as
    # False and get picked up by the next evaluate_pending pass.
    if "t0_evaluated" not in df.columns:
        if "evaluated" in df.columns:
            df["t0_evaluated"] = df["evaluated"].astype(bool)
        else:
            df["t0_evaluated"] = False
    # Rebound exit-date on legacy ledgers is NaT (recovered_flag is backfilled
    # False via _NEW_BOOL_COLUMNS above).
    if "actual_exit_date" not in df.columns:
        df["actual_exit_date"] = pd.NaT
    return df


def _write(df: pd.DataFrame) -> None:
    df.to_parquet(ledger_path(), index=False)


from functools import lru_cache


@lru_cache(maxsize=1)
def _trading_calendar_cached() -> pd.DatetimeIndex:
    """Union of cached OHLCV indices = the actual Vietnamese trading-day
    calendar. Excludes weekends AND every Vietnamese holiday the market
    historically closed for (Tết, April 30, May 1, Sept 2, ad-hoc closures).
    Returns empty if the cache hasn't been populated yet."""
    from .data.cache import cached_symbols, read_ohlcv
    syms = cached_symbols()
    if not syms:
        return pd.DatetimeIndex([])
    # Cap the union to a sample of liquid tickers for speed; if a date is a
    # trading day at all, at least one of these will have traded.
    union: set = set()
    for s in syms[:50]:
        try:
            df = read_ohlcv(s)
        except Exception:
            continue
        if not df.empty:
            union.update(pd.DatetimeIndex(df.index).normalize())
    return pd.DatetimeIndex(sorted(union))


def _invalidate_trading_calendar_cache() -> None:
    """Drop the cached trading-day calendar so the next call rebuilds it.
    Call after ingesting new OHLCV data within the same Python process."""
    _trading_calendar_cached.cache_clear()


# Vietnamese fixed-date public holidays (the market is closed). Tết and the
# Hung Kings Festival are lunar so their solar dates change each year — the
# real calendar in the OHLCV cache covers historical instances; for the
# couple of weeks we ever project forward we accept that those two might
# leak through (small, recoverable error: a wasted fetch returning nothing).
VN_FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset({
    (1, 1),    # New Year's Day
    (4, 30),   # Reunification Day
    (5, 1),    # Labor Day
    (9, 2),    # National Day
})


def _is_vn_fixed_holiday(d: pd.Timestamp) -> bool:
    return (d.month, d.day) in VN_FIXED_HOLIDAYS


def _extended_calendar(through: pd.Timestamp,
                       calendar: pd.DatetimeIndex | None = None
                       ) -> pd.DatetimeIndex:
    """Cached trading calendar + projected future weekdays (Mon-Fri, minus
    Vietnamese fixed-date holidays) up to ``through``. We don't know
    floating-date holidays (Tết, Hung Kings) in advance, so the projection
    may include those rare weekdays — worst case is one wasted fetch that
    returns nothing, and the next run catches up."""
    cal = calendar if calendar is not None else _trading_calendar_cached()
    through = pd.Timestamp(through).normalize()
    if len(cal) == 0:
        start = pd.Timestamp.today().normalize()
        future = pd.bdate_range(start=start, end=through)
        future = pd.DatetimeIndex(d for d in future if not _is_vn_fixed_holiday(d))
        return future
    if through <= cal[-1]:
        return cal
    future_start = cal[-1] + pd.Timedelta(days=1)
    future = pd.bdate_range(start=future_start, end=through)
    future = pd.DatetimeIndex(d for d in future if not _is_vn_fixed_holiday(d))
    return cal.append(future).sort_values()


def latest_expected_bar_date(now: dt.datetime | None = None,
                              post_close_buffer_minutes: int = 15,
                              calendar: pd.DatetimeIndex | None = None
                              ) -> pd.Timestamp | None:
    """Date of the most recent end-of-day bar that the broker should already
    have published.

    The OHLCV cache should be considered current iff
    ``cache.index.max() >= latest_expected_bar_date()``. Re-fetching beyond
    that date is wasted budget — the data either doesn't exist yet (mid-
    trading) or never will (weekends / holidays).

    Decision tree:
      * If today is in the trading calendar AND we're past 14:45 + buffer
        (default 15 min, so 15:00) → today's close has been published.
      * If today is in the trading calendar but we're earlier in the day
        → today's close is not yet final; the latest finalized bar is the
        previous trading day.
      * If today is NOT in the trading calendar (weekend, holiday) →
        the latest finalized bar is the most recent trading day before
        today.

    Returns ``None`` only when the trading calendar is empty (first run on
    a brand-new install)."""
    now = now or dt.datetime.now()
    today = pd.Timestamp(now.date()).normalize()

    cal = _extended_calendar(today, calendar=calendar)
    if len(cal) == 0:
        return None

    # Latest calendar entry at-or-before today.
    pos = int(cal.searchsorted(today, side="right")) - 1
    if pos < 0:
        return None
    latest = cal[pos].normalize()

    if latest != today:
        # Today is a non-trading day — latest finalized bar is whatever
        # the most recent trading day was.
        return latest

    # Today is a trading day. Decide whether today's close has published yet.
    cutoff = dt.time(14, 45)
    cutoff_min = cutoff.hour * 60 + cutoff.minute + int(post_close_buffer_minutes)
    now_min = now.time().hour * 60 + now.time().minute
    if now_min < cutoff_min:
        # Pre-close: today's close not yet final. Step back one trading day.
        if pos >= 1:
            return cal[pos - 1].normalize()
        return None
    return latest


def effective_today_for_trading(now: dt.datetime | None = None,
                                cutoff_hour: int = 14,
                                cutoff_minute: int = 30,
                                calendar: pd.DatetimeIndex | None = None
                                ) -> pd.Timestamp:
    """Return the date the picks should be stamped against.

    Vietnamese exchanges' continuous-trading session ends at 14:30 (ATC
    runs until 14:45). After 14:30 today's close is effectively locked
    in — a buy order placed now can't fill at today's close and will
    settle starting tomorrow. So picks made after the cutoff should treat
    the next trading day as T+0.

    Returns: a normalized pandas Timestamp.
      * trading day, `now <= cutoff`  -> today
      * trading day, `now > cutoff`   -> the next trading day after today
      * non-trading day (weekend /
        holiday), any time of day      -> the next trading day after today
    Weekends and Vietnamese holidays are skipped via the cached trading
    calendar; if the calendar is empty we fall back to BDay. A run on a
    non-trading day always rolls forward — there is no "today" to buy on, so
    the cutoff is irrelevant.
    """
    now = now or dt.datetime.now()
    today = pd.Timestamp(now.date()).normalize()
    cutoff = dt.time(cutoff_hour, cutoff_minute)
    past_cutoff = now.time() > cutoff
    cal = calendar if calendar is not None else _trading_calendar_cached()

    if len(cal) == 0:
        # No cached calendar — best-effort: today counts only if it's a
        # weekday that isn't a fixed-date holiday, and we're before the cutoff.
        today_tradeable = today.weekday() < 5 and not _is_vn_fixed_holiday(today)
        if today_tradeable and not past_cutoff:
            return today
        return (today + pd.tseries.offsets.BDay(1)).normalize()

    # Is today itself a trading day (present in the calendar)?
    pos_left = int(cal.searchsorted(today, side="left"))
    today_is_trading = pos_left < len(cal) and cal[pos_left].normalize() == today

    # Before the cutoff on a trading day: buy today (close not yet locked in).
    if today_is_trading and not past_cutoff:
        return today

    # Otherwise — past the cutoff, OR today is a weekend/holiday — advance to
    # the next trading day strictly after today.
    pos = int(cal.searchsorted(today, side="right"))
    if pos < len(cal):
        return cal[pos].normalize()
    # today is past the end of the cached calendar — project forward as
    # weekdays, still honoring the cutoff for a tradeable weekday.
    today_tradeable = today.weekday() < 5 and not _is_vn_fixed_holiday(today)
    if today_tradeable and not past_cutoff:
        return today
    return (today + pd.tseries.offsets.BDay(1)).normalize()


def _next_trading_offset(start_date: pd.Timestamp, offset: int) -> pd.Timestamp:
    """Return start_date + `offset` TRADING days using the actual cached
    Vietnamese trading-day calendar (weekends + holidays excluded).

    Future trading days beyond the cache are projected as weekdays
    (Mon-Fri). If start_date itself is past the end of the cache, we
    project from start_date — not from the last cached day.
    """
    cal = _trading_calendar_cached()
    start_norm = pd.Timestamp(start_date).normalize()
    if len(cal) == 0:
        # No OHLCV cache yet — fall back to BDay arithmetic from start_date.
        anchor = start_norm if start_norm.weekday() < 5 else (
            (start_norm + pd.tseries.offsets.BDay(1)).normalize()
        )
        return (anchor + pd.tseries.offsets.BDay(int(offset))).normalize()

    if start_norm > cal[-1]:
        # Start is past the end of the cached calendar — anchor on
        # start_date (or the next weekday if it's a weekend) and project
        # forward with BDay.
        anchor = start_norm if start_norm.weekday() < 5 else (
            (start_norm + pd.tseries.offsets.BDay(1)).normalize()
        )
        return (anchor + pd.tseries.offsets.BDay(int(offset))).normalize()

    # Build an extended calendar that reaches at least `offset` weekdays past
    # the start, so the position arithmetic always lands inside `cal`.
    horizon = start_norm + pd.tseries.offsets.BDay(int(offset) + 5)
    cal = _extended_calendar(horizon, calendar=cal)
    pos = int(cal.searchsorted(start_norm, side="left"))
    target_pos = pos + int(offset)
    if target_pos < len(cal):
        return cal[target_pos].normalize()
    # Should not happen with the extension above, but be safe.
    extra = target_pos - len(cal) + 1
    return (cal[-1] + pd.tseries.offsets.BDay(extra)).normalize()


def run_signature(mode: str,
                  hose_only: bool = False,
                  include_etfs: bool = True,
                  exclude: Iterable[str] | None = None) -> str:
    """Stable signature for a parameter set: distinct combinations get
    distinct signatures so saved artifacts don't override each other,
    while a re-run of the same parameters does override (idempotent).
    Used as both the filename suffix and the ledger ``run_id`` base.

    (The old T+2 pipeline embedded a ``d{horizon}`` token here; the rebound
    exit is flexible, so the horizon token is gone — old ``_d2`` artifacts
    simply never collide with new ones.)

    ``include_etfs`` defaults to True (ETFs mixed in). Only when it's False
    is the signature tagged ``noETF``. ``exclude`` is the per-session ticker
    blacklist; when non-empty the signature gets an ``x{TICKERS}`` suffix
    (sorted, dash-joined) so an excluded rerun doesn't clobber the full run.
    """
    parts = [mode]
    if hose_only:
        parts.append("HOSE")
    if not include_etfs:
        parts.append("noETF")
    if exclude:
        excl_sorted = sorted({str(s).upper() for s in exclude})
        if excl_sorted:
            parts.append("x" + "-".join(excl_sorted))
    return "_".join(parts)


def record(picks: pd.DataFrame, mode: str, as_of: str | dt.date | None = None,
           run_id: str | None = None,
           hose_only: bool = False,
           include_etfs: bool = True,
           exclude: Iterable[str] | None = None) -> int:
    """Append one row per pick to the ledger. Returns number of rows added.

    `picks` is the dataframe returned by mode runs; must have at minimum a
    ``symbol`` column, plus the rebound estimate columns
    (``pred_days`` / ``pred_profit``). The default ``run_id`` includes
    mode/hose-only/exclude so a same-day rerun with different parameters does
    not clobber prior rows.
    """
    if picks is None or len(picks) == 0:
        return 0

    as_of = (pd.to_datetime(as_of) if as_of is not None
             else effective_today_for_trading())
    sig = run_signature(mode=mode,
                        hose_only=hose_only,
                        include_etfs=include_etfs,
                        exclude=exclude)
    if run_id is None:
        rid = f"{as_of.strftime('%Y%m%d')}_{sig}"
    else:
        rid = run_id

    rows = []
    for i, r in picks.reset_index(drop=True).iterrows():
        sym = str(r["symbol"]).upper()
        # Low-prediction fields. ``pred_low`` is the predicted next-day
        # low return; ``entry_limit_price`` is the limit-buy price the
        # user was supposed to place (in thousand-VND, same scale as
        # ``entry_price``). When the low head wasn't trained, both are
        # NaN and evaluate_pending later treats the row as having no
        # limit (entry_limit_filled stays False).
        pred_low_val = (float(r["pred_low"])
                        if "pred_low" in r and pd.notna(r["pred_low"])
                        else np.nan)
        # Prefer the explicit close-price column when available; otherwise
        # fall back to ``entry_price`` (= close in the legacy ledger).
        close_for_limit = float(r["close"]) if "close" in r and pd.notna(r["close"]) else np.nan
        if not np.isnan(pred_low_val) and not np.isnan(close_for_limit):
            # ``pred_low`` is clipped at 0 in pricing; we mirror that
            # here so the recorded limit price is never above close.
            limit_price = close_for_limit * (1.0 + min(pred_low_val, 0.0))
        else:
            limit_price = np.nan
        # Each pick carries a predicted hold (N = pred_days). Its target_date is
        # as_of + round(N) trading days — the PREDICTED sell day, informational
        # only (the evaluator settles on the first actual recovery). Falls back
        # to 1 trading day if N is somehow missing.
        pred_days_val = (float(r["pred_days"])
                         if "pred_days" in r and pd.notna(r["pred_days"])
                         else np.nan)
        pred_profit_val = (float(r["pred_profit"])
                           if "pred_profit" in r and pd.notna(r["pred_profit"])
                           else np.nan)
        pred_prob_val = (float(r["pred_recovery_prob"])
                         if "pred_recovery_prob" in r and pd.notna(r["pred_recovery_prob"])
                         else np.nan)
        row_off = (max(int(round(pred_days_val)), 1)
                   if not np.isnan(pred_days_val) else 1)
        row_target = _next_trading_offset(as_of, row_off)
        rows.append({
            "run_id": rid,
            "signature": sig,
            "as_of": as_of.normalize(),
            "target_date": row_target,
            "mode": mode,
            "symbol": sym,
            "rank": int(r.get("rank", i + 1)),
            "pred_mean": float(r.get("pred_mean", np.nan)),
            "news_score": int(r["news_score"]) if "news_score" in r and pd.notna(r["news_score"]) else 0,
            "adjusted": float(r["adjusted"]) if "adjusted" in r and pd.notna(r["adjusted"]) else float(r.get("pred_mean", np.nan)),
            "entry_price": float(r["close"]) if "close" in r and pd.notna(r["close"]) else np.nan,
            "actual_exit": np.nan,
            "realized_return": np.nan,
            "evaluated": False,
            # Execution-quality fields filled by evaluate_pending once the
            # buy-day bar (the bar at `as_of` itself) lands in the cache.
            "t0_open": np.nan,
            "t0_low": np.nan,
            "t0_close": np.nan,
            "entry_slippage": np.nan,
            # Dimension tags Claude actually cited in Step 4 of the plan.
            # Comma-separated string; empty for non-claude modes or rows
            # where Step 4 was left blank.
            "dimensions_cited": _normalize_dimensions(r.get("dimensions_cited")),
            # Low-head outputs (NaN when low model isn't trained yet).
            "pred_low": pred_low_val,
            "entry_limit_price": limit_price,
            "entry_limit_filled": False,
            "t0_evaluated": False,
            # Rebound-strategy fields (NaN/NaT/False for legacy picks).
            "pred_days": pred_days_val,
            "pred_profit": pred_profit_val,
            "pred_recovery_prob": pred_prob_val,
            "actual_exit_date": pd.NaT,
            "recovered_flag": False,
            "exit_reason": "",
            # Ranking-formula version stamp (see _LEDGER_COLUMNS). Empty for
            # modes that have never redefined their score.
            "score_formula": str(r.get("score_formula", "") or ""),
        })
    add = pd.DataFrame(rows)
    df = _read()
    # de-dupe on (run_id, symbol) so re-runs replace prior rows for that day
    if not df.empty:
        keep_mask = ~df.set_index(["run_id", "symbol"]).index.isin(
            add.set_index(["run_id", "symbol"]).index
        )
        df = df[keep_mask]
    _write(pd.concat([df, add], ignore_index=True))
    return len(add)


def evaluate_pending(today: dt.date | None = None) -> pd.DataFrame:
    """Fill realized outcomes for any pending row whose data is now in cache.

    Two evaluation stages run independently:

    1. **T+0 limit-fill stage** (``t0_evaluated``): triggers as soon as the
       buy-day bar (the bar at ``as_of`` itself) lands in the OHLCV cache.
       Stamps ``t0_open / t0_low / t0_close``, ``entry_slippage`` (vs the
       close-anchored ``entry_price``), and — if ``entry_limit_price`` was
       quoted — ``entry_limit_filled`` (True iff ``t0_low <=
       entry_limit_price``). This stage lets the user / Claude run
       limit-fill self-correction the very next trading day, before T+N
       has elapsed.

    2. **T+N realized stage** (``evaluated``): existing logic — once
       ``target_date <= today`` AND OHLCV is cached through that day,
       fills ``actual_exit`` and ``realized_return`` (computed against
       the close-anchored ``entry_price``).

    Both flags are set independently. Returns the rows that were updated
    by either stage.
    """
    df = _read()
    if df.empty:
        return df
    today = today or dt.date.today()
    today_ts = pd.Timestamp(today).normalize()

    # Union of: rows needing T+0 stamping AND rows needing T+N stamping.
    # A row may need either, both, or neither on any given pass.
    t0_pending_mask = (~df["t0_evaluated"]) & (df["as_of"] <= today_ts)
    # Rebound picks (pred_profit set) exit flexibly — they may recover before or
    # after their predicted target_date, so re-check them every pass once the buy
    # day exists, not only after target_date. Legacy picks wait for target_date.
    is_rebound = df["pred_profit"].notna() if "pred_profit" in df.columns else pd.Series(False, index=df.index)
    tN_pending_mask = (~df["evaluated"]) & (
        (df["target_date"] <= today_ts) | (is_rebound & (df["as_of"] <= today_ts))
    )
    pending = df[t0_pending_mask | tN_pending_mask]
    if pending.empty:
        return pending

    updated_rows = []
    for idx, row in pending.iterrows():
        sym = row["symbol"]
        ohlcv = read_ohlcv(sym)
        if ohlcv.empty:
            continue
        as_of = pd.Timestamp(row["as_of"]).normalize()
        changed = False

        # --- Stage 1: T+0 limit-fill stamping ---------------------------
        # The user is supposed to BUY on `as_of` itself (the "T+0" buy
        # day — `effective_today_for_trading` rolled `as_of` forward past
        # the 14:30 cutoff at record time, so this is always the day the
        # user could actually trade). Stamp that bar's OHLC + slippage +
        # limit-fill outcome as soon as it's available in cache. Use
        # `index >= as_of` rather than `==` so weekend/holiday rolls
        # (where the buy day might land on a non-trading calendar date
        # because the cache calendar lags) still pick the next available
        # bar.
        needs_t0 = not bool(row["t0_evaluated"])
        if needs_t0:
            buy_day_bars = ohlcv[ohlcv.index >= as_of]
            if not buy_day_bars.empty:
                bar = buy_day_bars.iloc[0]
                t0_open = float(bar["open"]) if "open" in bar and pd.notna(bar["open"]) else np.nan
                t0_low = float(bar["low"]) if "low" in bar and pd.notna(bar["low"]) else np.nan
                t0_close = float(bar["close"]) if "close" in bar and pd.notna(bar["close"]) else np.nan
                df.at[idx, "t0_open"] = t0_open
                df.at[idx, "t0_low"] = t0_low
                df.at[idx, "t0_close"] = t0_close

                # Slippage to the day's low, measured vs. the close-anchored
                # ``entry_price`` (preserved for backward compat with
                # legacy ledgers). Negative => market dipped below
                # ``entry_price``; positive => unreachable at that price.
                entry_close = row["entry_price"]
                if pd.notna(t0_low) and pd.notna(entry_close) and float(entry_close) > 0:
                    df.at[idx, "entry_slippage"] = (t0_low - float(entry_close)) / float(entry_close)

                # Limit-fill check: if the row recorded an
                # ``entry_limit_price``, the limit-buy filled iff the
                # day's low touched/breached it.
                limit_price = row["entry_limit_price"]
                if pd.notna(t0_low) and pd.notna(limit_price):
                    df.at[idx, "entry_limit_filled"] = bool(t0_low <= float(limit_price))
                else:
                    df.at[idx, "entry_limit_filled"] = False

                df.at[idx, "t0_evaluated"] = True
                changed = True

        # --- Stage 2a: rebound flexible-exit stamping -------------------
        # A rebound pick (pred_profit set) is held until the close first clears
        # the profit threshold above the entry (today's close on the buy day).
        # Scan forward from the buy day; the first such day is the realized exit.
        # If no cached bar has recovered yet, the position is still open — leave
        # it pending (evaluated stays False) so a later pass resolves it.
        is_rebound_row = ("pred_profit" in row and pd.notna(row["pred_profit"]))
        if is_rebound_row and not bool(row["evaluated"]):
            entry = row["entry_price"]
            if pd.notna(entry) and float(entry) > 0:
                thr = profit_threshold()
                fut = ohlcv[ohlcv.index > as_of]
                ex = resolve_exit(fut["close"].astype(float).to_numpy(),
                                  float(entry), thr, min_hold_days=settle_days())
                # Only a resolved exit whose bar has actually landed closes the
                # position; otherwise it stays pending (open) for a later pass.
                if ex is not None:
                    k = int(ex["k"])
                    exit_date = pd.Timestamp(fut.index[k - 1]).normalize()
                    actual_exit = float(ex["exit_close"])
                    df.at[idx, "actual_exit"] = actual_exit
                    df.at[idx, "actual_exit_date"] = exit_date
                    df.at[idx, "realized_return"] = actual_exit / float(entry) - 1.0
                    df.at[idx, "recovered_flag"] = ex["reason"] == "recovery"
                    df.at[idx, "exit_reason"] = ex["reason"]
                    df.at[idx, "evaluated"] = True
                    changed = True

        # --- Stage 2b: legacy fixed-horizon realized-return stamping -----
        # Dividend picks are a HOLD with no target/no T+N exit — there is
        # nothing to resolve here (a real "realized outcome" for a dividend
        # thesis is "did the payout keep coming", not a price target; that's
        # a future enhancement, not modeled by this ledger yet). Leave them
        # pending forever rather than mis-stamping a fictional T+1 exit.
        is_dividend_row = str(row.get("mode", "")) == "dividend"
        needs_tN = (not is_rebound_row
                    and not is_dividend_row
                    and not bool(row["evaluated"])
                    and pd.Timestamp(row["target_date"]).normalize() <= today_ts)
        if needs_tN:
            target = pd.Timestamp(row["target_date"]).normalize()
            on_or_after = ohlcv[ohlcv.index >= target]
            if not on_or_after.empty:
                actual_exit = float(on_or_after.iloc[0]["close"])
                entry = row["entry_price"]
                if pd.isna(entry):
                    on_at = ohlcv[ohlcv.index <= as_of]
                    if not on_at.empty:
                        entry = float(on_at.iloc[-1]["close"])
                        df.at[idx, "entry_price"] = entry
                if pd.notna(entry) and float(entry) > 0:
                    realized = actual_exit / float(entry) - 1.0
                    df.at[idx, "actual_exit"] = actual_exit
                    df.at[idx, "realized_return"] = realized
                    df.at[idx, "evaluated"] = True
                    changed = True

        if changed:
            updated_rows.append(idx)

    _write(df)
    return df.loc[updated_rows]


def recent_performance(window_days: int = 90,
                       mode: str | None = None) -> dict:
    """Summary stats over the last ``window_days`` of evaluated rows.
    Grouped views: by run signature (exact parameter match), by news_score,
    and by cited research dimension."""
    df = _read()
    if df.empty:
        return {"n": 0, "note": "no predictions recorded yet"}
    if mode:
        df = df[df["mode"] == mode]
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)
    df = df[df["as_of"] >= cutoff]

    # limit_fill stats need only t0_evaluated (buy day closed), not full
    # T+N evaluation. Compute on the mode+window slice before pruning to
    # evaluated=True, so newly-stamped buy days are visible before T+N.
    limit_fill = _limit_fill_stats(df)

    df = df[df["evaluated"]]
    if df.empty:
        return {"n": 0, "note": f"no evaluated predictions in last {window_days} days", "limit_fill": limit_fill}

    rets = df["realized_return"].astype(float)
    out = {
        "n": int(len(df)),
        "hit_rate": float((rets > 0).mean()),
        "mean_return": float(rets.mean()),
        "median_return": float(rets.median()),
        "best_pick": _format_pick(df.loc[rets.idxmax()]),
        "worst_pick": _format_pick(df.loc[rets.idxmin()]),
        "by_news_score": _by_news_score(df),
        "by_run_signature": _by_run_signature(df),
        "recent_5": _recent_picks_sample(df, n=5),
        "entry_slippage": _entry_slippage_stats(df),
        "by_dimension": _by_dimension(df),
        "limit_fill": limit_fill,
    }
    return out


def _limit_fill_stats(df: pd.DataFrame) -> dict | None:
    """Stats on whether the limit-buy entry quoted by the low head actually
    filled. Returns ``None`` when no row has a recorded ``entry_limit_price``
    (e.g. ledgers from before the low-prediction release, or runs where
    ``models/low_latest.pkl`` didn't exist).

    Keys:
      - ``n`` — rows with a non-null ``entry_limit_price`` AND a stamped
        ``t0_evaluated=True`` (the buy-day bar has landed)
      - ``fill_rate`` — fraction of those where the limit actually filled
        (``t0_low <= entry_limit_price``)
      - ``mean_dip_quoted`` — average predicted dip relative to close
        (``pred_low``); negative numbers mean we asked for a dip below close
      - ``mean_dip_actual`` — average actual dip ``(t0_low - close) / close``
        on the buy day, conditional on ``t0_low`` being available
      - ``calibration`` — ``mean_dip_actual - mean_dip_quoted``: positive
        means actual dips were SHALLOWER than quoted (we set the limit
        too low and missed fills); negative means dips were DEEPER than
        quoted (we could have set the limit lower and still filled)
    """
    if "entry_limit_price" not in df.columns:
        return None
    sub = df[df["entry_limit_price"].notna() & df["t0_evaluated"].astype(bool)]
    if sub.empty:
        return None

    filled = sub["entry_limit_filled"].astype(bool)
    pred_low = sub["pred_low"].astype(float)
    # Actual dip relative to entry_price (= close at as_of). Skip rows
    # missing either side.
    has_dip = sub["t0_low"].notna() & sub["entry_price"].notna() & (sub["entry_price"] > 0)
    if has_dip.any():
        dip_actual = ((sub.loc[has_dip, "t0_low"].astype(float)
                       - sub.loc[has_dip, "entry_price"].astype(float))
                      / sub.loc[has_dip, "entry_price"].astype(float))
        mean_dip_actual = float(dip_actual.mean())
    else:
        mean_dip_actual = float("nan")
    mean_dip_quoted = float(pred_low.dropna().mean()) if pred_low.notna().any() else float("nan")
    calibration = (mean_dip_actual - mean_dip_quoted
                   if not (pd.isna(mean_dip_actual) or pd.isna(mean_dip_quoted))
                   else float("nan"))

    return {
        "n": int(len(sub)),
        "fill_rate": float(filled.mean()),
        "mean_dip_quoted": mean_dip_quoted,
        "mean_dip_actual": mean_dip_actual,
        "calibration": calibration,
    }


def _by_dimension(df: pd.DataFrame) -> dict:
    """Hit-rate / mean-return per dimension tag Claude actually cited in
    Step 4. A pick that cites three tags contributes to all three dimensions
    (so a row can land in multiple buckets).

    Returns ``{}`` when no row has any cited dimension — typical of pre-
    feature ledgers, base-only ledgers, or fresh installs where
    Claude hasn't yet completed Step 4 on a finished prediction.
    """
    if "dimensions_cited" not in df.columns:
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        tags_str = row.get("dimensions_cited", "") or ""
        if not isinstance(tags_str, str) or not tags_str.strip():
            continue
        ret = float(row["realized_return"])
        for tag in (t.strip() for t in tags_str.split(",")):
            if not tag:
                continue
            bucket = out.setdefault(tag, {"_returns": []})
            bucket["_returns"].append(ret)
    # Finalize: turn the per-tag return list into n / hit_rate / mean / median.
    final: dict = {}
    for tag, bucket in out.items():
        rets = bucket["_returns"]
        s = pd.Series(rets, dtype=float)
        final[tag] = {
            "n": int(len(s)),
            "hit_rate": float((s > 0).mean()),
            "mean_return": float(s.mean()),
            "median_return": float(s.median()),
        }
    return final


def _entry_slippage_stats(df: pd.DataFrame) -> dict | None:
    """Stats on whether predicted entry_price was actually achievable on the
    buy day. Returns ``None`` if no rows have entry_slippage filled (e.g. all
    rows pre-date the entry-slippage release and haven't been re-evaluated).

    Keys:
      - ``n`` — rows with a non-null entry_slippage
      - ``mean`` / ``median`` — average slippage to the buy day's low
      - ``pct_unreachable`` — fraction where t0_low > entry_price (the
        predicted entry was never available; realized_return is fictional)
      - ``mean_savings_when_reachable`` — average % below predicted entry
        a limit-order at t0_low would have hit, conditional on the entry
        being reachable. Always ≥ 0; bigger means we routinely could have
        gotten in cheaper than `entry_price`.
    """
    if "entry_slippage" not in df.columns:
        return None
    s = df["entry_slippage"].astype(float).dropna()
    if s.empty:
        return None
    reachable = s[s <= 0]
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "pct_unreachable": float((s > 0).mean()),
        "mean_savings_when_reachable": (
            float((-reachable).mean()) if not reachable.empty else 0.0
        ),
    }


def _by_run_signature(df: pd.DataFrame) -> dict:
    """Group evaluated picks by full run signature (mode + hose flag +
    exclude). Lets the LLM see, for example, that its `claude` runs hit 60%
    but `claude_HOSE` runs hit 40% — hose-only / exclude filters change the
    population."""
    if "signature" not in df.columns:
        return {}
    out: dict = {}
    for sig, g in df.groupby("signature"):
        if not sig:
            continue
        out[str(sig)] = {
            "n": int(len(g)),
            "hit_rate": float((g["realized_return"] > 0).mean()),
            "mean_return": float(g["realized_return"].mean()),
            "median_return": float(g["realized_return"].median()),
        }
    return out


def _format_pick(row: pd.Series) -> dict:
    out = {
        "symbol": str(row["symbol"]),
        "as_of": str(pd.Timestamp(row["as_of"]).date()),
        "news_score": int(row.get("news_score", 0) or 0),
        "realized_return": float(row["realized_return"]),
    }
    # Rebound estimate fields (absent on very old legacy rows).
    for k in ("pred_days", "pred_profit"):
        v = row.get(k)
        if v is not None and pd.notna(v):
            out[k] = float(v)
    return out


def _by_news_score(df: pd.DataFrame) -> dict:
    out = {}
    for s, g in df.groupby("news_score"):
        out[int(s)] = {
            "n": int(len(g)),
            "hit_rate": float((g["realized_return"] > 0).mean()),
            "mean_return": float(g["realized_return"].mean()),
        }
    return out


def _recent_picks_sample(df: pd.DataFrame, n: int = 5) -> list[dict]:
    df = df.sort_values("as_of", ascending=False).head(n)
    return [_format_pick(r) for _, r in df.iterrows()]


def feedback_block(window_days: int = 90, mode: str | None = None,
                   current_signature: str | None = None) -> str:
    """Markdown block summarising recent performance, designed for inclusion
    in LLM prompts so Claude can self-correct.

    If ``current_signature`` (full param set, e.g. ``claude_HOSE``) is
    provided, the by-signature table highlights the EXACT parameter
    combination history — the most apples-to-apples view."""
    perf = recent_performance(window_days=window_days, mode=mode)
    if perf.get("n", 0) == 0:
        return f"## Past performance feedback\n\n_{perf.get('note', 'no data')}_\n"

    lines = [
        f"## Past performance feedback (last {window_days} days)",
        "",
        f"- **n picks evaluated**: {perf['n']}",
        f"- **hit rate**: {perf['hit_rate']:.1%}",
        f"- **mean return**: {perf['mean_return']:+.4f}",
        f"- **median return**: {perf['median_return']:+.4f}",
        "",
    ]

    by_sig = perf.get("by_run_signature") or {}
    if by_sig:
        lines += [
            "### By full run signature (most apples-to-apples)",
            "",
            "Each row is a distinct parameter combination (mode + hose-only",
            "+ exclude). Re-runs of the same combo update the same row in the",
            "ledger. **THIS RUN** marks the exact parameters of today's prediction.",
            "",
            "| signature | n | hit_rate | mean_return | median_return | match? |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for sig in sorted(by_sig.keys()):
            s = by_sig[sig]
            match = "← **THIS RUN**" if (current_signature is not None and sig == current_signature) else ""
            lines.append(
                f"| `{sig}` | {s['n']} | {s['hit_rate']:.1%} | "
                f"{s['mean_return']:+.4f} | {s['median_return']:+.4f} | {match} |"
            )
        if current_signature is not None and current_signature not in by_sig:
            lines.append(
                f"| `{current_signature}` | 0 | _no prior history_ | - | - | ← **THIS RUN** |"
            )
        lines.append("")

    lines += [
        "### By prior news_score",
        "",
        "| news_score | n | hit_rate | mean_return |",
        "| --- | --- | --- | --- |",
    ]
    for s, stats in sorted(perf["by_news_score"].items()):
        lines.append(f"| {s:+d} | {stats['n']} | {stats['hit_rate']:.1%} | {stats['mean_return']:+.4f} |")

    lines += ["", "### Most recent evaluated picks", ""]
    lines.append("| as_of | symbol | pred N | pred P | news_score | realized |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for p in perf["recent_5"]:
        n_s = f"{p['pred_days']:.0f}d" if "pred_days" in p else "-"
        p_s = f"{p['pred_profit']:+.3f}" if "pred_profit" in p else "-"
        lines.append(
            f"| {p['as_of']} | {p['symbol']} | {n_s} | {p_s} | "
            f"{p['news_score']:+d} | {p['realized_return']:+.4f} |"
        )

    by_dim = perf.get("by_dimension") or {}
    if by_dim:
        # Sort by mean_return desc so the "what's been working" cluster lands
        # at the top — that's what Claude needs to know first when deciding
        # which dimensions to weight more heavily today. Only show tags with
        # n ≥ 2 to avoid noise from single-observation buckets.
        dims_sorted = sorted(
            ((tag, s) for tag, s in by_dim.items() if s["n"] >= 2),
            key=lambda kv: kv[1]["mean_return"],
            reverse=True,
        )
        if dims_sorted:
            lines += [
                "### By dimension category cited (which research tags actually predicted returns)",
                "",
                "Each row is a `[dimension-name]` tag Claude cited in Step 4 of",
                "a past plan. A pick that cited 3 tags contributes to all 3 rows.",
                "Only tags with n ≥ 2 are shown to filter noise. Sorted by",
                "mean_return so the dimensions that have actually worked land",
                "at the top.",
                "",
                "| dimension | n | hit_rate | mean_return | median_return |",
                "| --- | --- | --- | --- | --- |",
            ]
            for tag, s in dims_sorted:
                lines.append(
                    f"| `{tag}` | {s['n']} | {s['hit_rate']:.1%} | "
                    f"{s['mean_return']:+.4f} | {s['median_return']:+.4f} |"
                )
            lines.append("")

    fill = perf.get("limit_fill")
    if fill is not None:
        lines += [
            "",
            "### Limit-buy fill calibration (low head)",
            "",
            "When the low head was trained at run time, picks recorded an",
            "``entry_limit_price`` (= close × (1 + predicted dip)). After the",
            "buy day closes we mark each row's ``entry_limit_filled`` based",
            "on whether ``t0_low <= entry_limit_price``. This says nothing",
            "about whether the trade was profitable — just whether the limit",
            "order would have actually executed.",
            "",
            f"- **n picks with low-head limits**: {fill['n']}",
            f"- **fill_rate** (% of limits the market actually touched): "
            f"{fill['fill_rate']:.1%}",
            f"- **mean dip QUOTED** (pred_low): "
            f"{fill['mean_dip_quoted']:+.4f}",
            f"- **mean dip ACTUAL** ((t0_low − close)/close): "
            f"{fill['mean_dip_actual']:+.4f}",
            f"- **calibration** (actual − quoted, sign convention: positive "
            f"= dips were shallower than we quoted, so we missed fills): "
            f"{fill['calibration']:+.4f}",
            "",
            "Interpret: target fill_rate should track ``pricing.entry_low_alpha``",
            "(default 0.5 → ~50% of limits fill). If fill_rate is much lower",
            "than alpha, the low head is too bearish on dips and limits never",
            "trigger — raise alpha or retrain. If fill_rate is much higher,",
            "limits fill almost every day but the dip captured is tiny —",
            "lower alpha to find a deeper, cheaper entry.",
            "",
        ]

    slip = perf.get("entry_slippage")
    if slip is not None:
        lines += [
            "",
            "### Entry-execution sanity check",
            "",
            "We record the predicted `entry_price` (the close on the data",
            "bar that fed the model) and later compare it against the OHLC",
            "of the actual buy day, which IS `as_of` (`effective_today_for_trading`",
            "rolls `as_of` forward past 14:30 so it always names the day the",
            "user is supposed to trade). This measures whether the entry we",
            "quoted was actually fillable, not just whether the model got",
            "the direction right.",
            "",
            f"- **n picks with buy-day data**: {slip['n']}",
            f"- **mean entry_slippage** (t0_low vs entry_price): "
            f"{slip['mean']:+.4f}",
            f"- **median entry_slippage**: {slip['median']:+.4f}",
            f"- **% picks where entry was UNREACHABLE** "
            f"(t0_low > entry_price → realized_return is fictional): "
            f"{slip['pct_unreachable']:.1%}",
            f"- **mean savings when reachable** (how much cheaper a "
            f"limit-order at t0_low would have filled): "
            f"{slip['mean_savings_when_reachable']:.4f}",
            "",
            "Interpret: a large `pct_unreachable` means today's quoted",
            "entries are systematically too low — the market gaps past",
            "them and we never get filled. A large `mean_savings_when_",
            "reachable` means quoted entries are too high — we routinely",
            "could have bought lower. Either way, treat published",
            "`actionable=True` rows with caution if these numbers are",
            "large; the realized_return assumes we did fill at",
            "`entry_price`, which the data here may contradict.",
            "",
        ]

    lines += [
        "",
        "Use this to calibrate: weight the **THIS RUN** signature row most",
        "heavily (exact parameter match). If a particular news_score has been",
        "consistently wrong on this signature, weight it less. If recent picks",
        "at this signature have been losing, be more conservative today.",
        "",
    ]
    return "\n".join(lines)
