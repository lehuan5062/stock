"""Parquet cache for OHLCV. One file per ticker, indexed by date."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import cache_dir

_CORP_ACTION_LOG_COLUMNS = ["timestamp", "symbol", "message", "healed"]


def _corp_action_log_path() -> Path:
    return cache_dir() / "corp_action_events.csv"


def log_corp_action_event(symbol: str, message: str, healed: bool) -> None:
    """Append one row to the corp-action-guard event log so recurring or
    non-healing violations (a signal of a real bug rather than an expected
    corp-action artifact) can be spotted across runs instead of relying on
    scrollback of the console warning."""
    path = _corp_action_log_path()
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CORP_ACTION_LOG_COLUMNS)
        writer.writerow([dt.datetime.now().isoformat(timespec="seconds"), symbol, message, healed])


class SuspectedCorporateActionArtifact(Exception):
    """Raised by merge_ohlcv when a new incrementally-fetched bar's 1-day
    move against the last cached close exceeds the symbol's exchange band +
    margin -- physically impossible for a normal trading day.

    Root cause this guards against: the source (VCI/KBS) can retroactively
    rewrite a symbol's historical closes after a corporate action (stock
    dividend / rights issue) is processed server-side, but our incremental
    fetch only ever appends new dates and never re-validates already-cached
    ones. If an incremental fetch lands while a date is mid-rewrite, or the
    two sides of a dividend adjustment straddle an append boundary, the
    result is a bar that's an impossible jump/reversal against its neighbor
    (confirmed live on ABB's 2026-07 cache: an exact 1.150x checkerboard
    matching its 15% stock dividend ratio, sourced from a temporal mismatch,
    not a same-moment cross-source disagreement -- both VCI and KBS agree
    with each other on a fresh pull, disagreeing only with our stale cache).

    Signals the caller (update_symbol) to force a full re-fetch rather than
    write a row that would create or extend this kind of corruption."""


def _corp_action_threshold(symbol: str) -> float:
    """Per-symbol corporate-action threshold = that symbol's exchange price
    band + margin. Mirrors filters.py's _row_band_threshold (same band +
    _CORP_ACTION_MARGIN values) but as a scalar for one symbol instead of a
    vectorized per-row Series -- this is the write-time counterpart of that
    analysis-time check, applied here to prevent an impossible bar from ever
    being cached rather than filtering it out after the fact."""
    from ..filters import _CORP_ACTION_MARGIN, _ceiling_params
    from .universe import load_universe

    limits, _tol = _ceiling_params()
    widest = max(limits.values()) if limits else 0.15
    uni = load_universe()
    ex_map = (dict(zip(uni["symbol"].astype(str), uni["exchange"]))
              if uni is not None and len(uni) else {})
    band = limits.get(ex_map.get(symbol.upper()), widest)
    return band + _CORP_ACTION_MARGIN


def ohlcv_dir() -> Path:
    d = cache_dir() / "ohlcv"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ohlcv_path(symbol: str) -> Path:
    return ohlcv_dir() / f"{symbol.upper()}.parquet"


def read_ohlcv(symbol: str) -> pd.DataFrame:
    p = ohlcv_path(symbol)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df.sort_index()


def write_ohlcv(symbol: str, df: pd.DataFrame) -> None:
    """Persist the merged OHLCV frame for `symbol`.

    Atomic write: parquet is written to `<symbol>.parquet.tmp` then
    renamed onto the final path. This guarantees that if the process is
    killed (Ctrl+C, OS shutdown) mid-write, the on-disk file is either
    the previous complete version or the new complete version — never a
    partial / corrupt parquet that fails to read on the next run."""
    import os

    if df.empty:
        return
    out = df.copy()
    if out.index.name != "date":
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.set_index("date")
    # Normalize to calendar date: some fetches stamp midnight, others stamp
    # a mid-morning ICT time for the same trading day. Keeping both as
    # distinct index values lets a same-day bar satisfy `index > as_of`
    # (as_of normalized to midnight) in tracking.py's exit scan, producing
    # phantom same-day "recoveries" that are impossible to actually trade.
    out.index = out.index.normalize()
    # kind="stable" is essential here, not cosmetic: merge_ohlcv's whole
    # "new data overwrites stale existing data" contract (via concat +
    # duplicated(keep="last") below) depends on ties preserving their
    # ORIGINAL relative order -- existing's rows (concatenated first)
    # must sort before new's rows (concatenated second) for the same
    # date, so keep="last" picks new's fresher value. The default
    # sort_index() kind is quicksort, which is NOT stable -- ties can
    # land in either order, and duplicated(keep="last") would then pick
    # whichever happened to sort last, sometimes the STALE existing row
    # instead of the fresh new one. Confirmed live: a full re-fetch that
    # should have overwritten ABB's corrupted 06-24/06-25 rows with clean
    # VCI data left the stale corrupted values in place because quicksort
    # put the fresh rows before the stale ones for those exact dates.
    out = out.sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    target = ohlcv_path(symbol)
    tmp = target.with_suffix(target.suffix + ".tmp")
    out.reset_index().to_parquet(tmp, index=False)
    # os.replace is atomic on POSIX and Windows (since Python 3.3+).
    os.replace(tmp, target)


def _validate_no_impossible_move(symbol: str, closes: list[tuple], start_prev_close: float | None) -> None:
    """Walk ``closes`` (a sequence of (label, close) pairs, in date order)
    checking each consecutive pair's move against ``_corp_action_threshold``.
    ``start_prev_close`` seeds the running previous close (None to start from
    the first row with no prior comparison). Raises
    SuspectedCorporateActionArtifact on the first violation."""
    threshold = _corp_action_threshold(symbol)
    prev_close = start_prev_close
    for label, close in closes:
        if prev_close is not None and prev_close > 0:
            move = abs(close / prev_close - 1.0)
            if move > threshold:
                raise SuspectedCorporateActionArtifact(
                    f"{symbol} {label}: {move:.1%} move vs prior close "
                    f"{prev_close} exceeds {threshold:.1%} band+margin"
                )
        prev_close = close


def merge_ohlcv(symbol: str, new: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Append new rows into the cached parquet, dedupe on date, return merged frame.

    ``validate`` (default True) gates a consecutive-close-move check across
    the WHOLE combined series (existing's last close -> new's rows, in
    order) against ``_corp_action_threshold``. Intended for the incremental
    append path, where a single new day landing on top of trusted existing
    history really shouldn't ever jump beyond the exchange band.

    Full re-fetches (``full=True`` in update_symbol) pass ``validate=False``:
    a full refetch is EXPECTED to jump past whatever (possibly stale/
    corrupted) history was cached before, so gating on that boundary would
    risk a false rejection on the very refetch meant to heal the cache.

    IMPORTANT LIMITATION (deliberately accepted, not fixed): with
    validate=False, a bad row from a flaky source sandwiched between good
    neighbors WITHIN THE SAME fetch batch is NOT caught here (confirmed
    live: MSN briefly returned ABB's close as ~1039 for one date while
    VCI/KBS both agreed on ~15.8 for the same date). An earlier attempt to
    ALSO validate internal batch consistency during full=True broke far
    worse: it rejects ANY symbol whose real multi-year history contains a
    genuine, permanent corporate-action jump (extremely common for VN
    stocks), since a bare consecutive-move check can't distinguish "real
    permanent level change" from "phantom row that will revert" without a
    reversal-lookahead window — see scripts/repair_corporate_action_corruption.py
    for that reversal-based distinction, applied there instead of here.
    Mitigation for now: prefer a single trusted source (not a multi-source
    fallback chain) for full re-fetches when healing a specific symbol.
    """
    existing = read_ohlcv(symbol)
    if new.empty:
        # Nothing to add (e.g. a thin ticker with no matched trades in the
        # fetched window) — skip the write and the concat entirely. concat
        # with an empty/all-NA frame is a no-op here but triggers pandas'
        # FutureWarning on every call, which spams the console across a
        # large batch with many quiet tickers.
        return existing

    if validate and not existing.empty:
        new_sorted = new.sort_index()
        # Only validate rows STRICTLY AFTER the cached max. Sources sometimes
        # ignore the requested start date and return history overlapping (or
        # entirely before) the cache — e.g. delisted/suspended tickers where
        # the API replays old bars. Seeding the check with existing's LAST
        # close and walking from new's FIRST row would then compare
        # non-adjacent dates (a 2024 close vs a 2022 close) and flag a bogus
        # "impossible move" every single run. Overlapping rows are already
        # covered by the concat + keep="last" dedup and need no boundary check.
        cached_max = existing.index.max()
        idx_norm = pd.DatetimeIndex(new_sorted.index).normalize()
        after = new_sorted[idx_norm > cached_max]
        if not after.empty:
            closes = [(ts.date() if hasattr(ts, "date") else ts, float(c))
                      for ts, c in after["close"].items()]
            boundary_prev = float(existing["close"].iloc[-1])
            _validate_no_impossible_move(symbol, closes, boundary_prev)

    if existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new])
    write_ohlcv(symbol, merged)
    return read_ohlcv(symbol)


def cached_symbols() -> list[str]:
    return sorted(p.stem for p in ohlcv_dir().glob("*.parquet"))


# ---- fetch watermarks --------------------------------------------------
#
# We persist "we attempted to fetch <symbol> through date D" so that
# permanently-stuck tickers (delisted, halted, absent from the feed) don't
# trigger a wasted fetch on every run. Each attempt — successful or not —
# stamps the watermark to `latest_expected_bar_date()` for that ticker.
# When the expected bar advances (next trading day closes), the watermark
# falls behind and the ticker gets one fresh attempt.
#
# Storage: cache/watermarks/{SYMBOL}.txt — a single ISO date string per
# file. Tiny (~10 bytes × ~1,500 = ~15 KB total), atomic per-symbol writes,
# thread-safe because each thread writes its own file.

import datetime as _dt


def _watermark_dir() -> Path:
    d = cache_dir() / "watermarks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _watermark_path(symbol: str) -> Path:
    return _watermark_dir() / f"{symbol.upper()}.txt"


def get_watermark(symbol: str) -> _dt.date | None:
    """Return the date through which we've attempted to fetch this symbol,
    or None if no attempt has ever been recorded."""
    p = _watermark_path(symbol)
    if not p.exists():
        return None
    try:
        return _dt.date.fromisoformat(p.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def set_watermark(symbol: str, attempted_through: _dt.date) -> None:
    """Stamp the latest-attempted-through date for this symbol. Atomic
    write so a Ctrl+C can't leave the watermark file half-written."""
    import os

    p = _watermark_path(symbol)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(attempted_through.isoformat(), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        # Persisted watermarks are best-effort; if disk is full we still
        # want the run to succeed rather than crashing.
        pass
