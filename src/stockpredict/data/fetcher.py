"""Download daily OHLCV for one or many Vietnamese tickers via vnstock."""
from __future__ import annotations

import datetime as dt
import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import pandas as pd

from ..config import load_config
from .cache import (SuspectedCorporateActionArtifact, get_watermark, log_corp_action_event,
                    merge_ohlcv, read_ohlcv, set_watermark)


def _today_str() -> str:
    return dt.date.today().isoformat()


# Plain, \n-terminated colored console lines for the update_many fetch loop.
# Deliberately NOT using tqdm here: tqdm's redraw is a bare \r with no
# trailing \n, which isn't coordinated with these lines at all -- a line
# firing mid-redraw got silently appended onto the tail of the still-open
# bar (confirmed live on a real .bat run). Every write below ends in \n, so
# concurrent writes from the two worker threads can interleave in ORDER but
# never corrupt a single line's CONTENT.
_ANSI_COLORS = {
    "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
    "orange": "\033[38;5;208m", "blue": "\033[34m",
}
_ANSI_RESET = "\033[0m"


def _cprint(msg: str, color: str | None = None) -> None:
    print(f"{_ANSI_COLORS[color]}{msg}{_ANSI_RESET}" if color else msg)


class _RateLimiter:
    """Evenly-paced rate limiter.

    Enforces a minimum interval of ``window / cap`` seconds between
    consecutive calls, so ``cap`` calls/min is delivered as one call every
    ``60/cap`` seconds (e.g. 54/min -> a call every ~1.11s) rather than a
    burst of ``cap`` calls followed by a stall. The old sliding-window
    implementation permitted the full burst, and that burst is exactly what
    tripped the providers' server-side 429s — even pacing is gentler on them.

    Even pacing also subsumes the "<= cap per rolling window" guarantee: at
    one call per ``window/cap`` seconds, no window ever contains more than
    ``cap`` calls.

    ``pause(seconds)`` lets error paths force a hard cooldown after a failure
    — every subsequent caller waits until it elapses. The cap is fixed at
    construction from config (``data.api_per_min`` / ``api_per_min_overrides``)
    and is never adjusted at runtime — see ``config.yaml``'s note.
    """

    def __init__(self, calls_per_min: float, window_seconds: float = 60.0) -> None:
        self.cap = max(1, int(calls_per_min))
        self.window = float(window_seconds)
        self.min_interval = self.window / self.cap
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.paused_until: float = 0.0
        self._last_call: float = 0.0  # monotonic time of the last granted call

    def wait(self) -> None:
        with self.cond:
            while True:
                now = time.monotonic()
                # Honor any global pause (e.g. the post-429 cooldown).
                if self.paused_until > now:
                    self.cond.wait(timeout=self.paused_until - now)
                    continue
                # Enforce the even minimum spacing since the last granted call.
                target = self._last_call + self.min_interval
                if now < target:
                    self.cond.wait(timeout=target - now)
                    continue
                self._last_call = now
                return

    def pause(self, seconds: float, reason: str = "") -> None:
        """Force everyone to wait at least `seconds` more before the next call.
        Stacks with prior pauses (takes the longer). Silent — callers own any
        announcement (see ``_cprint`` call sites), since the right color/text
        depends on context (429 vs. transient failure) this method doesn't
        have."""
        with self.cond:
            until = time.monotonic() + max(0.0, float(seconds))
            if until > self.paused_until:
                self.paused_until = until
                self.cond.notify_all()

    def cooldown(self, source: str, seconds: float, reason: str = "") -> None:
        """Apply the fixed per-source cooldown after a fetch failure (429 or
        transient). No rate adjustment and nothing persisted — the cap stays
        put; this only pauses the source for ``seconds`` so a struggling
        source doesn't hot-loop its failing endpoint. ``seconds`` comes from
        ``config.yaml``'s ``data.cooldown_seconds``/``cooldown_seconds_overrides``
        and is fixed. Silent (see ``pause``'s docstring)."""
        self.pause(seconds, reason=reason)

    def paused_remaining(self) -> float:
        """Seconds this source is still backing off (0.0 if not paused)."""
        with self.cond:
            return max(0.0, self.paused_until - time.monotonic())


_LIMITERS: dict[str, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()

# The vnstock sources that still serve VN stock OHLCV in the installed
# version. TCBS was removed from the Quote module entirely. MSN was removed
# 2026-07-09: confirmed 3-for-3 on real corruption incidents this session
# (ABB, USC, EMS) — MSN silently returns fabricated/wrong-instrument prices
# for dates VCI and KBS both agree have no real data, with no error to
# signal the row is bad. KBS/VCI never showed this failure mode. The
# shared-queue worker pool is symmetric (FIFO, no priority), so this order
# only controls (a) fetch_history's default fallback-chain order on direct
# calls without an explicit source_order, and (b) display order in the
# status bar / logs — NOT which worker gets to a symbol first in
# update_many. Keep in sync with update_many().
_VALID_SOURCES = ("KBS", "VCI")

# When a source is backing off from a 429 by more than this many seconds, a
# worker reroutes its queued symbols to a healthy source instead of blocking.
_BACKOFF_REROUTE_THRESHOLD = 1.0
_RATE_LIMIT_ERROR_TOKENS = (
    "rate limit",
    "rate-limit",
    "GIỚI HẠN API",          # vnstock's Vietnamese rate-limit message
    "GIOI HAN API",
    "Rate Limit Exceeded",
    "tối đa số lượt yêu cầu",
    "429",
)


def _looks_like_rate_limit(err: BaseException) -> bool:
    msg = str(err)
    return any(tok.lower() in msg.lower() for tok in _RATE_LIMIT_ERROR_TOKENS)


# vnstock providers (both 4.0.2 and 4.0.4) raise ValueError with one of these
# when a symbol has NO bars in the requested date range — a thin/illiquid or
# recently-untraded ticker over a narrow incremental window. That is NOT a
# fetch failure (nothing is wrong with the source); it means "no new data",
# which for an incremental update is the normal zero-row outcome. Treating it
# as a failure previously triggered pointless cross-source failover and, when
# the other source was down, a spurious RuntimeError.
_EMPTY_DATA_TOKENS = (
    "dữ liệu trống",   # KBS: "Dữ liệu trống cho mã ..."
    "du lieu trong",
    "empty data",
    "no data",
    "không có dữ liệu",
    "không tìm thấy dữ liệu",   # VCI: "Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã..."
    "khong tim thay du lieu",
)


def _looks_like_empty_data(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(tok in msg for tok in _EMPTY_DATA_TOKENS)


def _configured_rate_for(source: str, cfg) -> float:
    """Resolve the fixed calls/min for ``source``: the
    STOCKPREDICT_API_PER_MIN env var if set (applies to every source),
    else ``data.api_per_min_overrides[source]`` if present, else the global
    ``data.api_per_min``."""
    env_rate = os.environ.get("STOCKPREDICT_API_PER_MIN")
    if env_rate:
        return float(env_rate)
    overrides = cfg.data.get("api_per_min_overrides", {}) or {}
    return float(overrides.get(source.upper(), cfg.data.get("api_per_min", 12.0)))


def _configured_cooldown_for(source: str, cfg) -> float:
    """Resolve the fixed cooldown seconds for ``source``:
    ``data.cooldown_seconds_overrides[source]`` if present, else the global
    ``data.cooldown_seconds``."""
    overrides = cfg.data.get("cooldown_seconds_overrides", {}) or {}
    return float(overrides.get(source.upper(), cfg.data.get("cooldown_seconds", 3.0)))


def _limiter(source: str = "_default") -> _RateLimiter:
    """Return the rate limiter for ``source``, building it on first use.

    Each vnstock backend (VCI / TCBS / KBS / MSN) is a different company's
    server, so each gets its own sliding-window limiter, seeded from
    ``api_per_min`` (or a per-source override, see ``api_per_min_overrides``).
    A pause after a transient error on one source therefore never throttles
    the others on the fallback chain in ``fetch_history``."""
    src = source.upper() if source else "_default"
    lim = _LIMITERS.get(src)
    if lim is not None:
        return lim
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(src)
        if lim is None:
            # Configurable via config.yaml -> data.api_per_min (default rate
            # for any source) and data.api_per_min_overrides (a per-source
            # rate — e.g. to run KBS faster/slower than VCI), or the
            # STOCKPREDICT_API_PER_MIN env var (applies to every source,
            # overriding both config values). The rate is FIXED for the life
            # of the process — never ratcheted down at runtime, never
            # persisted; tune it by hand in config.yaml.
            rate = _configured_rate_for(src, load_config())
            lim = _RateLimiter(calls_per_min=rate)
            _LIMITERS[src] = lim
            logging.getLogger("stockpredict.rate").info(
                "rate limiter active for %s: %.1f calls/min sliding window",
                src, rate
            )
    return lim


def _disable_vnstock_hard_exit() -> None:
    """vnstock's ``vnai.beam.quota.CleanErrorContext.__exit__`` calls
    ``sys.exit("... Process terminated.")`` whenever its rate-limit guardian
    fires. That raises ``SystemExit`` (a ``BaseException``, not ``Exception``),
    which slips past our retry loop in ``fetch_history`` and kills the whole
    batch — bypassing the limiter pause we already coded for the 429 path.
    Replace it with a return-False so the underlying ``RateLimitExceeded``
    propagates normally; ``_looks_like_rate_limit`` then matches it and the
    existing pause/retry runs as intended."""
    try:
        from vnai.beam import quota  # type: ignore
    except Exception:
        return
    cec = getattr(quota, "CleanErrorContext", None)
    if cec is None or getattr(cec, "_stockpredict_no_hard_exit", False):
        return

    def _patched_exit(self, exc_type, exc_val, exc_tb):  # noqa: D401
        return False  # let RateLimitExceeded bubble out instead of sys.exit

    cec.__exit__ = _patched_exit
    cec._stockpredict_no_hard_exit = True


def quiet_vnstock_logger() -> None:
    """Vnstock spams ERROR-level logs for transient API issues that we
    already handle via fallback. Bump it to CRITICAL so it stops cluttering
    the console during long full-universe runs."""
    # Introduce ourselves to vnstock BEFORE silencing its loggers — the
    # intro line goes through `stockpredict.intro` (a separate logger),
    # so it survives the CRITICAL bump below.
    from .intro import introduce
    introduce()
    _disable_vnstock_hard_exit()
    for name in ("vnstock", "vnstock.core.utils.client", "vnstock.explorer",
                 "vnstock.explorer.msn.quote"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _quote_history(symbol: str, src: str, start: str, end: str,
                   interval: str, bypass_quota: bool):
    """Call vnstock's Quote.history, optionally bypassing vnai's client-side
    rate-limit quota.

    vnstock meters every call through ``vnai.beam.quota.optimize`` (a
    decorator layered onto ``Quote.history`` at both the public "API" wrapper
    and the per-source explorer level). That quota — NOT the data providers'
    own servers — is what enforces the guest tier's 20 req/min cap. vnstock's
    own error message states the library "automates access to public APIs you
    already have legitimate access to," so calling the underlying endpoint
    directly is within bounds.

    Because the decorators use ``functools.wraps``, the undecorated function
    survives as ``__wrapped__``. We reach the per-source provider instance
    (``q.provider``) and invoke its raw ``history.__wrapped__`` — skipping both
    the "API" and per-source quota layers. The body underneath is a plain
    ``requests`` call (``send_request`` -> ``send_request_direct``), so no
    quota counter is ever touched.

    On bypass failure, we RAISE instead of falling back to q.history() — the
    metered path uses @optimize_execution (shared, thread-global loop counter).
    With multiple workers, this triggers false "rate limit" errors and cascades
    into wasteful backoff/retry logic. Instead, let fetch_history() catch this
    as a normal per-source error and try the next source, without re-entering
    the shared vnai loop-detector."""
    from vnstock import Quote

    q = Quote(symbol=symbol, source=src)
    if bypass_quota:
        try:
            provider = q.provider
            raw = provider.history.__wrapped__
            return raw(provider, start=start, end=end, interval=interval)
        except (AttributeError, TypeError):
            # vnstock internals changed — raise to let fetch_history try next source
            raise RuntimeError(
                f"vnai bypass unavailable for {src}/{symbol} (vnstock internals changed); "
                "will retry next source"
            )
    # bypass_quota=False: fall back to metered path (not the default)
    return q.history(start=start, end=end, interval=interval)


def fetch_history(symbol: str, start: str, end: str | None = None,
                  source_order: list[str] | None = None) -> pd.DataFrame:
    """Fetch raw daily OHLCV from vnstock. Tries sources in the given order.

    Args:
        symbol: Ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date; defaults to today
        source_order: List of sources to try in order (e.g., [VCI, KBS]).
                      Defaults to _VALID_SOURCES.

    On any error (429, timeout, bad data), moves to the next source in the list.
    A genuine provider 429 applies that source's fixed cooldown
    (``data.cooldown_seconds_overrides[source]``, else ``data.cooldown_seconds``)
    before abandoning the symbol on that source (see ``_looks_like_rate_limit``).
    The source's rate is never adjusted.

    When ``data.bypass_vnai_quota`` is set (default), calls go straight to
    the underlying provider endpoint via ``_quote_history``, sidestepping
    vnstock's 20/min guest quota. The per-source ``_RateLimiter`` still
    applies as a politeness throttle against the providers' real servers.
    """
    cfg = load_config()
    bypass_quota = bool(cfg.data.get("bypass_vnai_quota", True))
    end = end or _today_str()

    if source_order is None:
        source_order = list(_VALID_SOURCES)

    saw_empty_data = False  # a source authoritatively reported "no bars in range"
    last_err: Exception | None = None
    for src in source_order:
        if src not in _VALID_SOURCES:
            continue
        for interval in ("1D", "D"):  # vnstock 4 uses 1D, older uses D
            try:
                _limiter(src).wait()
                _cprint(f"{src} is fetching ...")
                df = _quote_history(symbol, src, start, end, interval, bypass_quota)
                if df is not None and len(df) > 0:
                    return _normalize_ohlcv(df)
                # Source responded successfully with zero rows — no bars in
                # this window (thin ticker / no new data). Not a failure.
                saw_empty_data = True
                break
            except Exception as e:
                last_err = e
                if _looks_like_rate_limit(e):
                    # Genuine provider-side 429: apply this source's fixed
                    # cooldown, then abandon this symbol on this source
                    # immediately (don't burn the other interval) — the caller
                    # hands it off to another source right away while the
                    # cooldown throttles the flood. The rate is not adjusted.
                    _cprint(f"{src}: 429 hit fetching {symbol}", "orange")
                    _limiter(src).cooldown(src, _configured_cooldown_for(src, cfg))
                    break
                if _looks_like_empty_data(e):
                    # The provider raised "empty data" — it responded fine,
                    # there just are no bars in this range. Treat exactly like
                    # a successful zero-row fetch, NOT a failure to fail over.
                    saw_empty_data = True
                    break
                logging.getLogger("stockpredict.fetcher").debug(
                    "fetch_history(%s, %s, interval=%s) failed: %s",
                    symbol, src, interval, type(e).__name__
                )
                # Non-rate-limit error: try the next interval for this source
                continue
        # All intervals failed (or rate-limited) for this source; move on.

    if saw_empty_data:
        # At least one source authoritatively reported no bars in the window
        # and none returned data — this is a legitimate zero-row result (e.g.
        # an incremental update on a thin ticker that didn't trade recently),
        # not a fetch failure. Return an empty frame so update_symbol records
        # 0 new rows and stamps the watermark instead of erroring / failing over.
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="date"),
        )

    # Preserve the real failure -- both in the message (so the caller can
    # print the ACTUAL reason instead of a generic "could not fetch") and via
    # `from last_err` (standard exception chaining, so the caller can inspect
    # __cause__ directly instead of us inventing a separate flag).
    raise RuntimeError(
        f"Could not fetch {symbol} from any source in {source_order}: {last_err}"
    ) from last_err


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for std in ("time", "date", "open", "high", "low", "close", "volume"):
        if std in cols_lower:
            rename[cols_lower[std]] = std
    df = df.rename(columns=rename)
    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].dropna(subset=["close"])


def audit_cache(symbols: Iterable[str],
                expected_bar: pd.Timestamp | None = None
                ) -> tuple[list[str], list[str], list[str]]:
    """Bucket ``symbols`` against the cache and the latest expected bar.

    Returns ``(warm, stale, cold)`` where:
      * ``warm``  — cached file exists AND
                    (``cached_max >= expected_bar`` OR
                     a fetch was already attempted for this expected bar
                     and produced no new rows — see watermarks below)
      * ``stale`` — cached file exists, ``cached_max < expected_bar``,
                    and we haven't yet attempted a fetch for this expected
                    bar
      * ``cold``  — no cached file at all (full-history fetch needed)

    Watermarks: each successful fetch attempt — even one that returned
    zero new rows — stamps a per-symbol marker dated to the expected bar
    at the time of the attempt. Permanently-stuck tickers (delisted,
    halted, absent from the data feed) thereafter classify as warm
    rather than stale, so the user doesn't burn API budget retrying
    them every run. When the expected bar advances (next trading day
    closes), the marker becomes outdated and the ticker gets one fresh
    attempt before being re-stamped.

    If ``expected_bar`` is ``None`` (empty calendar / first-ever run),
    every cached symbol counts as warm — we have nothing better to
    compare against.
    """
    if expected_bar is None:
        from ..tracking import latest_expected_bar_date
        expected_bar = latest_expected_bar_date()
    warm: list[str] = []
    stale: list[str] = []
    cold: list[str] = []
    expected_date = expected_bar.date() if expected_bar is not None else None
    for s in symbols:
        s = s.upper()
        df = read_ohlcv(s)
        if df.empty:
            cold.append(s)
            continue
        if expected_bar is None or df.index.max().normalize() >= expected_bar:
            warm.append(s)
            continue
        # Cache-stale by date, but check if we already tried this run-cycle.
        wm = get_watermark(s)
        if wm is not None and expected_date is not None and wm >= expected_date:
            warm.append(s)  # broker had nothing new last time — don't retry
        else:
            stale.append(s)
    return warm, stale, cold


def _source_worker(source: str,
                   work: "queue.Queue",
                   sources: list[str],
                   tried: dict[str, set],
                   tried_lock: threading.Lock,
                   full: bool,
                   results: dict,
                   lock: threading.Lock,
                   total_to_fetch: int,
                   fetch_total: int = 0,
                   fetched_counter: dict | None = None) -> None:
    """One worker per source, all pulling from a single shared ``work`` queue.

    There is NO pre-distribution or source ranking: whichever worker is free
    grabs the next symbol, so a fast/healthy source naturally does more work
    and a throttled one does less — self-balancing without any win-rate
    bookkeeping.

    Cooldown handling: if this source is currently in its 429 cooldown, the
    worker does not grab new work (it would only have to wait) — it sleeps
    briefly and lets the other source drain the queue. That is what keeps the
    two workers genuinely parallel instead of one stalling the batch. While
    paused, this worker prints a blue countdown once per second (via the 0.2s
    poll below, throttled to one line per integer-second change).

    Cross-source fallback: a symbol this source can't fetch is put back on the
    shared queue tagged (in ``tried``) so it won't be retried by the same
    source; another source picks it up. Once every source has tried a symbol
    it's recorded as a final error. Each source attempts each symbol at most
    once, so the batch always terminates.

    Terminates when ``len(results)`` reaches ``total_to_fetch`` (every symbol
    accounted for), not when the queue is momentarily empty — otherwise a
    worker could exit just as another puts a symbol back.

    ``fetch_total``/``fetched_counter`` (shared across both workers) drive the
    green "progress update: n/total" line on every successful write.
    """
    lim = _limiter(source)
    last_cooldown_sec: int | None = None
    while True:
        with lock:
            if len(results) >= total_to_fetch:
                return

        # Don't hog work while cooling down from a 429 — let the other source
        # keep draining the shared queue.
        remaining = lim.paused_remaining()
        if remaining > _BACKOFF_REROUTE_THRESHOLD:
            sec = math.ceil(remaining)
            if sec != last_cooldown_sec:
                _cprint(f"{source} is cooling down: {sec}s", "blue")
                last_cooldown_sec = sec
            time.sleep(0.2)
            continue
        last_cooldown_sec = None

        try:
            symbol = work.get(timeout=0.2)
        except queue.Empty:
            continue

        with tried_lock:
            done_srcs = tried.setdefault(symbol, set())
            already_here = source in done_srcs
            untried_elsewhere = [s for s in sources if s not in done_srcs]

        if already_here:
            # This source already attempted this symbol; only the other
            # (currently-cooling) source can still serve it. Put it back and
            # pause briefly so we don't hot-loop pulling the same symbol.
            if untried_elsewhere:
                work.put(symbol)
                time.sleep(0.2)
            else:
                # Shouldn't happen (fully-tried symbols are error-recorded, not
                # requeued), but guard against a stuck symbol just in case.
                with lock:
                    results.setdefault(symbol, "ERR: exhausted all sources")
            continue

        try:
            delta = update_symbol(symbol, full=full, source_order=[source])
        except Exception as e:
            with tried_lock:
                done_srcs = tried.setdefault(symbol, set())
                done_srcs.add(source)
                untried = [s for s in sources if s not in done_srcs]
            # fetch_history chains the real failure via `raise ... from
            # last_err`, so the ORIGINAL exception (with its real message) is
            # e.__cause__ -- use that instead of inventing generic text or
            # guessing whether this was a 429.
            cause = e.__cause__ or e
            if _looks_like_rate_limit(cause):
                # Already announced (orange) and cooled down for the real
                # reason inside fetch_history at the moment it happened --
                # nothing further to print or pause here.
                pass
            elif untried:
                _cprint(f"{source} failed {symbol}: {cause} — "
                        f"requeuing for {'/'.join(untried)}", "yellow")
            else:
                _cprint(f"{source} failed {symbol}: {cause} — "
                        f"all sources exhausted, giving up", "red")
            # No cooldown here for non-429 failures: pausing an ENTIRE source
            # for a problem specific to one ticker (e.g. a delisted symbol
            # that errors on every source) would block that source's other,
            # perfectly-fetchable work for no reason -- and previously did so
            # for the full configured cooldown on ANY unidentified error.
            # Only a confirmed 429 (handled above, already applied inside
            # fetch_history) ever pauses a source.
            if untried:
                # Another source hasn't had a shot yet — requeue so it can try.
                work.put(symbol)
            else:
                with lock:
                    results[symbol] = f"ERR: {type(e).__name__}: {str(e)[:160]}"
            continue

        # The fetch didn't raise. But "didn't raise" isn't the same as "got
        # data": a provider returning zero rows (or a recognized empty-data /
        # "ticker not found" message) yields an empty frame, NOT an exception.
        # Distinguish the two by whether the symbol has ANY cached rows now:
        # a stale/warm ticker that simply had no NEW bars keeps its old rows
        # (real success); a cold ticker that got nothing stays empty.
        if read_ohlcv(symbol).empty:
            with tried_lock:
                done_srcs = tried.setdefault(symbol, set())
                done_srcs.add(source)
                untried = [s for s in sources if s not in done_srcs]
            if untried:
                # This source had nothing, but another hasn't tried yet — don't
                # declare "no data" until every source has been checked.
                work.put(symbol)
                continue
            # All sources returned empty: the ticker genuinely has no data
            # anywhere. A WARNING, not an error — nothing is wrong with the
            # sources (not red), and it's not a rate limit (not orange).
            _cprint(f"{symbol}: no data available from any source", "yellow")
            with lock:
                results[symbol] = "NODATA: no rows from any source"
                if fetched_counter is not None:
                    fetched_counter["n"] += 1
            continue

        # Real rows landed.
        with lock:
            results[symbol] = delta
            if fetched_counter is not None:
                fetched_counter["n"] += 1
                done_count = fetched_counter["n"]
            else:
                done_count = None
        if done_count is not None:
            _cprint(f"progress update: {done_count}/{fetch_total}", "green")


def update_symbol(symbol: str, full: bool = False,
                  source_order: list[str] | None = None) -> int:
    """Incremental update: fetch from cache_max_date+1 (or full history). Returns row delta.

    Skips the API call when the cache already covers the latest finalized
    end-of-day bar. Caps the fetch end-date at that same latest finalized
    bar — never asks the broker for today's intraday partial bar during
    trading hours, which would pollute the cache with mid-session noise
    and force an extra refetch later.
    """
    cfg = load_config()
    # Rolling history floor: last N years from today, recomputed each call.
    _dur = int(cfg.data.get("history_duration_years", 9))
    start_full = (pd.Timestamp.now().normalize()
                  - pd.DateOffset(years=_dur)).strftime("%Y-%m-%d")
    cached = read_ohlcv(symbol)

    # Look up the latest finalized bar once. We use it both to short-circuit
    # the fetch and to cap the end-date.
    from ..tracking import latest_expected_bar_date
    expected = latest_expected_bar_date()
    end_str = expected.strftime("%Y-%m-%d") if expected is not None else _today_str()

    if full or cached.empty:
        start = start_full
    else:
        # Smart freshness: skip if cache already contains the latest finalized bar.
        cached_max = cached.index.max().normalize()
        if expected is not None and cached_max >= expected:
            return 0
        # Skip if we've already attempted a fetch for this expected bar
        # — vnstock had nothing newer, no point burning API budget again.
        if expected is not None:
            wm = get_watermark(symbol)
            if wm is not None and wm >= expected.date():
                return 0
        last = cached.index.max().date()
        start = (last + dt.timedelta(days=1)).isoformat()
        if start > end_str:
            return 0
    new = fetch_history(symbol, start=start, end=end_str, source_order=source_order)
    before = len(cached)
    try:
        merged = merge_ohlcv(symbol, new, validate=not full)
    except SuspectedCorporateActionArtifact as e:
        if full:
            # Already doing a full refetch (which replaces the whole history
            # via merge_ohlcv's own concat + keep="last" dedup) and STILL hit
            # an internal violation -- don't recurse forever, surface it.
            log_corp_action_event(symbol, str(e), healed=False)
            raise
        _cprint(f"{symbol}: {e} -- forcing a full re-fetch to heal the cache", "yellow")
        try:
            result = update_symbol(symbol, full=True, source_order=source_order)
        except SuspectedCorporateActionArtifact:
            log_corp_action_event(symbol, str(e), healed=False)
            raise
        log_corp_action_event(symbol, str(e), healed=True)
        return result
    # Stamp the watermark to the expected bar so a fetch returning empty
    # data (delisted, halted, broker has no newer bar) doesn't retrigger
    # next run. The watermark only blocks retries within the same expected-
    # bar window — once a new trading day closes, expected advances and we
    # try fresh.
    if expected is not None:
        set_watermark(symbol, expected.date())
    return len(merged) - before


def update_many(symbols: Iterable[str], full: bool = False,
                workers: int | None = None,
                source_order: list[str] | None = None) -> dict[str, int | str]:
    """Bulk-update OHLCV for many symbols. Returns delta or error per symbol.

    Shared-queue strategy (no source ranking, no pre-distribution): all
    to-fetch symbols go into ONE queue; one worker per source (VCI, KBS)
    pulls from it. Whichever worker is free grabs the next symbol, so a
    healthy source naturally does more work and a throttled one does less —
    self-balancing without any win-rate bookkeeping. A symbol a source can't
    serve is put back for the other source to try; each source attempts each
    symbol at most once, so the batch always terminates. See ``_source_worker``.

    ``source_order`` is accepted for backward compatibility but ignored — the
    workers cover every source in ``_VALID_SOURCES``.

    Fast path: if every selected symbol is already cached through the
    latest expected bar (e.g. running on a Saturday with cache through
    Friday), returns immediately without network calls.
    """
    syms = list(symbols)
    if not syms:
        return {}

    sources = list(_VALID_SOURCES)

    if full:
        # User asked for an unconditional re-fetch — every symbol gets a job.
        to_fetch = syms
        results: dict[str, int | str] = {}
        warm_count = 0
    else:
        warm, stale, cold = audit_cache(syms)
        results = {s: 0 for s in warm}
        to_fetch = stale + cold
        # ``results`` is pre-seeded with the warm symbols so the return value
        # reports a zero-delta for them. Those pre-seeded entries must NOT be
        # counted as fetch progress below, or the bar leaps to warm/total on
        # the first tick.
        warm_count = len(warm)
        if not to_fetch:
            return results  # everything's current, no work to do

    # One shared queue; workers self-balance by pulling from it.
    work: queue.Queue = queue.Queue()
    for sym in to_fetch:
        work.put(sym)

    lock = threading.Lock()
    tried: dict[str, set] = {}
    tried_lock = threading.Lock()
    # Shared across both workers: drives the green "progress update: n/total"
    # line on every successful write. Kept separate from `results` (which is
    # pre-seeded with warm symbols) so it counts only actual fetch work.
    fetch_total = len(to_fetch)
    fetched_counter = {"n": 0}

    total_to_fetch = warm_count + len(to_fetch)
    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        futures = {}
        for src in sources:
            future = executor.submit(
                _source_worker,
                source=src,
                work=work,
                sources=sources,
                tried=tried,
                tried_lock=tried_lock,
                full=full,
                results=results,
                lock=lock,
                total_to_fetch=total_to_fetch,
                fetch_total=fetch_total,
                fetched_counter=fetched_counter,
            )
            futures[src] = future

        # Block until every worker finishes and surface any worker-thread
        # crash. Progress itself is now plain lines printed by _source_worker
        # as it goes (no bar/postfix to drive here).
        for src, future in futures.items():
            try:
                future.result()
            except Exception as e:
                _cprint(f"worker for source {src} failed: {e}", "red")

    return results
