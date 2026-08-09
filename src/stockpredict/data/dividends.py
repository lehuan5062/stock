"""Deterministic dividend-history fetcher for the dividend strategy.

Same vnai-quota-bypass technique as ``fetcher.py:fetch_history`` — reach the
per-source provider instance and call its raw ``__wrapped__`` endpoint,
skipping the metered ``@optimize_execution`` decorator. Unlike OHLCV, neither
KBS's nor VCI's installed ``Company`` explorer classes actually implement a
``dividends()`` method (the generic ``vnstock.Company.dividends()`` wrapper
just delegates to a per-source method that doesn't exist for KBS/VCI — it
raises ``AttributeError`` if called). The one real, working dividend-history
endpoint found on inspection is **VCI's company events feed**
(``Company(symbol, source="VCI").events()``), which returns ALL corporate
events (dividends, issuances, AGMs, insider deals, ...) tagged by
``event_code``; dividend-cash events carry ``event_code == "DIV"`` with
``record_date`` / ``exright_date`` / ``payout_date`` and the per-share amount
in ``value_per_share`` (absolute VND) / ``exercise_ratio`` (fraction of the
10,000 VND par value). KBS exposes an analogous ``events(event_type=2)``
"Trả cổ tức" filter on its own ``Company``, but it returned empty for every
symbol tried during implementation (looks unpopulated / dead on the current
KBS endpoint) — so VCI is the sole live source today. If KBS's dividend
events ever come back populated, add it to ``_DIVIDEND_SOURCES`` below; the
per-source rate limiter and vnai bypass already generalize to it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..config import cache_dir, load_config
from .fetcher import _limiter, _looks_like_rate_limit, _cprint

_DIVIDEND_SOURCES = ("VCI",)


def dividends_cache_dir() -> Path:
    d = cache_dir() / "dividends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dividend_cache_path(symbol: str) -> Path:
    return dividends_cache_dir() / f"{symbol.upper()}.parquet"


def _events_history(symbol: str, source: str, bypass_quota: bool) -> pd.DataFrame:
    """Call the source's raw (unmetered) ``Company.events()``, same bypass
    pattern as ``fetcher._quote_history``: reach the explorer's own
    ``Company`` class (NOT the generic ``vnstock.Company`` wrapper, which
    only forwards to a per-source method that may not exist) and invoke the
    undecorated function via ``__wrapped__``."""
    if source == "VCI":
        from vnstock.explorer.vci.company import Company as _VCICompany
        c = _VCICompany(symbol=symbol)
        if bypass_quota:
            try:
                raw = c.events.__wrapped__
                return raw(c)
            except (AttributeError, TypeError):
                raise RuntimeError(
                    f"vnai bypass unavailable for VCI/{symbol} dividends "
                    "(vnstock internals changed); will retry next source")
        return c.events()
    raise ValueError(f"no dividend-events implementation for source={source!r}")


# ---------------------------------------------------------------------------
# Canonical on-disk schema
#
# One row per corporate payout/issuance event. ``kind`` splits them by who
# receives what:
#   cash      — a cash dividend; ``cash_per_share_vnd`` is the amount.
#   stock     — new shares to EXISTING holders for free (stock dividend or bonus
#               issue); ``ratio`` is the exercise ratio. This is the "paid me in
#               shares instead of cash" signal.
#   rights    — new shares offered to existing holders at a price; takes cash
#               OUT of the holder.
#   placement — new shares to third parties or staff (private placement, ESOP).
#               Dilutive, but NOT a distribution to the holder, so it is counted
#               separately and never summed into the stock-dividend ratio.
# Only ``cash`` rows feed the yield; the rest exist so the dividend mode gets a
# deterministic dilution signal instead of asking the agent to web-search it.
#
# NOTE legacy cache files (see ``_normalize_history``) predate the placement
# split and lump stock dividends, bonus issues AND private placements together
# as ``stock``. Their stock-dividend ratio is therefore an upper bound. Only a
# refetch produces the finer classification.
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "kind", "ex_date", "record_date", "pay_date", "announce_date",
    "cash_per_share_vnd", "ratio", "subscription_vnd",
    "title", "source", "parsed_from", "ex_date_estimated",
]

_DATE_COLUMNS = ("ex_date", "record_date", "pay_date", "announce_date")
_NUM_COLUMNS = ("cash_per_share_vnd", "ratio", "subscription_vnd")

# Par value of a Vietnamese listed share. Used as the fallback subscription
# price for a rights issue whose price field the feed doesn't carry.
_PAR_VALUE_VND = 10_000.0


def empty_history() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="bool") if c == "ex_date_estimated"
                         else pd.Series(dtype="datetime64[ns]") if c in _DATE_COLUMNS
                         else pd.Series(dtype="float64") if c in _NUM_COLUMNS
                         else pd.Series(dtype="object")
                         for c in CANONICAL_COLUMNS})


def _coerce_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Force the canonical column set, order and dtypes onto ``df``.

    Rows with no ``ex_date`` fall back to ``announce_date`` as their timeline
    anchor, flagged via ``ex_date_estimated``. This matters a lot: in the real
    cache 172 rows lack an ex-date and **164 of them are the stock-dividend and
    rights-issue events** — i.e. precisely the dilution signal. Dropping them
    (as an earlier version did) silently understated dilution and made 13
    symbols read as having no history at all. An event we can only place to
    within a few weeks is still worth counting; an event we can't place at all
    is not, so a row with neither date is still dropped.
    """
    out = pd.DataFrame(index=df.index)
    for c in CANONICAL_COLUMNS:
        if c == "ex_date_estimated":
            continue
        col = df[c] if c in df.columns else None
        if c in _DATE_COLUMNS:
            out[c] = pd.to_datetime(col, errors="coerce") if col is not None else pd.NaT
        elif c in _NUM_COLUMNS:
            out[c] = pd.to_numeric(col, errors="coerce") if col is not None else float("nan")
        else:
            out[c] = col.astype("object") if col is not None else ""

    estimated = out["ex_date"].isna() & out["announce_date"].notna()
    out.loc[estimated, "ex_date"] = out.loc[estimated, "announce_date"]
    out["ex_date_estimated"] = estimated.fillna(False).astype(bool)

    out = out.dropna(subset=["ex_date"])
    out = out[CANONICAL_COLUMNS]
    return out.sort_values("ex_date").reset_index(drop=True)


# VCI ``ISS`` subtype -> canonical kind. Matched against ``event_title_en``
# (then ``event_title_vi``), which the feed formats as
# "Share Issue - <subtype> ratio <pct>%". Verified against the live endpoint:
# the payload carries NO subscription/issue-price field for ISS events, so the
# title is the only discriminator available.
#
# The distinction that matters is WHO gets the new shares:
#   stock     — existing holders, free (stock dividend / bonus issue). This is
#               the "paid me in shares instead of cash" signal.
#   rights    — existing holders, for a price (a rights offering takes cash OUT).
#   placement — third parties (private placement) or staff (ESOP). Dilutive, but
#               NOT a distribution to the holder, so it must not be summed into
#               the stock-dividend ratio.
_ISS_SUBTYPES = (
    ("stock dividend", "stock"),
    ("cổ tức bằng cổ phiếu", "stock"),
    ("bonus issue", "stock"),
    ("cổ phiếu thưởng", "stock"),
    ("rights", "rights"),
    ("quyền mua", "rights"),
    ("esop", "placement"),
    ("cbcnv", "placement"),
    ("private placement", "placement"),
    ("riêng lẻ", "placement"),
    ("public offering", "placement"),
    ("chào bán", "placement"),
)


def _classify_issuance(row: pd.Series) -> tuple[str, float, str]:
    """Classify a VCI ``ISS`` event, returning ``(kind, subscription_vnd,
    parsed_from)``.

    Driven entirely by the event title — see ``_ISS_SUBTYPES``. An unrecognised
    subtype becomes ``placement`` (the conservative choice: it still counts as
    dilution, but it does NOT inflate the stock-dividend ratio, which is the
    number the agent reads as "this company pays me in shares").
    """
    for key in ("event_title_en", "event_title_vi"):
        title = str(row.get(key, "") or "").lower()
        if not title:
            continue
        for needle, kind in _ISS_SUBTYPES:
            if needle in title:
                # A rights offering has a subscription price, but the feed
                # doesn't carry one; fall back to par so the field isn't empty.
                sub = _PAR_VALUE_VND if kind == "rights" else float("nan")
                return kind, sub, f"ISS/{kind}/{needle}"
    return "placement", float("nan"), "ISS/placement/unclassified"


def _parse_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a VCI ``events()`` frame onto the canonical schema.

    Takes both ``DIV`` (cash dividend) and ``ISS`` (issuance) events. Earlier
    versions kept only ``DIV``, which silently discarded every stock dividend
    and rights issue — i.e. the entire dilution picture.
    """
    code = raw["event_code"].astype(str).str.upper()
    keep = raw[code.isin(["DIV", "ISS"])].copy()
    if keep.empty:
        return empty_history()
    keep["_code"] = code[keep.index]

    rows = []
    for _, r in keep.iterrows():
        if r["_code"] == "DIV":
            kind, subscription, parsed_from = ("cash", float("nan"),
                                              "DIV/value_per_share[vnd]")
            cash = pd.to_numeric(r.get("value_per_share"), errors="coerce")
        else:
            kind, subscription, parsed_from = _classify_issuance(r)
            cash = float("nan")
        rows.append({
            "kind": kind,
            # `exright_date` is the ex-date; vnstock already converts it from a
            # ms timestamp. `issue_date` is the ISS analogue of `payout_date`.
            "ex_date": r.get("exright_date"),
            "record_date": r.get("record_date"),
            "pay_date": r.get("payout_date") if r["_code"] == "DIV" else r.get("issue_date"),
            "announce_date": r.get("public_date"),
            "cash_per_share_vnd": cash,
            "ratio": pd.to_numeric(r.get("exercise_ratio"), errors="coerce"),
            "subscription_vnd": subscription,
            "title": r.get("event_title_en") or r.get("event_title_vi") or "",
            "source": "VCI",
            "parsed_from": parsed_from,
        })
    return _coerce_canonical(pd.DataFrame(rows))


def fetch_dividend_history(symbol: str) -> pd.DataFrame:
    """Fetch the corporate payout/issuance history for ``symbol``.

    Returns a frame in ``CANONICAL_COLUMNS`` (one row per cash dividend, stock
    dividend or rights issue, most-recent last), or an empty frame if the symbol
    has none / every source fails.
    """
    cfg = load_config()
    bypass_quota = bool(cfg.data.get("bypass_vnai_quota", True))

    last_err: Exception | None = None
    for src in _DIVIDEND_SOURCES:
        try:
            _limiter(src).wait()
            _cprint(f"{src} is fetching dividends...")
            raw = _events_history(symbol, src, bypass_quota)
        except Exception as e:  # noqa: BLE001 - try next source
            last_err = e
            if _looks_like_rate_limit(e):
                cooldowns = cfg.data.get("cooldown_seconds_overrides", {}) or {}
                cooldown = float(cooldowns.get(src, cfg.data.get("cooldown_seconds", 60)))
                _limiter(src).pause(cooldown)
            continue
        if raw is None or raw.empty or "event_code" not in raw.columns:
            continue
        return _parse_events(raw)
    if last_err is not None:
        raise RuntimeError(f"dividend fetch failed for {symbol} on all sources: {last_err}")
    return empty_history()


def update_dividends(symbols: list[str]) -> dict:
    """Refresh the dividend-history parquet cache for each symbol. Returns
    ``{symbol: n_rows | error_str}``, mirroring ``fetcher.update_many``'s
    result shape."""
    results: dict = {}
    for sym in symbols:
        sym = sym.upper()
        try:
            df = fetch_dividend_history(sym)
        except Exception as e:  # noqa: BLE001
            results[sym] = str(e)
            continue
        df.to_parquet(dividend_cache_path(sym), index=False)
        results[sym] = int(len(df))
    return results


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map any KNOWN on-disk dividend-cache shape onto ``CANONICAL_COLUMNS``.
    Returns None for an unrecognised shape (caller treats that as empty).

    Three shapes exist in the wild, because the cache outlived two rewrites:

    1. **canonical** — what ``fetch_dividend_history`` writes now.
    2. **legacy-wide** — written 2026-06-30 by a script that is no longer in the
       repo (absent from git history entirely). It is RICHER than what replaced
       it: it already split cash/stock/rights and captured the announcement
       date. 148 of 150 cached files are this shape, and because the previous
       reader rejected any frame missing its exact column set, every one of them
       read as "no dividend history" — VCB, GAS, SAB, REE included. Adapting it
       here is what makes those symbols visible again without a refetch.
    3. **div-only** — the immediately-previous schema (2 files: VNM, FPT). It had
       already filtered to ``DIV`` events, so every row is a cash dividend.
    """
    cols = set(df.columns)

    if {"kind", "ex_date"}.issubset(cols):
        return _coerce_canonical(df)

    if {"cutoff_date", "cash_vnd"}.issubset(cols):
        renamed = df.rename(columns={
            "cutoff_date": "ex_date",
            "cash_vnd": "cash_per_share_vnd",
            "public_date": "announce_date",
        })
        # That writer's `rights` label is NOT trustworthy. Every such row it
        # produced carries `parsed_from = ISS/rights/par-fallback` — its guess
        # bucket for any issuance it couldn't map — and inspecting the real
        # cache shows it swept ESOP grants in there too (ratios as low as
        # 0.0007). Downgrade to `placement`: still counted as dilutive issuance,
        # but we don't assert a rights offering we can't evidence. A refetch
        # reclassifies properly from the event title.
        if "kind" in renamed.columns:
            renamed["kind"] = renamed["kind"].astype(str).replace(
                {"rights": "placement"})
        return _coerce_canonical(renamed)

    if {"ex_date", "cash_per_share_vnd"}.issubset(cols):
        renamed = df.rename(columns={
            "payout_date": "pay_date",
            "exercise_ratio": "ratio",
        })
        # That writer kept only event_code == "DIV".
        renamed["kind"] = "cash"
        renamed["parsed_from"] = "legacy/div-only"
        return _coerce_canonical(renamed)

    return None


def read_dividend_history(symbol: str) -> pd.DataFrame:
    """Read ``symbol``'s cached payout history, normalized to
    ``CANONICAL_COLUMNS``. An unreadable or unrecognised file reads as empty
    rather than raising, so one bad cache file can't break a whole run."""
    p = dividend_cache_path(symbol)
    if not p.exists():
        return empty_history()
    try:
        df = pd.read_parquet(p)
    except Exception:  # noqa: BLE001 - a corrupt file shouldn't kill the run
        return empty_history()
    if df.empty:
        return empty_history()
    out = _normalize_history(df)
    return empty_history() if out is None else out


def dividend_summary(symbol: str, close_vnd_thousand: float | None = None,
                     as_of: dt.date | None = None,
                     years: int | None = None) -> dict:
    """Compute the plain data columns the dividend mode hands the LLM agent:
    ``dividend_yield_ttm`` (trailing-12-month cash dividends / current price),
    ``years_paid_consecutive`` (consecutive calendar years with >=1 cash
    dividend, counting back from the most recent), ``last_ex_date``, and
    ``payout_trend`` (rising/flat/declining — compares the total cash/share
    paid in the most recent payout year vs the year before).

    It also returns the inputs the agent extrapolates from when forecasting the
    NEXT 12 months (``cash_paid_ttm_vnd``, ``payouts_per_year``,
    ``announce_lead_days``) and the deterministic DILUTION signal
    (``stock_div_ratio_recent``, ``rights_recent``) — a company "paying" in new
    shares, or funding cash payouts with a rights issue, is diluting the holder.

    ``close_vnd_thousand`` is the OHLCV ``close`` (thousand-VND units, same
    scale as the rest of the pipeline) used to compute the yield; if omitted,
    ``dividend_yield_ttm`` is NaN. ``years`` bounds the lookback for
    ``payout_trend`` / cadence / dilution (default:
    ``strategy.dividend.trend_lookback_years`` in config).

    NOTE every income metric below is computed from ``kind == "cash"`` rows only.
    The history now also carries stock dividends and rights issues, and counting
    those as "a dividend was paid" would inflate ``years_paid_consecutive`` and
    corrupt ``payout_trend``.
    """
    as_of = as_of or dt.date.today()
    hist = read_dividend_history(symbol)
    out = {
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
    if hist.empty:
        return out

    cfg = load_config()
    div_cfg = dict(getattr(cfg, "strategy", {}) or {}).get("dividend", {}) or {}
    lookback_years = int(years if years is not None
                         else div_cfg.get("trend_lookback_years", 3))

    hist = hist.dropna(subset=["ex_date"]).sort_values("ex_date")
    as_of_ts = pd.Timestamp(as_of)
    window_start = as_of_ts - pd.DateOffset(years=lookback_years)

    kind = hist["kind"].astype(str)
    cash = hist[kind == "cash"]
    out["n_dividend_events"] = int(len(cash))

    # ---- dilution over the lookback window -------------------------------
    # Three distinct things, kept apart on purpose: shares handed to holders
    # (stock), shares sold to holders (rights), shares issued to others
    # (placement/ESOP). Summing them would tell the agent a private placement is
    # a dividend paid in shares, which it is not.
    recent = hist[(hist["ex_date"] > window_start) & (hist["ex_date"] <= as_of_ts)]
    recent_kind = recent["kind"].astype(str)
    out["stock_div_ratio_recent"] = round(
        float(recent.loc[recent_kind == "stock", "ratio"].fillna(0).sum()), 4)
    out["rights_recent"] = int((recent_kind == "rights").sum())
    out["placement_ratio_recent"] = round(
        float(recent.loc[recent_kind == "placement", "ratio"].fillna(0).sum()), 4)

    if cash.empty:
        return out

    out["last_ex_date"] = cash["ex_date"].max().date().isoformat()

    # ---- trailing 12-month cash ------------------------------------------
    ttm_start = as_of_ts - pd.DateOffset(years=1)
    ttm = cash[(cash["ex_date"] > ttm_start) & (cash["ex_date"] <= as_of_ts)]
    ttm_cash = float(ttm["cash_per_share_vnd"].fillna(0).sum())
    out["cash_paid_ttm_vnd"] = round(ttm_cash, 2)
    if close_vnd_thousand is not None and float(close_vnd_thousand) > 0:
        close_vnd = float(close_vnd_thousand) * 1000.0
        out["dividend_yield_ttm"] = round(ttm_cash / close_vnd, 4)

    # ---- cadence: cash payments per year over the lookback window --------
    n_recent_cash = int(((cash["ex_date"] > window_start)
                        & (cash["ex_date"] <= as_of_ts)).sum())
    out["payouts_per_year"] = round(n_recent_cash / lookback_years, 2)

    # ---- announcement lead time -----------------------------------------
    # Excludes rows whose ex_date WAS the announce date (see
    # `_coerce_canonical`) — including them would drag the median toward 0.
    lead_src = cash[~cash["ex_date_estimated"].astype(bool)
                   & cash["announce_date"].notna()]
    if not lead_src.empty:
        lead = (lead_src["ex_date"] - lead_src["announce_date"]).dt.days
        lead = lead[lead >= 0]
        if not lead.empty:
            out["announce_lead_days"] = float(lead.median())

    # ---- consecutive calendar years with at least one CASH dividend ------
    years_paid = set(cash["ex_date"].dt.year.tolist())
    y = cash["ex_date"].max().year
    streak = 0
    while y in years_paid:
        streak += 1
        y -= 1
    out["years_paid_consecutive"] = streak

    # ---- payout trend: most recent payout year vs the year before --------
    by_year = cash.groupby(cash["ex_date"].dt.year)["cash_per_share_vnd"].sum()
    recent_years = sorted(by_year.index)[-lookback_years:]
    by_year = by_year.loc[recent_years]
    if len(by_year) >= 2:
        latest, prior = by_year.iloc[-1], by_year.iloc[-2]
        if prior <= 0:
            out["payout_trend"] = "rising" if latest > 0 else "unknown"
        elif latest > prior * 1.05:
            out["payout_trend"] = "rising"
        elif latest < prior * 0.95:
            out["payout_trend"] = "declining"
        else:
            out["payout_trend"] = "flat"
    return out
