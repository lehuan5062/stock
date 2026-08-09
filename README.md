# HUANREAL Vietnamese Stock Predictor

A 100% LLM-agent-driven stock screener for the Vietnamese market. There is
**no machine-learning model anywhere in the live path** — the program's job
is only to fetch data (OHLCV + dividend history) and apply true mechanical
gates (stale cache / limit-up-locked / unadjusted corporate-action). An LLM
coding agent (Claude Code, Cowork, or an equivalent tool-using agent) does all
the picking, ranking, and pricing, driven by [`agent_prompt.md`](agent_prompt.md).

## Three modes

| Mode | Horizon | Agent predicts | Exit |
| ---- | ------- | --------------- | ---- |
| **momentum** | **short term** — trend-following | `N_days` + `P` (profit) | buy at close, target = `close×(1+P)`, no stop, hold until target |
| **rebound** | **medium term** — mean-reversion bounce | `N_days` + `P` | same as momentum |
| **dividend** | **long term** — income hold | `fwd_dps_vnd` (next-12m dividend) + `expected_hold_years` + `confidence` | buy at close, no target, no stop — a hold |

The modes aren't better-or-worse than one another — they're different holding
horizons, so which to run is your call. The agent never recommends one.

Momentum and rebound share one objective (`score = P/N`, profit per day
held) and one trade shape (buy at close, flexible exit, no stop-loss —
backtests showed a stop backfires on this kind of mean-reversion/momentum
bet). Dividend is a hold, and it **forecasts**: the deterministic fetcher
computes real payout history (trailing yield, cash paid, payout cadence,
consecutive years, trend) plus a dilution signal (stock-dividend ratio and
rights-issue count), and the agent predicts `fwd_dps_vnd` — the cash dividend
per share it expects over the next 12 months. Ranking is
`pred_forward_yield × confidence`, so the forecast decides the order; the
trailing yield is an input, never the answer. The agent never has to search for
the historical numbers.

You can run one mode or several in a session; when more than one runs, the
agent gives a final cross-mode recommendation.

## How to run

### A. Agent session (recommended)

1. Open a capable LLM coding-agent session (Claude Code, Cowork, or
   equivalent).
2. Paste the contents of [`agent_prompt.md`](agent_prompt.md) into the chat.
3. The agent asks for mode(s) / picks / hose-only / etfs / data freshness /
   exclude, then drives the whole pipeline: `run` → research → fill the
   plan → `finalize`. Picking "full refresh" opens the fetch in its own
   terminal window so you can watch KBS/VCI progress live.

### B. CLI (advanced / scripting)

```bash
# Fetch data + emit the plan markdown for one mode
.venv\Scripts\python -m stockpredict.cli run --mode rebound --picks 1

# Once the plan markdown is filled in (by you or an agent):
.venv\Scripts\python -m stockpredict.cli finalize "reports\rebound_plan_<date>_<sig>.md"

# Refresh dividend history for specific symbols (dividend mode)
.venv\Scripts\python -m stockpredict.cli update-dividends -s VNM -s FPT

# Other commands
.venv\Scripts\python -m stockpredict.cli update-data           # refresh full OHLCV cache
.venv\Scripts\python -m stockpredict.cli evaluate               # stamp resolved picks
.venv\Scripts\python -m stockpredict.cli track --limit 20
.venv\Scripts\python -m stockpredict.cli status                 # what's cached
.venv\Scripts\python -m stockpredict.cli compare-modes --window 90
```

### `run` flags

| flag | meaning | default |
| ---- | ------- | ------- |
| `--mode {momentum,rebound,dividend}` | which strategy | required |
| `--picks N`, `-n N` | how many picks the agent should surface | `pricing.default_picks` (1) |
| `--hose-only` | restrict the universe to HOSE-listed tickers | `False` |
| `--etfs` / `--no-etfs` | include/exclude HOSE ETFs | included |
| `--exclude ACB,HPG` | per-session ticker blacklist | none |
| `--warm-only {yes,always,no}` | cache-fetch strategy (see below) | `yes` |

## The pipeline, in detail

1. **Mechanical gates only** ([`filters.py`](src/stockpredict/filters.py)):
   `staleness_mask` (stale cache), `ceiling_lock_mask` (limit-up locked,
   can't fill a buy), `corporate_action_mask` (unadjusted split/rights spike
   poisons the technical columns). These are the only hard-coded excludes —
   liquidity size, overbought RSI, and downtrend shape are **not** gates any
   more; the underlying columns (`adv_vnd_20`, `adv_active_days_20`, `close`,
   `rsi_14`, `history_days`, plus `mom_5`/`mom_20`/`high_prox_20`/`atr_14`)
   are handed to the agent as plain data (see
   [`selector.eligible_universe`](src/stockpredict/selector.py)) — it judges
   tradability, size, and trend/overbought state itself.
2. **Plan markdown** ([`news/llm_plan_runner.py`](src/stockpredict/news/llm_plan_runner.py)
   for momentum/rebound, [`modes/dividend.py`](src/stockpredict/modes/dividend.py)
   for dividend): the whole eligible universe, unranked, with an empty
   results table shaped for that mode.
3. **Agent research pass**: the agent selects, researches, and fills the
   plan (see [`agent_prompt.md`](agent_prompt.md) for the full per-mode
   rubric).
4. **Finalize** ([`modes/swing.py`](src/stockpredict/modes/swing.py), shared by
   momentum + rebound, which are mechanically identical and differ only in
   their prompt rubric — [`modes/momentum.py`](src/stockpredict/modes/momentum.py)
   and [`modes/rebound.py`](src/stockpredict/modes/rebound.py) are thin shims
   binding the mode name; plus
   [`modes/dividend.py`](src/stockpredict/modes/dividend.py)): ranks, prices
   via [`pricing.py`](src/stockpredict/pricing.py)
   (`add_recovery_price_suggestions` for momentum/rebound,
   `add_dividend_price_suggestions` for dividend), writes
   `picks_<mode>_<date>_<sig>.json`, records to the ledger.

### Fees (ACBS default)

[`config.yaml`](config.yaml) → `broker:`. Round-trip ≈ **0.43%** of trade
value:
```
buy_fee  = trade_value × 0.15% × 1.10           = 0.165%
sell_fee = trade_value × 0.15% × 1.10 + 0.10%   = 0.265%
total    ≈ 0.43%
```
This feeds `pricing.profit_threshold()` (round-trip + `pricing.profit_margin`
≈ 0.93% total), the bar momentum/rebound's `P` must clear to avoid the
`below_recovery_bar` flag.

### Dividend data fetcher

[`data/dividends.py`](src/stockpredict/data/dividends.py) uses the same
vnai-quota-bypass technique as OHLCV (`fetcher.py`), but the only source with
a real, populated dividend-events endpoint on the installed vnstock version
is **VCI's company-events feed** (`Company(symbol, source="VCI").events()`)
— KBS's analogous endpoint returned empty for every symbol tried during
implementation. Both `DIV` (cash dividend) and `ISS` (issuance) events are
kept, classified by who receives the new shares: `cash`, `stock` (free to
existing holders — stock dividend / bonus issue), `rights` (sold to existing
holders), `placement` (third parties or staff — private placement / ESOP). Only
cash rows feed the yield; the rest give the dividend mode a deterministic
**dilution** signal instead of asking the agent to research it. The classifier
reads `event_title_en` because VCI's ISS payload carries no subscription-price
field. Legacy cache files predate the split and lump placements into `stock`, so
their stock-dividend ratio is an upper bound until refetched.

Cached to `cache/dividends/<SYM>.parquet`. `read_dividend_history` normalizes
every historical on-disk shape onto one canonical schema — the cache outlived
two rewrites, and an earlier reader rejected any file it didn't recognise,
which silently made ~148 of 150 cached symbols read as "no dividend history".
Rows with no ex-date fall back to the announcement date (flagged
`ex_date_estimated`) rather than being dropped, since most of them are the
stock/rights events that carry the dilution signal.

`update-dividends` is separable from `update-data` so a dividend-only refresh
doesn't require a full OHLCV re-fetch.

## Universe coverage, cache, and `--warm-only`

Every run covers the **entire universe** (all of HOSE + HNX + UPCOM, ~1,760
tickers) with no time cap. The **stock list is consolidated from both KBS and
VCI** — each source is fetched, then merged by symbol-completeness. ETF lists
come from KBS only (VCI doesn't support ETFs).

OHLCV comes from **KBS and VCI** via vnstock, with cross-source failover on
429s. The fetcher bypasses vnstock's 20/min guest quota and self-throttles to
`data.api_per_min` (default 60, per-source overrides in
`data.api_per_min_overrides`). Expect ~30 minutes the first time; subsequent
runs are near-instant because up-to-date tickers cost 0 API calls.

`--warm-only` is tri-state, default `yes`:

| `--warm-only` | warm | stale | cold (no parquet) | use when |
| ------------- | ---- | ----- | ------------------ | -------- |
| **`yes`** (smart lazy) | skip | fetch new bar | fetch full history | every-day usage |
| **`always`** (offline) | keep | keep | drop | guaranteed zero API calls |
| **`no`** (force) | refetch | refetch | refetch | backfill / corrections |

A write-time guard rejects any incrementally fetched bar whose move against
the last cached close exceeds the exchange band + margin (physically
impossible for a normal trading day) and heals the symbol with an automatic
full re-fetch.

## Trading-day calendar & T+2 settlement

Vietnamese settlement is **T+2**: a bought share can't be sold before two
trading days have passed — momentum/rebound's flexible-exit evaluator only
looks for a profitable exit from T+2 onward. The calendar is built from the
actual cached OHLCV index, so weekends and Vietnamese holidays are skipped
(see [`tracking.py`](src/stockpredict/tracking.py)).

**After-14:30 cutoff:** runs at/before 14:30 stamp T+0 = today; after 14:30
(close locked in) T+0 = the next trading day.

## Same-day re-runs: distinct params → distinct files

Every artifact is suffixed with a **run signature** capturing the
pick-affecting parameters (`mode[_HOSE][_noETF][_x<TICKERS>]`), and that
signature is the ledger `run_id` base. Re-running the same parameters
overrides (idempotent); different parameters never clobber each other.

## Setup (one-time)

Double-click [`setup.bat`](setup.bat): creates `.venv` (Python 3.13),
installs `stockpredict` + all dependencies, and prints the installed
versions.

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e ".[dev,llm]"
```

Optional API keys live in [`.env.example`](.env.example); copy to `.env`.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```

All synthetic — no network. Coverage spans the mechanical gates, the
eligible-universe builder, momentum/rebound pricing + finalize, dividend
pricing, the ledger + flexible-exit evaluator, the trading calendar, cache
freshness + watermarks, the rate limiter, the write-time corporate-action
guard, and the phantom-spike cache-repair detector.

## Configuration

All knobs in [`config.yaml`](config.yaml). Only two kinds of things live
there now: (a) true mechanical-gate thresholds
(`universe.liquidity_filter.max_staleness_days`, `pricing.ceiling_limits` /
`ceiling_tol`, `pricing.corp_action_lookback`) and (b) data-plumbing / fee
math (`data.*` fetch rates, `broker:`, `pricing.profit_margin` /
`settle_days`, `strategy.dividend.*` fetch depth). Judgment thresholds that
used to be config knobs (liquidity size, overbought RSI, downtrend shape,
recovery-probability minimum) are gone — the agent sees the raw columns and
reasons over them itself, per run, informed by `agent_prompt.md`'s rubric.

## Layout

```
<repo root>/
  setup.bat                double-click: create/update .venv + install dependencies
  update_data.bat          double-click: interactive OHLCV cache pre-fill (KBS/VCI), stays open on exit
  agent_prompt.md          paste into an LLM coding-agent session to run the pipeline
  config.yaml              mechanical-gate thresholds + data-plumbing knobs

  src/stockpredict/
    cli.py                 click entry point (run / finalize / update-data / update-dividends / ...)
    selector.py             curated bluechip list + universe top-up + eligible_universe (pure filter)
    filters.py              staleness / ceiling-lock / corporate-action gates (mechanical only)
    dataset.py               feature panel builder (technical + microstructure, no ML target)
    tracking.py               ledger + flexible-exit evaluator + fee-aware resolve_exit
    pricing.py                momentum/rebound + dividend price suggestions, fee model
    data/                    vnstock wrappers (+ vnai quota bypass): OHLCV, dividends, cache, universe, rate limiter
    features/                 technical + microstructure indicators
    news/                    llm_plan_runner (shared plan writer/parser) + sources + company_info
    modes/                   swing.py (momentum+rebound engine) + momentum.py / rebound.py shims
                             + dividend.py + common.py shared helpers
    analyze/                 mode_compare.py — cross-mode ledger comparison

  scripts/                 one-off diagnostics (cache repair) + demo helper
  tests/                   pytest, synthetic data only
  cache/                   OHLCV parquet cache, dividend-history cache, predictions ledger
  reports/                 picks, plans (gitignored output)
```

## License

- **This code** is **AGPL-3.0-or-later** ([`LICENSE`](LICENSE)). Forks stay
  open-source under the same terms; running a modified version as a network
  service obliges you to publish your modifications.
- **vnstock**, the runtime data dependency, is **non-commercial /
  personal-research only** under its own license — see [`NOTICE`](NOTICE).
  The AGPL here does not override that; commercial use requires a separate
  vnstock license.
- **Contributions** are accepted under AGPL-3.0 with a DCO sign-off; see
  [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Caveats

- The **20 req/min "guest" cap is vnstock's own client-side quota**, not the
  providers' server limit; `data.bypass_vnai_quota: true` calls the
  underlying endpoint directly, leaving only our own `data.api_per_min`
  throttle. A genuine 429 still triggers a pause; failed ticker fetches
  never abort the run.
- With no stop and no cap, a rare broken name that slips past the agent's
  vetting is held indefinitely (momentum/rebound) — the strategy is meant to
  win often and fast, but can warehouse an underwater unsold tail. Monitor
  and prune those manually.
- **Not investment advice.** Use at your own risk; past performance ≠ future
  results.
