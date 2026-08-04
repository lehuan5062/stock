"""Shared engine for the two swing modes (momentum / rebound).

These two strategies are mechanically IDENTICAL — same universe gate, same
agent-supplied N/P predictions, same ``score = P / N`` ranking, same
buy-at-close / target = ``close × (1 + P)`` / no-stop pricing. They differ
ONLY in the rubric paragraph the agent is given: momentum hunts an organic
sustainable uptrend (and avoids blow-off tops), rebound hunts a temporary
healthy dip (and avoids falling knives). That text lives in
``news.llm_plan_runner._RUBRIC``, keyed by mode.

So there is one implementation here, and ``momentum.py`` / ``rebound.py`` are
thin shims that bind the mode name. Keeping those two modules (rather than
deleting them and teaching the CLI about this one) means ``cli._mode_module``,
``cli.MODES`` and the ``{mode}_plan_*`` filename convention that
``cli.finalize`` uses to detect the mode all keep working untouched.

The dividend mode deliberately does NOT route through here: it differs in its
universe columns, its results-table shape, its validation, its pricing and its
scoring — see ``modes/dividend.py``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..news.llm_plan_runner import (SWING_MODES, parse_llm_plan,
                                    write_llm_plan)
from ..pricing import add_recovery_price_suggestions
from ..selector import eligible_universe
from ..tracking import run_signature
from .common import (default_n_picks, emit_universe_meta, read_candidates_sidecar,
                     read_meta, resolve_on_date, write_picks_json)

# Reference columns recovered from the ``.candidates.parquet`` sidecar at
# finalize time so the picks JSON records the state each pick was judged on.
REF_COLS = ("symbol", "close", "rsi_14", "mom_5", "mom_20", "high_prox_20",
            "adv_vnd_20", "organ_name", "instrument_type")

MODES = SWING_MODES


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(
            f"swing: unsupported mode {mode!r} (expected one of {MODES})")
    return mode


def run(mode: str, on: str | None = None, n_picks: int | None = None,
       symbols: list[str] | None = None, hose_only: bool = False,
       include_etfs: bool = True, exclude: list[str] | None = None
       ) -> tuple[pd.DataFrame, Path]:
    """Emit the plan markdown for ``mode``. Returns (universe_df, plan_path)."""
    _check_mode(mode)
    requested_n = default_n_picks(n_picks)
    universe = eligible_universe(on=on, symbols=symbols)
    on_date = resolve_on_date(on)
    excl_list = sorted({s.upper() for s in (exclude or [])})
    sig = run_signature(mode=mode, hose_only=hose_only,
                        include_etfs=include_etfs, exclude=excl_list)
    plan_path = write_llm_plan(mode, universe, on=on_date, run_signature=sig,
                              n_picks=requested_n)
    emit_universe_meta(plan_path, universe, method="llm_only",
                       n_picks=requested_n, hose_only=hose_only,
                       include_etfs=include_etfs, exclude=excl_list, sig=sig,
                       mode=mode)
    return universe, plan_path


def finalize(mode: str, plan_path: str | Path) -> tuple[pd.DataFrame, Path]:
    """Parse the filled plan for ``mode``, price and rank it, write the picks
    JSON + ledger row. Returns (picks_df, picks_json_path)."""
    _check_mode(mode)
    plan_path = Path(plan_path)
    scored = parse_llm_plan(plan_path)
    if scored.empty:
        raise RuntimeError(f"no picks parsed from {plan_path} — fill the Results table")

    dropped = scored[scored["dropped"]]
    if not dropped.empty:
        print(f"[{mode}] DROP: excluding {len(dropped)} ticker(s): "
              f"{', '.join(dropped['symbol'].tolist())}")
    scored = scored[~scored["dropped"]].drop(columns=["dropped"])
    if scored.empty:
        raise RuntimeError("all picks dropped")

    bad = scored[scored["pred_days"].isna() | (scored["pred_days"] < 1)
                | scored["pred_profit"].isna() | (scored["pred_profit"] <= 0)]
    if not bad.empty:
        print(f"[{mode}] WARNING: dropping {len(bad)} pick(s) with a missing/"
              f"invalid N_days or P: {', '.join(bad['symbol'].tolist())}")
    scored = scored.drop(bad.index)
    if scored.empty:
        raise RuntimeError("no picks with a valid N_days and P")

    universe = read_candidates_sidecar(plan_path)
    if universe is not None:
        ref_cols = [c for c in REF_COLS if c in universe.columns]
        merged = scored.merge(universe[ref_cols], on="symbol", how="left")
    else:
        merged = scored

    merged = add_recovery_price_suggestions(merged)
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    meta = read_meta(plan_path)
    out, sig, _ = write_picks_json(mode, merged, plan_path, meta)
    return merged, out
