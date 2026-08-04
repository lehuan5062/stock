"""Shared plumbing for the three LLM-agent-only modes (momentum / rebound /
dividend): universe assembly, plan emission bookkeeping, and picks JSON +
ledger writing. Each mode module (``momentum.py`` / ``rebound.py`` /
``dividend.py``) is a thin ``run()`` / ``finalize()`` pair that calls into
these helpers — this is where the duplication that used to live in
``modes/claude.py`` + ``modes/base.py`` was consolidated.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..config import load_config, reports_dir
from ..picks_meta import picks_suffix
from ..tracking import effective_today_for_trading, record, run_signature


def emit_universe_meta(plan_path: Path, universe: pd.DataFrame, *,
                       method: str, n_picks: int, hose_only: bool,
                       include_etfs: bool, exclude: list[str],
                       sig: str, mode: str | None = None) -> None:
    """Write the ``.candidates.parquet`` sidecar (full universe, for
    finalize to recover reference columns) + ``.meta.json`` (run params) next
    to the plan markdown — same convention across all 3 modes.

    ``mode`` is written so ``cli.finalize`` can read it back directly instead of
    inferring the mode from the ``{mode}_plan_*`` filename. That branch already
    existed in the CLI but was dead, because this function never wrote the key.
    The filename fallback still applies to plans emitted before this change.
    """
    sidecar = plan_path.with_suffix(".candidates.parquet")
    universe.to_parquet(sidecar, index=False)
    meta_path = plan_path.with_suffix(".meta.json")
    payload = {
        "method": method,
        "n_picks": n_picks,
        "hose_only": hose_only,
        "include_etfs": include_etfs,
        "exclude": exclude,
        "run_signature": sig,
    }
    if mode is not None:
        payload["mode"] = mode
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_meta(plan_path: Path) -> dict:
    meta_path = plan_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_candidates_sidecar(plan_path: Path) -> pd.DataFrame | None:
    sidecar = plan_path.with_suffix(".candidates.parquet")
    if not sidecar.exists():
        return None
    return pd.read_parquet(sidecar)


def write_picks_json(mode: str, merged: pd.DataFrame, plan_path: Path,
                     meta: dict, *, bar_col: str = "below_recovery_bar",
                     extra: dict | None = None) -> tuple[Path, str, pd.Timestamp]:
    """Write ``picks_<mode>_<date>_<sig>.json`` and record the ledger row.
    Returns (out_path, sig, today_ts)."""
    eff_hose = bool(meta.get("hose_only", False))
    eff_etfs = bool(meta.get("include_etfs", True))
    eff_excl = list(meta.get("exclude") or [])
    sig = meta.get("run_signature") or run_signature(
        mode=mode, hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)

    requested_n = meta.get("n_picks")
    if requested_n and len(merged) > int(requested_n):
        merged = merged.head(int(requested_n)).reset_index(drop=True)

    n_below = (int(merged[bar_col].fillna(True).sum())
              if bar_col in merged.columns else 0)
    today_ts = effective_today_for_trading()
    today = today_ts.strftime("%Y-%m-%d")
    out = reports_dir() / f"picks_{mode}_{today}_{sig}{picks_suffix(merged)}.json"
    payload = {
        "as_of": today,
        "mode": mode,
        "hose_only": eff_hose,
        "include_etfs": eff_etfs,
        "exclude": eff_excl,
        "run_signature": sig,
        "selection": "llm_pick",
        "requested_picks": requested_n,
        "n_picks": int(len(merged)),
        f"n_below_{bar_col}" if bar_col != "below_recovery_bar" else "n_below_recovery_bar": n_below,
        "plan_file": str(plan_path),
        "picks": json.loads(merged.to_json(orient="records")),
    }
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    record(merged, mode=mode, as_of=today_ts,
          hose_only=eff_hose, include_etfs=eff_etfs, exclude=eff_excl)
    return out, sig, today_ts


def resolve_on_date(on: str | None) -> dt.date:
    if on is not None:
        return dt.date.fromisoformat(on)
    return effective_today_for_trading().date()


def default_n_picks(n_picks: int | None) -> int:
    return int(n_picks) if n_picks else int(load_config().pricing.get("default_picks", 5))
