"""Head-to-head comparison of prediction methods over the ledger.

Compares the realized performance of different modes — base (pure ML),
claude (ML/LLM hybrid), claude_llm (LLM-only) — restricted to
*comparable* runs: same trading day AND same run parameters (horizon /
hose-only / etfs / exclude). A single day is far too noisy (often 1-3
picks per mode), so the verdict pools over a window of days; the named-day
breakdown is shown only as context.

Comparability key: two runs are comparable when they share the same
``param_key`` — the run signature with its leading mode token stripped
(so ``base``, ``claude`` and ``claude_llm`` all map to the empty key,
but ``claude_HOSE`` does NOT match ``base``). A ``(as_of, param_key)``
cell with two or more distinct modes is a *comparable cell*; everything
pools over those cells only.

This is advisory analysis, not a program edit: mode is a per-run user
choice, so the output is "which method has been winning", not a config
knob to tune.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ..tracking import _read

# Friendly labels for the report; unknown modes fall through to themselves.
MODE_LABELS = {
    "momentum": "momentum (short-term trend-following)",
    "rebound": "rebound (mean-reversion bounce)",
    "dividend": "dividend (long-term hold)",
}


def _label(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def _param_key(signature: str, mode: str) -> str:
    """Run signature with its leading mode token stripped, so the same
    parameters under different modes share a key. A default-params run has
    signature == mode exactly (e.g. ``base`` / ``claude``), which maps to the
    empty key. Falls back to the raw signature when it doesn't start with the
    mode prefix."""
    sig = str(signature)
    if sig == str(mode):
        return ""
    prefix = f"{mode}_"
    return sig[len(prefix):] if sig.startswith(prefix) else sig


def compare_modes(window_days: int = 90,
                  as_of: str | dt.date | None = None,
                  modes: list[str] | None = None) -> dict:
    """Pool realized performance per mode over comparable cells in the window.

    Returns a dict with: ``window_days``, ``as_of`` (the named context day or
    None), ``n_comparable_cells``, ``per_mode`` (pooled stats list),
    ``pairwise`` (matched A-vs-B over common cells), ``unique_vs_shared`` (per
    mode), and ``today`` (the named-day breakdown, when ``as_of`` is given).
    ``note`` is set instead when there is nothing comparable to report.
    """
    df = _read()
    if df.empty:
        return {"window_days": window_days, "note": "ledger is empty"}

    df = df[df["evaluated"].fillna(False) & df["realized_return"].notna()].copy()
    if df.empty:
        return {"window_days": window_days,
                "note": "no evaluated picks with realized returns yet"}

    df["as_of"] = pd.to_datetime(df["as_of"])
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(window_days))
    win = df[df["as_of"] >= cutoff].copy()
    if modes:
        keep = {str(m) for m in modes}
        win = win[win["mode"].astype(str).isin(keep)]
    if win.empty:
        return {"window_days": window_days,
                "note": f"no evaluated picks in the last {window_days} days"}

    win["param_key"] = [
        _param_key(s, m) for s, m in zip(win["signature"], win["mode"])
    ]
    win["cell"] = list(zip(win["as_of"], win["param_key"]))

    # Comparable cells: same (day, params) with >= 2 distinct modes.
    cell_modes = win.groupby("cell")["mode"].nunique()
    comparable_cells = set(cell_modes[cell_modes >= 2].index)
    comp = win[win["cell"].isin(comparable_cells)].copy()
    if comp.empty:
        return {
            "window_days": window_days,
            "n_comparable_cells": 0,
            "note": ("no comparable cells — need >= 2 modes run on the SAME day "
                     "with the SAME params (picks/horizon/hose-only/etfs/exclude) "
                     "in the window"),
        }

    out: dict = {
        "window_days": window_days,
        "as_of": str(pd.to_datetime(as_of).date()) if as_of is not None else None,
        "n_comparable_cells": len(comparable_cells),
        "per_mode": _per_mode_stats(comp),
        "pairwise": _pairwise(comp),
        "unique_vs_shared": _unique_vs_shared(comp),
    }
    if as_of is not None:
        out["today"] = _named_day(win, pd.to_datetime(as_of), comparable_cells)
    return out


def _agg(returns: pd.Series) -> dict:
    r = returns.dropna()
    n = int(len(r))
    return {
        "n_picks": n,
        "hit_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_return": float(r.mean()) if n else float("nan"),
        "median_return": float(r.median()) if n else float("nan"),
    }


def _per_mode_stats(comp: pd.DataFrame) -> list[dict]:
    """Pooled stats per mode over comparable cells. ``mean_per_day`` averages
    each (day, params) cell first, then across cells, so a day with many picks
    doesn't dominate a day with one."""
    rows = []
    for mode, g in comp.groupby("mode"):
        stats = _agg(g["realized_return"])
        per_cell = g.groupby("cell")["realized_return"].mean()
        # Which ranking formula(s) produced these rows. The ledger keeps `rank`
        # but not `score`, so if a mode redefined its scoring mid-history this
        # pooled row silently averages two different strategies. Surface that
        # rather than hiding it — "" means a row predating the stamp.
        if "score_formula" in g.columns:
            formulas = sorted({str(v or "") for v in g["score_formula"]})
        else:
            formulas = [""]
        rows.append({
            "mode": str(mode),
            "label": _label(mode),
            "n_days": int(g["cell"].nunique()),
            **stats,
            "mean_per_day": float(per_cell.mean()) if len(per_cell) else float("nan"),
            "score_formulas": formulas,
            "mixed_formula": len(formulas) > 1,
        })
    rows.sort(key=lambda d: (d["mean_per_day"] if d["mean_per_day"] == d["mean_per_day"]
                             else -1e9), reverse=True)
    return rows


def _pairwise(comp: pd.DataFrame) -> list[dict]:
    """For every mode pair, match on the cells where BOTH ran and compare the
    per-cell mean return. ``a_wins`` / ``b_wins`` count days each method's
    cell-mean beat the other's."""
    # cell -> {mode: mean_return}
    cell_mode_ret = (comp.groupby(["cell", "mode"])["realized_return"]
                     .mean().unstack("mode"))
    modes = sorted(comp["mode"].unique())
    res = []
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            sub = cell_mode_ret[[a, b]].dropna()
            if sub.empty:
                continue
            res.append({
                "a": a, "b": b, "a_label": _label(a), "b_label": _label(b),
                "n_matched_days": int(len(sub)),
                "a_mean": float(sub[a].mean()),
                "b_mean": float(sub[b].mean()),
                "a_wins": int((sub[a] > sub[b]).sum()),
                "b_wins": int((sub[b] > sub[a]).sum()),
                "ties": int((sub[a] == sub[b]).sum()),
            })
    return res


def _unique_vs_shared(comp: pd.DataFrame) -> list[dict]:
    """Per mode: split its picks into those UNIQUE to it within a comparable
    cell vs those also picked by another mode that day, and compare realized
    returns. Shows whether a method's distinctive picks actually add value."""
    # For each cell, count how many modes picked each symbol.
    sym_counts = (comp.groupby(["cell", "symbol"])["mode"].nunique()
                  .rename("n_modes").reset_index())
    merged = comp.merge(sym_counts, on=["cell", "symbol"], how="left")
    rows = []
    for mode, g in merged.groupby("mode"):
        uniq = g[g["n_modes"] == 1]["realized_return"]
        shared = g[g["n_modes"] >= 2]["realized_return"]
        rows.append({
            "mode": str(mode),
            "label": _label(mode),
            "n_unique": int(uniq.notna().sum()),
            "unique_mean_return": float(uniq.mean()) if uniq.notna().any() else float("nan"),
            "unique_hit_rate": float((uniq > 0).mean()) if uniq.notna().any() else float("nan"),
            "n_shared": int(shared.notna().sum()),
            "shared_mean_return": float(shared.mean()) if shared.notna().any() else float("nan"),
        })
    rows.sort(key=lambda d: d["mode"])
    return rows


def _named_day(win: pd.DataFrame, day: pd.Timestamp,
               comparable_cells: set) -> dict:
    """Context-only breakdown of the single named day across modes."""
    day = day.normalize()
    sub = win[win["as_of"] == day]
    if sub.empty:
        return {"n_modes": 0, "note": f"no evaluated picks on {day.date()}"}
    picks = []
    for mode, g in sub.groupby("mode"):
        picks.append({
            "mode": str(mode), "label": _label(mode),
            "symbols": g["symbol"].tolist(),
            "mean_return": float(g["realized_return"].mean()),
            "comparable": any((c in comparable_cells) for c in g["cell"]),
        })
    picks.sort(key=lambda d: d["mean_return"], reverse=True)
    return {"n_modes": int(sub["mode"].nunique()), "picks": picks}


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{x:+.2%}"


def _rate(x: float) -> str:
    return " n/a" if x != x else f"{x:.0%}"


def format_report(result: dict) -> str:
    """Render the compare_modes() dict as a markdown report."""
    w = result.get("window_days")
    lines = [f"# Mode comparison — pooled over the last {w} days", ""]
    if result.get("note"):
        lines.append(f"**{result['note']}.**")
        return "\n".join(lines)

    lines.append(f"Comparable cells (same day + same params, >=2 modes): "
                 f"**{result['n_comparable_cells']}**.")
    lines.append("")

    lines.append("## Per-method (pooled over comparable cells)")
    lines.append("")
    lines.append("| method | days | picks | hit | mean/pick | median | mean/day |")
    lines.append("| ------ | ---- | ----- | --- | --------- | ------ | -------- |")
    for r in result["per_mode"]:
        flag = " ⚠" if r.get("mixed_formula") else ""
        lines.append(
            f"| {r['label']}{flag} | {r['n_days']} | {r['n_picks']} | "
            f"{_rate(r['hit_rate'])} | {_pct(r['mean_return'])} | "
            f"{_pct(r['median_return'])} | {_pct(r['mean_per_day'])} |"
        )
    lines.append("")
    lines.append("`mean/day` weights each day equally (averages a day's picks "
                 "first, then across days) — the fairest single number.")
    mixed = [r for r in result["per_mode"] if r.get("mixed_formula")]
    if mixed:
        lines.append("")
        for r in mixed:
            shown = ", ".join(f"`{f}`" if f else "`(pre-stamp)`"
                              for f in r["score_formulas"])
            lines.append(
                f"⚠ **{r['label']} pools more than one ranking formula** "
                f"({shown}). The ledger stores `rank` but not `score`, so these "
                f"rows were ranked by different objectives and this pooled "
                f"number blends two strategies. Segment by `score_formula` "
                f"before drawing a conclusion."
            )
    lines.append("")

    if result.get("pairwise"):
        lines.append("## Head-to-head (matched on shared days only)")
        lines.append("")
        lines.append("| A vs B | matched days | A mean/day | B mean/day | A wins | B wins | ties |")
        lines.append("| ------ | ------------ | ---------- | ---------- | ------ | ------ | ---- |")
        for p in result["pairwise"]:
            lines.append(
                f"| {p['a_label']} vs {p['b_label']} | {p['n_matched_days']} | "
                f"{_pct(p['a_mean'])} | {_pct(p['b_mean'])} | "
                f"{p['a_wins']} | {p['b_wins']} | {p['ties']} |"
            )
        lines.append("")

    if result.get("unique_vs_shared"):
        lines.append("## Distinctive picks — unique vs shared")
        lines.append("")
        lines.append("How each method's picks did when they were UNIQUE to it that "
                     "day (no other method picked the same name) vs SHARED.")
        lines.append("")
        lines.append("| method | unique n | unique mean | unique hit | shared n | shared mean |")
        lines.append("| ------ | -------- | ----------- | ---------- | -------- | ----------- |")
        for u in result["unique_vs_shared"]:
            lines.append(
                f"| {u['label']} | {u['n_unique']} | {_pct(u['unique_mean_return'])} | "
                f"{_rate(u['unique_hit_rate'])} | {u['n_shared']} | "
                f"{_pct(u['shared_mean_return'])} |"
            )
        lines.append("")

    today = result.get("today")
    if today:
        lines.append(f"## Named day {result.get('as_of')} (context only — noisy)")
        lines.append("")
        if today.get("note"):
            lines.append(f"_{today['note']}._")
        else:
            for p in today["picks"]:
                tag = "" if p["comparable"] else "  (not a comparable cell)"
                lines.append(f"- **{p['label']}**: {_pct(p['mean_return'])} "
                             f"on {', '.join(p['symbols'])}{tag}")
        lines.append("")

    lines.append("---")
    lines.append("Advisory only: a knob tweak can't change which method you run "
                 "— this tells you which to *prefer*. Pooled edges from few days "
                 "are noisy; re-check as more picks evaluate.")
    return "\n".join(lines)


def compare_picks_same_day(as_of: str | dt.date) -> dict:
    """Compare the actual picks selected by different modes on the same day,
    before any have resolved. Reads picks JSON files directly.

    Returns a dict with ``as_of``, ``modes`` (dict mode→picks list), ``overlap``
    (symbols picked by 2+ modes), and ``mode_rationales`` (LLM reasoning for each pick).
    """
    from pathlib import Path
    import json

    as_of_str = str(as_of)
    if hasattr(as_of, "date"):
        as_of_str = as_of.date().isoformat()

    reports_dir = Path("reports")
    pattern = f"picks*_{as_of_str}_*.json"
    picks_files = sorted(reports_dir.glob(pattern))

    if not picks_files:
        return {"as_of": as_of_str, "note": f"no picks reports found for {as_of_str}"}

    modes_data = {}
    all_symbols = {}  # symbol → set of modes that picked it

    for fpath in picks_files:
        with open(fpath, encoding='utf-8') as f:
            report = json.load(f)

        mode = report.get("mode", "unknown")
        picks = report.get("picks", [])

        mode_picks = []
        for pick in picks:
            sym = pick.get("symbol")
            if sym:
                mode_picks.append({
                    "symbol": sym,
                    "score": pick.get("score"),
                    "pred_profit": pick.get("pred_profit"),
                    "pred_recovery_prob": pick.get("pred_recovery_prob"),
                    "rationale": pick.get("rationale"),  # LLM modes only
                    "dimensions_cited": pick.get("dimensions_cited"),  # LLM modes only
                })
                if sym not in all_symbols:
                    all_symbols[sym] = set()
                all_symbols[sym].add(mode)

        modes_data[mode] = mode_picks

    # Overlap: symbols picked by 2+ modes
    overlap = {sym: sorted(list(modes)) for sym, modes in all_symbols.items() if len(modes) >= 2}

    # Mode rationales: for each mode, show LLM reasoning if present
    mode_rationales = {}
    for mode, picks in modes_data.items():
        rationales = {}
        for pick in picks:
            if pick.get("rationale") or pick.get("dimensions_cited"):
                rationales[pick["symbol"]] = {
                    "rationale": pick.get("rationale"),
                    "dimensions_cited": pick.get("dimensions_cited"),
                }
        if rationales:
            mode_rationales[mode] = rationales

    return {
        "as_of": as_of_str,
        "modes": modes_data,
        "overlap": overlap,
        "mode_rationales": mode_rationales,
    }


def format_picks_comparison(result: dict) -> str:
    """Render compare_picks_same_day() dict as a markdown report."""
    lines = [f"# Same-day picks comparison — {result.get('as_of')}", ""]

    if result.get("note"):
        lines.append(f"_{result['note']}._")
        return "\n".join(lines)

    modes = result.get("modes", {})
    if not modes:
        lines.append("No modes found.")
        return "\n".join(lines)

    # Per-mode picks
    lines.append("## Picks by mode")
    lines.append("")
    for mode in sorted(modes.keys()):
        picks = modes[mode]
        lines.append(f"### {_label(mode)}")
        lines.append("")
        if not picks:
            lines.append("*(no picks)*")
        else:
            for pick in picks:
                sym = pick["symbol"]
                score = pick.get("score")
                prob = pick.get("pred_recovery_prob")
                profit = pick.get("pred_profit")
                score_str = f"{score:.4f}" if score is not None else "n/a"
                prob_str = f"{prob:.2%}" if prob is not None else "n/a"
                profit_str = f"{profit:.2%}" if profit is not None else "n/a"
                lines.append(f"- **{sym}**: score={score_str}, "
                           f"recovery_prob={prob_str}, profit={profit_str}")
        lines.append("")

    # Overlap
    overlap = result.get("overlap", {})
    if overlap:
        lines.append("## Overlap (picked by 2+ modes)")
        lines.append("")
        for sym in sorted(overlap.keys()):
            mode_list = ", ".join(overlap[sym])
            lines.append(f"- **{sym}**: {mode_list}")
        lines.append("")

    # LLM rationales
    rationales = result.get("mode_rationales", {})
    if rationales:
        lines.append("## LLM rationales (claude/claude_llm only)")
        lines.append("")
        for mode in sorted(rationales.keys()):
            if mode in ("base",):
                continue  # base doesn't have rationales
            lines.append(f"### {_label(mode)}")
            lines.append("")
            for sym, details in rationales[mode].items():
                lines.append(f"**{sym}:**")
                dims = details.get("dimensions_cited")
                if dims:
                    # Handle both comma-separated string and list
                    if isinstance(dims, str):
                        dims_str = dims
                    else:
                        dims_str = ", ".join(dims)
                    lines.append(f"  Dimensions: {dims_str}")
                if details.get("rationale"):
                    lines.append(f"  Rationale: {details['rationale']}")
            lines.append("")

    return "\n".join(lines)


def mode_accountability(as_of: str | dt.date) -> dict:
    """For each resolved pick on the day, show which modes selected it and which avoided it.
    Reveals mode divergence: were the errors shared (all modes picked the knife) or
    mode-specific (only LLM picked it, or only base picked it)?

    Returns a dict with ``as_of``, ``picks_by_outcome`` (symbol → modes/avoided),
    and a summary table for diagnosis.
    """
    as_of_str = str(as_of)
    if hasattr(as_of, "date"):
        as_of_str = as_of.date().isoformat()

    # Read ledger to find which modes picked each ticker on this day
    df = _read()
    if df.empty:
        return {"as_of": as_of_str, "note": "ledger is empty"}

    df["as_of"] = pd.to_datetime(df["as_of"])
    day = pd.Timestamp(as_of_str)
    day_df = df[df["as_of"] == day].copy()

    if day_df.empty:
        return {"as_of": as_of_str, "note": f"no picks found on {as_of_str}"}

    # Build map: symbol → set of modes that picked it (from ledger)
    all_modes = {}  # symbol → set of modes
    for _, row in day_df.iterrows():
        sym = row["symbol"]
        mode = row.get("mode", "unknown")
        if sym not in all_modes:
            all_modes[sym] = set()
        all_modes[sym].add(mode)

    # Filter to only evaluated picks for accountability
    evaluated_df = day_df[day_df["evaluated"].fillna(False)]

    if evaluated_df.empty:
        return {"as_of": as_of_str, "note": f"no evaluated picks on {as_of_str}"}

    # Build accountability table: for each resolved pick, show modes and outcome
    picks_by_outcome = {}
    for _, row in evaluated_df.iterrows():
        sym = row["symbol"]
        outcome = row.get("realized_return", 0)
        recovered = row.get("recovered_flag", False)

        picked_by = sorted(list(all_modes.get(sym, set())))
        all_possible_modes = {"momentum", "rebound", "dividend"}
        avoided_by = sorted(list(all_possible_modes - all_modes.get(sym, set())))

        picks_by_outcome[sym] = {
            "outcome": outcome,
            "recovered": recovered,
            "picked_by": picked_by,
            "avoided_by": avoided_by,
        }

    return {
        "as_of": as_of_str,
        "picks_by_outcome": picks_by_outcome,
    }


def format_mode_accountability(result: dict) -> str:
    """Render mode_accountability() dict as markdown. Shows which modes picked/avoided
    each resolved ticker, to help diagnose whether errors were shared or mode-specific."""
    lines = [f"# Mode accountability — {result.get('as_of')}", ""]

    if result.get("note"):
        lines.append(f"_{result['note']}._")
        return "\n".join(lines)

    picks = result.get("picks_by_outcome", {})
    if not picks:
        lines.append("No resolved picks.")
        return "\n".join(lines)

    # Group by outcome: winners (recovered) and losers (fell/underwater)
    winners = {s: p for s, p in picks.items() if p["recovered"]}
    losers = {s: p for s, p in picks.items() if not p["recovered"]}

    if winners:
        lines.append("## Winners (recovered)")
        lines.append("")
        lines.append("| Symbol | Outcome | Picked by | Avoided by |")
        lines.append("|--------|---------|-----------|------------|")
        for sym in sorted(winners.keys()):
            p = winners[sym]
            outcome_str = f"+{p['outcome']:.2%}" if p['outcome'] > 0 else f"{p['outcome']:.2%}"
            picked = ", ".join(p["picked_by"]) if p["picked_by"] else "*(none)*"
            avoided = ", ".join(p["avoided_by"]) if p["avoided_by"] else "*(none)*"
            lines.append(f"| {sym} | {outcome_str} | {picked} | {avoided} |")
        lines.append("")

    if losers:
        lines.append("## Losers (fell/no recovery)")
        lines.append("")
        lines.append("| Symbol | Outcome | Picked by | Avoided by |")
        lines.append("|--------|---------|-----------|------------|")
        for sym in sorted(losers.keys()):
            p = losers[sym]
            outcome_str = f"{p['outcome']:.2%}"
            picked = ", ".join(p["picked_by"]) if p["picked_by"] else "*(none)*"
            avoided = ", ".join(p["avoided_by"]) if p["avoided_by"] else "*(none)*"
            lines.append(f"| {sym} | {outcome_str} | {picked} | {avoided} |")
        lines.append("")

    # Diagnostic summary
    lines.append("## Diagnostic summary")
    lines.append("")
    all_picked = set()
    all_avoided = set()
    for p in picks.values():
        all_picked.update(p["picked_by"])
        all_avoided.update(p["avoided_by"])

    lines.append(f"**Modes involved:** {', '.join(sorted(all_picked)) if all_picked else '(none)'}")
    lines.append("")

    # Analyze loss patterns
    loss_patterns = {}
    for sym, p in losers.items():
        key = tuple(sorted(p["picked_by"]))
        if key not in loss_patterns:
            loss_patterns[key] = []
        loss_patterns[key].append((sym, p["outcome"]))

    if loss_patterns:
        lines.append("**Loss patterns:**")
        lines.append("")
        for modes, syms in sorted(loss_patterns.items()):
            if not modes:
                diagnosis = "No modes picked this (data error?)"
            elif set(modes) == {"momentum", "rebound", "dividend"}:
                diagnosis = "All modes agreed → shared universe issue (mechanical gates too loose)"
            else:
                diagnosis = "Mixed modes → mode-divergence signal (tighten agent_prompt.md rubric)"

            symbols_str = ", ".join([s for s, _ in syms])
            lines.append(f"- {diagnosis}")
            lines.append(f"  Picked by: {', '.join(modes)}")
            lines.append(f"  Symbols: {symbols_str}")
        lines.append("")

    return "\n".join(lines)
