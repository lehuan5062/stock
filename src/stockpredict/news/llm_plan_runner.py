"""Build/parse the markdown plan for the LLM-agent-only pipeline, shared by
all three strategies (momentum / rebound / dividend).

There is NO machine-learning model anywhere in this program: the candidates
handed to the agent are the WHOLE mechanically-gated universe (see
``selector.eligible_universe``) — unranked. The agent's job is to select,
research, rank and price every name itself:

  * momentum / rebound: predict **N** (trading days to a profitable exit) and
    **P** (the profit at that exit, a decimal return fraction). Finalize
    computes ``score = P / N`` and ranks by it. Buy at today's close, target =
    ``close * (1 + P)``, no stop (hold until the target).
  * dividend: judge payout sustainability from the deterministic dividend
    fetcher's numbers and predict ``expected_hold_years`` + a confidence.
    There is no N/P/target — this is a long hold, not a swing trade.

Workflow:
  1. ``selector.eligible_universe`` produces the filtered, UNCAPPED universe
     (no scoring)
  2. ``write_llm_plan(mode, ...)`` emits a universe reference table +
     instructions + an empty results table shaped for that mode
  3. The agent researches, selects, ranks, prices (WebSearch / WebFetch)
  4. ``parse_llm_plan(mode, ...)`` reads the filled results table + per-pick
     sections
  5. ``modes.<mode>.finalize`` ranks, applies prices, writes the picks JSON

The per-ticker section format (``### TICKER — Company`` with Step 1 / 2 / 4)
and the ``[dimension-tag]`` convention are shared across modes so one set of
parsing helpers below covers all three.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from ..config import reports_dir
from .sources import global_urls, vn_urls

# ---------------------------------------------------------------------------
# Shared per-ticker section parsing (moved from the retired hybrid
# ``claude_runner.py`` — the section/step/tag conventions are identical
# across all three modes, so one implementation covers all of them).
# ---------------------------------------------------------------------------

_SYM_RE = re.compile(r"^[A-Z0-9]{2,8}$")
_NUM_RE = re.compile(r"^[+\-]?\d+(?:\.\d+)?$")

# Dimension tags the agent writes in the form "[tag-name]" at the start of
# each Step 4 bullet. Extracted so the ledger can later aggregate hit-rate by
# dimension category and feed it back into the next run's prompt.
_DIMENSION_TAG_RE = re.compile(r"\[([a-z0-9][a-z0-9_/.+-]*)\]", re.IGNORECASE)


def split_per_ticker_sections(text: str) -> dict[str, str]:
    """Slice the plan markdown into a {ticker: section_text} dict using the
    ``### TICKER  —  Company`` headings emitted by ``write_llm_plan``."""
    sections: dict[str, str] = {}
    current_sym: str | None = None
    current_lines: list[str] = []
    heading_re = re.compile(r"^###\s+([A-Z0-9]{2,8})\s+(?:—|--).*$")
    for line in text.splitlines():
        m = heading_re.match(line)
        if m:
            if current_sym:
                sections[current_sym] = "\n".join(current_lines)
            current_sym = m.group(1)
            current_lines = []
            continue
        if current_sym is not None:
            # Stop accumulating at ANY level-2 heading — the per-ticker
            # sections only ever contain `###` headings and `**Step N**`
            # labels, so a `## ` line marks a global footer (the Results
            # table). Matching only that here would let a section swallow
            # the seed-source list and the whole Results table into its
            # findings.
            if line.strip().startswith("## "):
                sections[current_sym] = "\n".join(current_lines)
                current_sym = None
                current_lines = []
                continue
            current_lines.append(line)
    if current_sym:
        sections[current_sym] = "\n".join(current_lines)
    return sections


def extract_step(section: str, step_label: str) -> str:
    """Pull the user-written text under a `**Step N — <label>**` heading.
    Returns the joined non-empty bullet/paragraph lines, stripped."""
    pat = re.compile(
        rf"\*\*Step \d+ — {re.escape(step_label)}\*\*[^\n]*\n(.*?)(?=\*\*Step \d+ —|\Z)",
        re.DOTALL,
    )
    m = pat.search(section)
    if not m:
        return ""
    body = m.group(1)
    out_lines: list[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s == "-":
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        out_lines.append(s)
    return " ".join(out_lines).strip()


def extract_findings_list(section: str) -> list[str]:
    """Same as ``extract_step('Findings')`` but keep each bullet as a list
    item."""
    pat = re.compile(
        r"\*\*Step \d+ — Findings\*\*[^\n]*\n(.*?)(?=\*\*Step \d+ —|\Z)",
        re.DOTALL,
    )
    m = pat.search(section)
    if not m:
        return []
    out: list[str] = []
    for raw in m.group(1).splitlines():
        s = raw.strip()
        if not s or s == "-":
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        out.append(s)
    return out


def extract_dimension_tags(findings: list[str]) -> list[str]:
    """Pull `[tag-name]` markers out of Step 4 bullets, deduped, lower-cased,
    in first-seen order. See the module docstring of the retired
    ``claude_runner`` for the full tagging convention."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for bullet in findings:
        for m in _DIMENSION_TAG_RE.finditer(bullet):
            tag = m.group(1).strip().lower()
            if not tag or tag.isdigit() or tag in {"drop", "net"}:
                continue
            if tag in seen_set:
                continue
            seen_set.add(tag)
            seen.append(tag)
    return seen


# Backward-compatible aliases. These four helpers were private until the
# momentum/rebound engine was unified; ``modes.dividend`` reached across the
# module boundary for them, which is what prompted publicizing the names.
# Existing callers (notably ``tests/test_dimension_tracking.py``) keep working.
_split_per_ticker_sections = split_per_ticker_sections
_extract_step = extract_step
_extract_findings_list = extract_findings_list
_extract_dimension_tags = extract_dimension_tags


def _parse_price_cell(cell: str) -> float:
    """Parse an optional VND price cell into a float. Blank / '-' /
    unparseable -> NaN."""
    s = (cell or "").strip().replace(",", "").replace("đ", "").replace("VND", "").strip()
    if not s or s == "-":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_profit_cell(cell: str) -> float:
    """Parse the P cell into a decimal return fraction. Accepts ``0.05`` or
    ``5%`` (percent form divided by 100). Blank / unparseable -> NaN."""
    s = (cell or "").strip().replace(",", "")
    if not s or s == "-":
        return float("nan")
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v / 100.0 if pct else v


def parse_num_cell(cell: str) -> float:
    """Strict numeric cell parse: returns NaN unless the cell is a plain
    number. Used for the gate field of a results table (see ``ResultField``)."""
    s = (cell or "").strip()
    if not _NUM_RE.match(s):
        return float("nan")
    return float(s)


# ---------------------------------------------------------------------------
# Shared results-table parsing
#
# Every mode's results table is ``| rank | symbol | <field> ... |``; the modes
# differ only in which fields follow ``symbol``. ``ResultField`` describes one
# such field so a single scan loop covers all of them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultField:
    """One agent-filled cell of a results table.

    ``header`` is the markdown column label (documentation only — parsing is
    positional, matching the original per-mode parsers). ``out_col`` is the
    DataFrame column produced. ``parse`` maps the raw cell text to a float.

    ``drop_sentinel`` marks the ONE field that carries the ``DROP`` keyword and
    doubles as the row gate: if it is present but doesn't parse to a number,
    the whole row is skipped. That is what makes an unfilled template row
    (blank cells) parse to nothing rather than to a NaN pick.

    ``required=False`` marks a trailing field that older plan files predate.
    It is excluded from the minimum-cell-count check and reads as an empty
    cell when absent, so a plan emitted before the field existed still
    finalizes.
    """
    header: str
    out_col: str
    parse: Callable[[str], float]
    drop_sentinel: bool = False
    required: bool = True


def parse_results_table(path: str | Path, fields: Sequence[ResultField]
                        ) -> pd.DataFrame:
    """Read a filled plan's ``## Results`` table plus its per-ticker research
    sections.

    Returns DataFrame[symbol, dropped, *<field.out_col for field in fields>,
    business, dimensions, key_news, dimensions_cited]. Rows are deduped by
    symbol, first occurrence winning.
    """
    text = Path(path).read_text(encoding="utf-8")
    sections = split_per_ticker_sections(text)
    min_cells = 2 + sum(1 for f in fields if f.required)

    rows = []
    in_results = False
    seen: set[str] = set()
    for line in text.splitlines():
        if line.strip().startswith("## Results"):
            in_results = True
            continue
        if not in_results:
            continue

        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < min_cells:
            continue
        sym = cells[1].upper()
        if not _SYM_RE.match(sym) or sym in seen:
            continue

        values: dict[str, float] = {}
        dropped = False
        reject = False
        for i, field in enumerate(fields):
            raw = cells[2 + i] if len(cells) > 2 + i else ""
            if field.drop_sentinel and raw.strip().upper() == "DROP":
                dropped = True
                values = {f.out_col: float("nan") for f in fields}
                break
            value = field.parse(raw)
            if field.drop_sentinel and pd.isna(value):
                reject = True
                break
            values[field.out_col] = value
        if reject:
            continue

        seen.add(sym)
        sec = sections.get(sym, "")
        findings = extract_findings_list(sec)
        rows.append({
            "symbol": sym,
            "dropped": dropped,
            **values,
            "business": extract_step(sec, "Business"),
            "dimensions": extract_step(sec, "Research dimensions")
                          or extract_step(sec, "Key drivers"),
            "key_news": findings,
            "dimensions_cited": ",".join(extract_dimension_tags(findings)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared universe-table rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One universe-reference-table column: source frame key, markdown header,
    and a cell formatter. ``fmt`` must render NaN/None as an empty string."""
    key: str
    header: str
    fmt: Callable[[object], str]


def fmt_text(v: object) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def fmt_symbol(v: object) -> str:
    return str(v).upper()


def fmt_type(v: object) -> str:
    s = fmt_text(v).strip()
    return (s or "STOCK").upper()


def fmt_int(v: object) -> str:
    return f"{v:.0f}" if pd.notna(v) else ""


def fmt_signed3(v: object) -> str:
    return f"{v:+.3f}" if pd.notna(v) else ""


def fmt_comma(v: object) -> str:
    return f"{v:,.0f}" if pd.notna(v) else ""


def fmt_pct2(v: object) -> str:
    return f"{v:.2%}" if pd.notna(v) else ""


def render_universe_table(universe: pd.DataFrame,
                          columns: Sequence[Column]) -> list[str]:
    """Render the UNRANKED universe reference table as markdown lines (header,
    separator, one row per symbol). Pipe characters inside any cell are
    replaced with ``/`` so a company name can't break the table."""
    lines = [
        "| " + " | ".join(c.header for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in universe.iterrows():
        cells = [c.fmt(row.get(c.key, float("nan"))).replace("|", "/")
                 for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# The momentum/rebound universe reference table. Both modes show the same
# columns — the strategies differ only in the rubric text below.
SWING_UNIVERSE_COLUMNS = (
    Column("symbol", "symbol", fmt_symbol),
    Column("organ_name", "company", fmt_text),
    Column("close", "close", fmt_int),
    Column("rsi_14", "rsi_14", fmt_int),
    Column("mom_5", "mom_5", fmt_signed3),
    Column("mom_20", "mom_20", fmt_signed3),
    Column("high_prox_20", "high_prox_20", fmt_signed3),
    Column("adv_vnd_20", "adv_vnd_20", fmt_comma),
    Column("history_days", "history_days", fmt_int),
    Column("instrument_type", "type", fmt_type),
)


# ---------------------------------------------------------------------------
# Mode-specific rubric text + results-table shape
# ---------------------------------------------------------------------------

_RUBRIC = {
    "momentum": {
        "verb": "trend-follow",
        "method_lines": [
            "This is a **momentum** (short-term trend-following) pick: you are",
            "looking for names with an ORGANIC, sustainable uptrend — real demand",
            "or a real catalyst — as opposed to a pump-and-dump / blow-off top",
            "that is about to reverse. The table below is the ENTIRE",
            "mechanically-gated universe (staleness / ceiling-lock / corporate-",
            "action filtered) — it is NOT ranked. Judge liquidity (`adv_vnd_20`),",
            "tradability (`close`), and overbought risk (`rsi_14`) yourself from",
            "the raw columns.",
        ],
        "vet_lines": [
            "**Vet the trend**: is the move backed by real volume, a genuine",
            "catalyst, or fundamentals — or is it a thin, low-liquidity spike that",
            "looks about to reverse (pump-and-dump / blow-off top)? Favor the",
            "former; DROP or avoid the latter even if momentum columns look strong.",
        ],
    },
    "rebound": {
        "verb": "rebound",
        "method_lines": [
            "This is a **rebound** (mean-reversion bounce) pick: you are looking",
            "for names in a temporary, healthy dip that will recover — not a",
            "falling knife (fraud, delisting, insolvency, structural decline). The",
            "table below is the ENTIRE mechanically-gated universe (staleness /",
            "ceiling-lock / corporate-action filtered) — it is NOT ranked. Judge",
            "liquidity (`adv_vnd_20`), tradability (`close`), downtrend shape",
            "(`mom_20`, `high_prox_20`) and oversold/overbought state (`rsi_14`)",
            "yourself from the raw columns — there is no coded downtrend filter",
            "any more.",
        ],
        "vet_lines": [
            "**Vet the bounce**: healthy dip that will bounce, or falling knife",
            "(fraud, delisting, insolvency, structural decline)? Confirm the",
            "healthy ones, `DROP` the un-tradeable ones outright.",
        ],
    },
}


#: The modes ``write_llm_plan`` / ``parse_llm_plan`` can serve — i.e. the ones
#: driven by ``modes/swing.py``. Public so callers don't reach into ``_RUBRIC``.
SWING_MODES = tuple(_RUBRIC)


def write_llm_plan(mode: str, universe: pd.DataFrame, on: dt.date | None = None,
                   run_signature: str | None = None,
                   n_picks: int = 5) -> Path:
    """Emit the LLM-agent-only markdown plan for ``mode in {'momentum',
    'rebound'}``. (Dividend has its own writer — see ``modes.dividend`` — the
    results shape is different enough it isn't worth shoehorning here.)

    ``universe`` is the full eligible cross-section from
    ``selector.eligible_universe`` — includes plain liquidity/technical
    columns (``adv_vnd_20``, ``adv_active_days_20``, ``close``, ``rsi_14``,
    ``mom_5``, ``mom_20``, ``high_prox_20``, ``history_days``) that used to be
    coded gates; the agent now reasons over them directly.
    """
    if mode not in _RUBRIC:
        raise ValueError(f"write_llm_plan: unsupported mode {mode!r}")
    rubric = _RUBRIC[mode]
    on = on or dt.date.today()
    out_dir = reports_dir()
    stem = f"{mode}_plan_{on.isoformat()}"
    if run_signature:
        path = out_dir / f"{stem}_{run_signature}.md"
    else:
        path = out_dir / f"{stem}.md"

    from .company_info import enrich
    universe = enrich(universe)

    lines = [
        f"# {mode.title()} pick plan — {on.isoformat()}",
        "",
        "## Method — YOU pick everything (no ML ranking, no coded gate beyond",
        "## staleness / ceiling-lock / corporate-action)",
        "",
        *rubric["method_lines"],
        "",
        f"1. **Select** the best **{int(n_picks)}** name(s) from the universe",
        "   table, using your own research — fundamentals, news, sector, macro,",
        "   technicals, liquidity. You may research as many candidates as you",
        "   need.",
        "2. **Predict N and P** for each chosen name, from your research:",
        "   `N_days` = expected TRADING days until it reaches a profitable",
        "   point; `P` = the expected profit at that point, as a decimal return",
        "   fraction (e.g. `0.05` = +5%; `5%` also accepted). P should clear the",
        "   round-trip fee bar (~0.95%) or the pick is flagged weak.",
        "3. Finalize computes `score = P / N` (profit per day held) and **ranks",
        "   your picks by it**. You BUY AT TODAY'S CLOSE (the `close` column,",
        "   ×1000 for VND); the sell target is `close × (1 + P)`. No entry",
        "   price, no stop — hold until the target.",
        "4. For each chosen name, write a `### TICKER — Company` section",
        "   (template below) documenting the business, the dimensions you",
        "   researched, and your findings — then fill the results table at the",
        "   bottom.",
        "",
        *rubric["vet_lines"],
        "",
        "**Hard override**: if you find a delisting / trading halt / bankruptcy",
        "filing, do NOT pick the name (or write `DROP` in its `N_days` cell).",
        "",
        "## Global / macro context (read once, before picking)",
        "",
        "**Major-conflict / geopolitical check.** Scan for major global conflicts "
        "or geopolitical shocks breaking today (wars, ceasefires, sanctions / "
        "tariffs, oil-supply or shipping disruptions, sharp oil / gold / USD-VND "
        "moves). A market-wide catalyst can move the whole VN-Index and specific "
        "sectors regardless of any single company; carry it into every pick.",
        "",
        "**VN-Index trend call.** Research where the VN-Index is likely headed "
        "over the expected holding window (a few days to a couple of weeks): "
        "recent trend / momentum, position vs 50/200-day MAs, breadth, foreign "
        "net buy/sell, scheduled macro events. State an explicit UP / SIDEWAYS "
        "/ DOWN view with confidence and let it tilt every pick.",
        "",
    ]
    for name, url in global_urls().items():
        lines.append(f"- [{name}]({url})")
    lines += [
        "",
        "## Universe (UNRANKED — the full mechanically-gated set)",
        "",
        *render_universe_table(universe, SWING_UNIVERSE_COLUMNS),
    ]

    lines += [
        "",
        "## Per-pick research sections",
        "",
        "For EACH name you choose, add a section in this exact format (the",
        "finalize parser keys on the `### TICKER` heading and the `**Step N —`",
        "labels). Use WebFetch / WebSearch, cross-check at least 2 sources, and",
        "tag each finding with `[dimension]`.",
        "",
        "Seed sources you can reuse per ticker (replace TICKER):",
        "",
    ]
    for name, url in vn_urls("TICKER").items():
        lines.append(f"- [{name}]({url})")

    lines += [
        "",
        "Section template — append your filled sections directly below it,",
        "immediately before `## Results`:",
        "",
        "```",
        "### TICKER  —  Company name",
        "",
        "**Step 1 — Business**: one line on what the company does.",
        "- ",
        "",
        f"**Step 2 — Research dimensions**: the 3-7 drivers you judged matter for",
        f"THIS ticker's {rubric['verb']}.",
        "- ",
        "",
        "**Step 4 — Findings** (one bullet per dimension, tagged `[dimension-name]`,",
        "with dates + sources):",
        "- ",
        "```",
        "",
        "## Results — fill this with your chosen picks",
        "",
        "One row per pick, ordered however you like (the finalize step computes",
        "`score = P / N` and re-sorts, highest first). `N_days` = expected trading",
        "days to the profitable exit (>= 1); `P` = expected profit as a decimal",
        "fraction (`0.05` = +5%; a `5%` cell is also accepted). You buy at the",
        "close; the target is `close × (1 + P)`; no stop. Write `DROP` in",
        "`N_days` to exclude a row you listed.",
        "",
        "| rank | symbol | N_days | P |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(int(n_picks)):
        lines.append(f"| {i + 1} |  |  |  |")
    lines += [
        "",
        "When done, run:",
        f"  `python -m stockpredict.cli finalize reports/{path.name}`",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


SWING_RESULT_FIELDS = (
    ResultField("N_days", "pred_days", parse_num_cell, drop_sentinel=True),
    ResultField("P", "pred_profit", _parse_profit_cell),
)


def parse_llm_plan(path: str | Path) -> pd.DataFrame:
    """Read the filled momentum/rebound plan and return DataFrame[symbol,
    dropped, pred_days, pred_profit, business, dimensions, key_news,
    dimensions_cited]. Entry is the close and the target is
    ``close × (1 + pred_profit)``, both set at finalize; there is no stop.
    ``dropped=True`` marks a DROP row."""
    return parse_results_table(path, SWING_RESULT_FIELDS)
