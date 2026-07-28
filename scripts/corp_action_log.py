"""Summarize cache/corp_action_events.csv -- the log of SuspectedCorporateAction-
Artifact guard trips written by data.cache.log_corp_action_event() each time an
incremental fetch's move exceeds a symbol's exchange band + margin and forces a
full re-fetch.

Groups by symbol so recurring or non-healing trips (healed=False, or the same
symbol firing across many runs) can be spotted -- that's the "genuine bug"
signal instead of an expected one-off corporate-action artifact. Read-only.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stockpredict.data.cache import _corp_action_log_path  # noqa: E402


def main() -> None:
    path = _corp_action_log_path()
    if not path.exists():
        print(f"No log yet at {path} -- no guard trips recorded.")
        return

    rows_by_symbol: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_symbol[row["symbol"]].append(row)

    if not rows_by_symbol:
        print(f"{path} exists but has no rows.")
        return

    print(f"{path}\n")
    header = f"{'symbol':<10}{'count':>6}{'healed':>8}{'unhealed':>10}  last_seen"
    print(header)
    print("-" * len(header))
    for symbol, rows in sorted(rows_by_symbol.items(), key=lambda kv: -len(kv[1])):
        healed = sum(1 for r in rows if r["healed"] == "True")
        unhealed = len(rows) - healed
        last_seen = max(r["timestamp"] for r in rows)
        flag = "  <-- check this" if unhealed > 0 or len(rows) > 1 else ""
        print(f"{symbol:<10}{len(rows):>6}{healed:>8}{unhealed:>10}  {last_seen}{flag}")


if __name__ == "__main__":
    main()
