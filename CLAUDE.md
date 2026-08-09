# Rules for Claude in this repo

## Git
- **NEVER run `git commit` unless the user's latest message explicitly says to commit.**
  Implementing a change, verifying it works, or the user approving the *result* is NOT
  permission to commit. Finish the work, then ask "Ready to commit?" and wait for an
  explicit yes. This rule has been violated repeatedly — treat it as a hard stop.
- Commit on the currently checked-out branch (e.g. `master`); do not auto-create
  feature branches.
- Never `git restore` / discard uncommitted changes without asking first.

## Data
- Never run a bare `update-data` (full-universe refetch). Use `evaluate` or
  `update-data -s <SYM>` for specific symbols.
- Same rule for dividends: never run a bare `update-dividends`. Use
  `update-dividends -s <SYM>` for specific symbols.
- `reports/` is gitignored output-only — never put scripts there; one-off analysis
  scripts go in `scripts/`.

## Architecture (2026-07-27 refactor)
- The program is 100% LLM-agent-driven — no ML/DL model anywhere in the live
  path. Three modes: `momentum`, `rebound`, `dividend` (see `agent_prompt.md`
  and `README.md`). `self_correct_prompt.md` and `claude_prompt.md` were
  retired; `agent_prompt.md` is the only prompt file now.
- `momentum` and `rebound` are mechanically IDENTICAL — same universe gate,
  same N/P prediction, same `score = P/N`, same pricing. They differ ONLY in
  the rubric paragraph in `_RUBRIC` (`news/llm_plan_runner.py`). Both run on
  `modes/swing.py`; `modes/momentum.py` and `modes/rebound.py` are shims that
  bind the mode name. Put shared changes in `swing.py`, strategy wording in
  `_RUBRIC`. `dividend` is genuinely separate (own plan writer, results shape,
  pricing and scoring).
- There is NO coded downtrend filter and no recovery-probability model. The
  agent judges dip/trend shape from the raw `mom_20` / `high_prox_20` /
  `rsi_14` columns.
- `dividend` mode is about the DIVIDEND, not about timing an entry. It ranks on
  the agent's FORWARD forecast (`fwd_dps_vnd` → `pred_forward_yield ×
  confidence`); trailing yield is an input, never the ranking objective. Do not
  reintroduce price/technical columns (rsi/momentum) into its universe table —
  that was tried and reverted as rebound-flavoured drift.
- The dividend cache has outlived two schema rewrites.
  `data.dividends.read_dividend_history` normalizes ALL known on-disk shapes to
  `CANONICAL_COLUMNS`; never re-add a strict schema guard that returns empty on
  an unrecognised file, that silently hid ~148 of 150 symbols. Both `DIV` and
  `ISS` events are kept (`kind` = cash/stock/rights/placement, classified from
  `event_title_en` — VCI's ISS payload has no price field). Income metrics must
  be computed from `kind == "cash"` rows ONLY, and never sum `stock` with
  `placement`: shares handed to holders and shares sold to third parties are
  different facts.
- The ledger (`predictions.parquet`) persists `rank` but NOT `score`. If you
  change how a mode ranks, you MUST bump its `score_formula` stamp (see
  `modes.dividend.SCORE_FORMULA`) or historical `rank` rows silently become
  incomparable and `compare-modes` will blend two strategies under one label.
