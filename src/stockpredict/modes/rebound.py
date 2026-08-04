"""Rebound (mean-reversion bounce) mode: 100% LLM-agent-driven.

The whole mechanically-gated universe (staleness / ceiling-lock / corporate-
action) is handed to the agent, unranked. The agent selects, researches, and
for each pick predicts N (trading days to a profitable exit) and P (the
profit at that exit). Finalize computes ``score = P / N``, ranks by it, and
prices via ``pricing.add_recovery_price_suggestions`` (buy at close, target =
close × (1 + P), no stop — hold until the target).

Mechanically identical to ``momentum`` — the strategies differ only in the
rubric the agent is given (here: find a temporary, healthy dip that will
recover rather than a falling knife). Note there is no coded downtrend filter:
the agent judges dip shape itself from the raw ``mom_20`` / ``high_prox_20`` /
``rsi_14`` columns. Both modes therefore share ``modes/swing.py``; this module
just binds the mode name.
"""
from __future__ import annotations

from . import swing

MODE = "rebound"


def run(**kwargs):
    return swing.run(MODE, **kwargs)


def finalize(plan_path):
    return swing.finalize(MODE, plan_path)
