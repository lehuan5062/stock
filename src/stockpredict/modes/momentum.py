"""Momentum (short-term trend-following) mode: 100% LLM-agent-driven.

The whole mechanically-gated universe (staleness / ceiling-lock / corporate-
action) is handed to the agent, unranked. The agent selects, researches, and
for each pick predicts N (trading days to a profitable exit) and P (the
profit at that exit). Finalize computes ``score = P / N``, ranks by it, and
prices via ``pricing.add_recovery_price_suggestions`` (buy at close, target =
close × (1 + P), no stop — hold until the target).

Mechanically identical to ``rebound`` — the strategies differ only in the
rubric the agent is given (here: find an ORGANIC, sustainable uptrend rather
than a pump-and-dump / blow-off top about to reverse). Both therefore share
``modes/swing.py``; this module just binds the mode name.
"""
from __future__ import annotations

from . import swing

MODE = "momentum"


def run(**kwargs):
    return swing.run(MODE, **kwargs)


def finalize(plan_path):
    return swing.finalize(MODE, plan_path)
