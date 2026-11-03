"""Phase 9 — regulatory capital aggregation and the risk dashboard.

Closes the loop: consumes VaR/SVaR (Ph 3/7), ES (Ph 4/7), the backtesting
multiplier and IMA gate (Ph 6), and the SbM SA charge (Ph 5) into a single
capital number, Basel 2.5 and FRTB IMA side by side, with a six-panel view.
"""
from capital.regulatory_capital import (
    CapitalResult, compute_regulatory_capital, OUTPUT_FLOOR_PCT,
)
from capital.dashboard import build_dashboard

__all__ = [
    "CapitalResult", "compute_regulatory_capital", "OUTPUT_FLOOR_PCT",
    "build_dashboard",
]
