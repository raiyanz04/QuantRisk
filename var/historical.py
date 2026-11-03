"""Historical simulation VaR.

The desk's workhorse. Replay each historical day's RF change vector
through today's full Greek set via the Phase 2 hypothetical P&L pipeline.

  - No distributional assumption.
  - Cross-asset correlations baked in as observed (correlation spikes
    during stress preserved).
  - Convexity via the Γ term.

Limitation: VaR can never exceed the worst day in the window — addressed
by Stressed VaR and ES (Phase 4 / Phase 7).
"""

from __future__ import annotations

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from var.types import VaRResult, fill_var_es_from_distribution


def compute_historical_var(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    window_days: int | None = 250,
) -> VaRResult:
    hp = hypothetical_pnl(portfolio, sensitivities)
    pnl = hp.total

    if window_days is not None and len(pnl) > window_days:
        pnl = pnl.iloc[-window_days:]

    result = VaRResult(
        method="historical_simulation",
        n_scenarios=len(pnl),
        pnl_distribution=pnl.rename("hs_pnl"),
    )
    fill_var_es_from_distribution(
        result,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
    )
    return result
