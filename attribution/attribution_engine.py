"""Attribution facade: parametric + historical decompositions side by side,
plus concentration, mirroring var_engine.run_all_methods.

The two estimators answer the same question under different distributional
assumptions, and the gap is itself informative: where the historical
component VaR exceeds the parametric one, the empirical tail loads more on
that position than a Gaussian would predict (fat tails, or option convexity
biting in the tail that the delta-normal decomposition omits).
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from attribution.component_var import (
    VaRAttribution, parametric_var_attribution, historical_var_attribution,
)
from attribution.concentration import ConcentrationMetrics, concentration


@dataclass
class AttributionReport:
    confidence: float
    horizon_days: int
    parametric: VaRAttribution
    historical: VaRAttribution
    concentration: ConcentrationMetrics

    def comparison(self) -> pd.DataFrame:
        """Component VaR per position, parametric vs historical."""
        df = pd.DataFrame({
            "param_component": self.parametric.component_var.round(0),
            "param_pct":       (self.parametric.pct_contribution * 100).round(1),
            "hist_component":  self.historical.component_var.round(0),
            "hist_pct":        (self.historical.pct_contribution * 100).round(1),
            "marginal_/1m":    self.parametric.marginal_var_per_1m.round(0),
        })
        return df.sort_values("param_component", ascending=False)

    def summary(self) -> str:
        return (
            f"=== VaR ATTRIBUTION @ {self.confidence:.1%}, {self.horizon_days}d "
            f"(parametric VaR {self.parametric.portfolio_var:,.0f} | "
            f"historical VaR {self.historical.portfolio_var:,.0f}) ===\n"
            + self.comparison().to_string()
            + "\n\n" + self.concentration.summary()
        )


def run_attribution(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence: float = 0.99,
    horizon_days: int = 1,
    window_days: int | None = 250,
) -> AttributionReport:
    param = parametric_var_attribution(
        portfolio, sensitivities, confidence, horizon_days, window_days)
    hist = historical_var_attribution(
        portfolio, sensitivities, confidence, horizon_days, window_days)
    conc = concentration(param)
    return AttributionReport(
        confidence=confidence, horizon_days=horizon_days,
        parametric=param, historical=hist, concentration=conc,
    )
