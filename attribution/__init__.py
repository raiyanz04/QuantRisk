"""Phase 8 — risk attribution & concentration.

Turns aggregate VaR into a management tool: component VaR (where the risk
lives), marginal VaR (what a trade does to it), and concentration limits.
"""
from attribution.component_var import (
    VaRAttribution, parametric_var_attribution, historical_var_attribution,
)
from attribution.concentration import ConcentrationMetrics, concentration
from attribution.attribution_engine import AttributionReport, run_attribution

__all__ = [
    "VaRAttribution", "parametric_var_attribution", "historical_var_attribution",
    "ConcentrationMetrics", "concentration",
    "AttributionReport", "run_attribution",
]
