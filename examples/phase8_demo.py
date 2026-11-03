import warnings; warnings.filterwarnings("ignore")
import numpy as np

from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from attribution.attribution_engine import run_attribution
from var.parametric import compute_parametric_var
from var.historical import compute_historical_var

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)
print(f"Built: {len(pf)} positions")

rep = run_attribution(pf, sens, confidence=0.99, horizon_days=1, window_days=250)
print()
print(rep.summary())

# Reconciliation: the attribution totals must equal the Phase 3 VaR engines,
# and the component VaRs must sum to the portfolio VaR.
print()
print("=== RECONCILIATION ===")
pv = compute_parametric_var(pf, sens, confidence_levels=(0.99,), horizons_days=(1,))
hv = compute_historical_var(pf, sens, confidence_levels=(0.99,), horizons_days=(1,))
pv_99 = pv.var[(0.99, 1)]
hv_99 = hv.var[(0.99, 1)]
print(f"parametric: attribution {rep.parametric.portfolio_var:,.0f}  vs  "
      f"compute_parametric_var {pv_99:,.0f}  (match={abs(rep.parametric.portfolio_var-pv_99)<1.0})")
print(f"historical: attribution {rep.historical.portfolio_var:,.0f}  vs  "
      f"compute_historical_var {hv_99:,.0f}  (match={abs(rep.historical.portfolio_var-hv_99)<1.0})")
print(f"sum(parametric component) {rep.parametric.component_var.sum():,.0f}  "
      f"vs VaR {rep.parametric.portfolio_var:,.0f}  "
      f"(match={abs(rep.parametric.component_var.sum()-rep.parametric.portfolio_var)<1.0})")
print(f"sum(historical component) {rep.historical.component_var.sum():,.0f}  "
      f"vs VaR {rep.historical.portfolio_var:,.0f}  "
      f"(match={abs(rep.historical.component_var.sum()-rep.historical.portfolio_var)<1.0})")
