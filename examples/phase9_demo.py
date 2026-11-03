import warnings; warnings.filterwarnings("ignore")

from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from capital.regulatory_capital import compute_regulatory_capital
from capital.dashboard import build_dashboard

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)
print(f"Built: {len(pf)} positions")

cap = compute_regulatory_capital(pf, sens)
print()
print(cap.summary())
print()
print(cap.comparison().to_string(index=False))

print()
path = build_dashboard(pf, sens, save_path="risk_dashboard.png", capital=cap)
print(f"Dashboard written to {path}")
