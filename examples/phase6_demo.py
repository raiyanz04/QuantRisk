import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from greeks.pnl_attribution import hypothetical_pnl
from frtb.actual_pnl import compute_hpl
from backtest.backtest_engine import run_backtest_suite

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)

rtpl = hypothetical_pnl(pf, sens).total     # risk-model (Taylor) stream
hpl = compute_hpl(pf).total                 # full-revaluation stream
print(f"Built: {len(pf)} positions; RTPL/HPL series length {len(rtpl)}")

res = run_backtest_suite(rtpl, hpl, confidence_levels=(0.975, 0.99), window=250)

print()
print("=== COVERAGE TESTS (rolling 250d VaR, full-history evaluation) ===")
print(res.comparison().to_string(index=False))
print()
print("=== CAPITAL MULTIPLIER (trailing 250d, 99% binding) ===")
print(res.summary())
