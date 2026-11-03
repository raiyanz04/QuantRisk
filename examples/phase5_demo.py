import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, time

from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from greeks.pnl_attribution import hypothetical_pnl
from frtb.actual_pnl import compute_hpl
from frtb.pla_test import run_pla_test
from frtb.sbm_capital import compute_sbm_capital
from frtb.expected_shortfall import compute_frtb_es

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)
print(f"Built: {len(pf)} positions, {rfs.changes_matrix().shape[0]} aligned days")

t0 = time.time()
hpl = compute_hpl(pf)              # full-revaluation stream (FRTB HPL)
rtpl = hypothetical_pnl(pf, sens)  # sensitivities/Taylor stream (FRTB RTPL)
print(f"P&L streams computed in {time.time()-t0:.1f}s")

aligned = pd.concat([rtpl.total.rename("rtpl"), hpl.total.rename("hpl")], axis=1).dropna()
print(f"Pearson corr (RTPL vs HPL): {aligned.corr().iloc[0,1]:+.4f}")
print()

print("=== PLA TEST (RTPL vs HPL) ===")
pla = run_pla_test(rtpl=rtpl.total, hpl=hpl.total)
print(pla.summary())
print()

print("=== SbM CAPITAL ===")
sbm = compute_sbm_capital(pf, sens)
print(f"Total SbM capital: ${sbm.total:,.0f}")
print(sbm.summary().to_string(index=False))
print()

chg = rfs.changes_matrix()
ima = compute_frtb_es(pf, sens, changes=chg.iloc[-250:])
print(f"IMA ES @ 97.5% (10-day base): ${ima.es_aggregated:,.0f}")
print(f"SA  delta charge (binding {sbm.binding_scenario} corr): ${sbm.total:,.0f}")
