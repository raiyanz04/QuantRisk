import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from stress.scenarios import named_scenarios, from_historical_window
from stress.scenario_engine import run_scenario_suite, apply_scenario

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)
print(f"Built: {len(pf)} positions")

# 1) Prescribed named scenarios (incl. GFC, which pre-dates the data).
scenarios = named_scenarios(rfs)
results, table = run_scenario_suite(pf, sens, scenarios)
print()
print("=== NAMED SCENARIO SUITE (full revaluation, sorted worst-first) ===")
print(table.to_string(index=False))

print()
print("=== DETAIL: worst scenario ===")
print(results[0].summary())

# 2) Data-extracted COVID: validate the prescribed magnitudes against the
#    actual cumulative move (2020-02-19 peak -> 2020-03-23 trough).
print()
print("=== HISTORICAL EXTRACTION: COVID (peak->trough from data) ===")
covid_hist = from_historical_window(rfs, "2020-02-19", "2020-03-23", name="COVID (data)")
res_hist = apply_scenario(pf, sens, covid_hist)
print(res_hist.summary())
print()
print("  resolved shocks (native units):")
print(covid_hist.shocks_native[covid_hist.shocks_native != 0].round(4).to_string())
