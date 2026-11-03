import warnings; warnings.filterwarnings("ignore")
from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from frtb.stressed_calibration import compute_stressed_es

rfs = build_us_risk_factor_set(start="2010-01-01", end="2024-12-31", use_cache=True)
pf = build_book(rfs)
sens = build_sensitivities(pf)

print("Running stressed-ES calibration over 15 years...")
s = compute_stressed_es(pf, sens, window_days=250, step_days=5)
print(f"Current ES:     ${s.current_es:>14,.0f}")
print(f"Stressed ES:    ${s.stressed_es:>14,.0f}")
print(f"Amplification:  {s.stress_amplification:.2f}x")
print(f"Worst window:   {s.stressed_window_start.date()} -> {s.stressed_window_end.date()}")

print()
print("Top 5 stress periods:")
top5 = s.es_by_window.nlargest(5)
chg_idx = rfs.changes_matrix().index
for end_date, es in top5.items():
    end_loc = chg_idx.get_loc(end_date)
    start_loc = max(0, end_loc - 249)
    start_date = chg_idx[start_loc].date()
    print(f"  {start_date} -> {end_date.date()} : ${es:>14,.0f}")
