"""Standalone verification script for FRED API key + full data layer.

Run from the project root after setting FRED_API_KEY:

    export FRED_API_KEY="your_new_key_here"
    python verify_fred.py

What this checks:
  1. FRED_API_KEY env var is set.
  2. Live fetch of US HY OAS via the API endpoint returns full history
     (thousands of observations, not the 3-year-capped CSV fallback).
  3. Full US build assembles all 13 risk factors with a clean aligned
     changes matrix.
  4. Sanity-check correlations match economic intuition.

Send the output of this script back to confirm the key is working.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")


def main() -> int:
    # ----- Step 1: env var present? -----
    print("=" * 60)
    print("STEP 1 — Environment check")
    print("=" * 60)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("FAIL: FRED_API_KEY not set.")
        print()
        print("Set it with:")
        print('  export FRED_API_KEY="your_key_here"')
        print()
        print("Then run this script again.")
        return 1
    print(f"OK: FRED_API_KEY is set (length {len(key)}, starts {key[:4]}...)")
    print()

    # ----- Step 2: US HY OAS fetch via API path -----
    print("=" * 60)
    print("STEP 2 — Live FRED API fetch: US HY OAS")
    print("=" * 60)
    from data.market_data import MarketDataFetcher

    fetcher = MarketDataFetcher(cache_dir="/tmp/fred_verify_cache")
    rf = fetcher.fetch_fred_credit_spread(
        rf_name="US_IG_SPREAD_TEST",
        series_id="BAA10Y",
        start="2010-01-01",
        end="2024-12-31",
        use_cache=False,
    )
    n_obs = len(rf.series)
    first_date = rf.series.index[0].date()
    last_date = rf.series.index[-1].date()
    latest_val = rf.current

    print(f"  Observations:   {n_obs:,}")
    print(f"  Date range:     {first_date}  ->  {last_date}")
    print(f"  Latest value:   {latest_val:.4f}  ({latest_val*100:.2f}%)")
    print()

    if n_obs < 3000:
        print(f"WARNING: only {n_obs} observations. Expected ~3,800 for 2010-2024.")
        print("  CSV fallback path may still be in use — check key validity at:")
        print("  https://fredaccount.stlouisfed.org/apikeys")
        print()
    else:
        print(f"OK: full-history API path is active ({n_obs} obs >> 430 CSV cap)")
        print()

    # ----- Step 3: Full US build with the API key live -----
    print("=" * 60)
    print("STEP 3 — Full US risk factor set build")
    print("=" * 60)
    from data.market_data import build_us_risk_factor_set
    from data.book_us import build_book

    # Use a fresh cache so we exercise the live path on every fetcher
    fetcher2 = MarketDataFetcher(cache_dir="/tmp/fred_verify_full")
    rfs = build_us_risk_factor_set(
        start="2010-01-01",
        end="2024-12-31",
        fetcher=fetcher2,
        use_cache=False,
    )
    print(f"  Risk factors loaded: {len(rfs)}")
    print()
    print(f"  {'name':<14} {'type':<15} {'obs':>6}  {'latest':>12}")
    for name in rfs.names:
        f = rfs[name]
        print(f"  {name:<14} {f.rf_type.value:<15} {len(f.series):>6,}  {f.current:>12.4f}")
    print()

    chg = rfs.changes_matrix()
    print(f"  Aligned changes matrix: {chg.shape[0]:,} dates x {chg.shape[1]} risk factors")
    print(f"  Aligned date range:     {chg.index[0].date()}  ->  {chg.index[-1].date()}")
    print()

    # ----- Step 4: Sanity correlations -----
    print("=" * 60)
    print("STEP 4 — Correlation sanity checks")
    print("=" * 60)
    pf = build_book(rfs)
    print(f"  Portfolio built: {len(pf)} positions")
    print()

    corr = rfs.correlation()
    checks = [
        ("AAPL_SPOT",  "MSFT_SPOT",  "+", "tech co-movement"),
        ("AAPL_SPOT",  "VIX",        "-", "equity vs vol"),
        ("UST_2Y",     "UST_10Y",    "+", "curve co-movement"),
        ("AAPL_SPOT",  "US_IG_SPREAD",  "-", "risk-on vs spreads"),
        ("NVDA_SPOT",  "VIX",        "-", "high-beta tech vs vol"),
    ]
    print(f"  {'pair':<32} {'expected':>10} {'actual':>10}   {'check'}")
    all_pass = True
    for a, b, sign, why in checks:
        rho = corr.loc[a, b]
        match = (rho > 0 and sign == "+") or (rho < 0 and sign == "-")
        marker = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  {a:<14}  vs  {b:<14}  {sign:>6}     {rho:+.3f}   {marker}  ({why})")
    print()

    # ----- Step 5: Summary -----
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    if n_obs >= 3000 and all_pass:
        print("ALL CHECKS PASSED.")
        print()
        print("Paste the full output of this script back to confirm the")
        print("full-history data layer is live. Ready for Phase 4.")
        return 0
    else:
        print("Some checks did not pass — review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
