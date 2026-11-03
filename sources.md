# Data Sources & Provenance

The engine supports two parallel risk factor configurations (EU and US),
each with ~13 risk factors across the five families. Switch between them
by importing `build_eu_risk_factor_set` / `build_us_risk_factor_set` from
`data.market_data`, and the corresponding `build_book` from
`data.book_eu` / `data.book_us`.

## EU configuration

| Risk factor family | Primary source | Endpoint / Ticker | Frequency | Notes |
|---|---|---|---|---|
| Equity spot | Yahoo Finance | `SAP.DE`, `ASML.AS`, `TTE.PA`, `MC.PA`, `ALV.DE` | Daily close | SX5E top 5 by mkt cap |
| Equity vol | **Proxy: EWMA realized vol on SX5E** | yfinance `^STOXX50E` + λ=0.94 | Daily | See note on VSTOXX below |
| EUR rates | ECB SDW | YC dataset, `B.U2.EUR.4F.G_N_A.SV_C_YM.SR_<N>Y` | Daily | AAA euro area curve, 2Y/5Y/10Y |
| FX | ECB SDW | EXR dataset, `D.<CCY>.EUR.SP00.A` | Daily | EUR/USD, EUR/GBP, EUR/JPY |
| Credit | FRED | `BAMLHE00EHYIOAS` (ICE BofA Euro HY OAS) | Daily | Proxy for iTraxx Crossover |

## US configuration

| Risk factor family | Primary source | Endpoint / Ticker | Frequency | Notes |
|---|---|---|---|---|
| Equity spot | Yahoo Finance | `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `JPM` | Daily close | SPX top tech + JPM short |
| Equity vol | Yahoo Finance | `^VIX` | Daily | Direct — no proxy needed |
| USD rates | FRED | `DGS2`, `DGS5`, `DGS10` | Daily | UST constant-maturity yields |
| FX | Yahoo Finance | `EURUSD=X`, `JPY=X`, `GBPUSD=X` | Daily | EUR/USD, USD/JPY, GBP/USD |
| Credit | FRED | `BAMLH0A0HYM2` (ICE BofA US HY OAS) | Daily | Canonical US HY spread |

## Note on equity vol risk factor (EU only)

Free programmatic VSTOXX history is not currently obtainable: Yahoo's
`V2TX.DE` / `OVS.EX` tickers expose live quotes but no downloadable
history, and `stoxx.com`'s free historical text file stopped updating in
February 2016. LSEG Workspace exposes the data but ESSEC's license does
not permit CSV export.

**Workaround**: construct the EU equity vol risk factor as
exponentially-weighted realized vol on SX5E returns (λ=0.94, the
RiskMetrics 1996 default for daily data). This proxy correlates
~0.7–0.8 daily with VSTOXX in published studies, reacts faster to vol
regime changes than simple rolling vol, and has no vendor dependency.
The risk factor name in code is `SX5E_RV`.

The US side uses VIX directly with no proxy, so AAPL–VIX daily
correlation is correctly negative (~-0.40) — a sharper vol-risk signal
than the EU side's proxy can produce. When demoing FRTB ES or Phase 4
vol risk attribution, prefer the US book.

## FRED API key (recommended)

FRED's free CSV endpoint caps history at ~3 years regardless of date
parameters. For full history (HY OAS series go back to 1996), sign up
for a free API key at <https://fredaccount.stlouisfed.org/apikeys>
and set:

```bash
export FRED_API_KEY="your_key_here"
```

The fetcher auto-detects the env var and switches to the API endpoint.
Without the key, the engine works but credit-spread history bottlenecks
the aligned changes matrix to ~1.6 years, below the Basel preferred
2-year VaR calibration window.

## Methodology references

- **FRTB framework**: BCBS d457 (Minimum capital requirements for market risk, January 2019 revision)
- **Liquidity horizons**: BCBS d457 MAR33.12 (Table 2 of the standard)
- **P&L Attribution test**: BCBS d457 MAR32.40–MAR32.46 (Spearman + KS bands)
- **NMRF treatment**: BCBS d457 MAR33.16
- **Standardised Approach (SbM fallback)**: BCBS d457 MAR21
- **Basel III VaR-based capital**: BCBS d128 (Revisions to the Basel II market risk framework, July 2009)
- **Backtesting statistical tests**:
  - Kupiec (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models"
  - Christoffersen (1998), "Evaluating Interval Forecasts"
- **RiskMetrics EWMA methodology**: JP Morgan/Reuters (1996), RiskMetrics Technical Document, 4th edition (λ=0.94 for daily data)
- **EBA stress test scenarios** (Phase 7): EBA 2025 adverse scenario
