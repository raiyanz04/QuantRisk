"""Market data fetching layer.

Pulls daily historical data from free sources and returns RiskFactor
objects ready to drop into a RiskFactorSet. Cache files to disk so
re-running during development doesn't hammer APIs.

Sources:
    - Yahoo Finance (yfinance)  : equity spots, VIX, FX
    - ECB SDW                    : euro area AAA yield curve, FX reference rates
    - FRED                       : credit spreads, UST yields

FRED auto-detects FRED_API_KEY env var. With key set, uses the official
API endpoint (full history back to series inception). Without it, falls
back to fredgraph.csv (capped at ~3 years).
"""

from __future__ import annotations
from io import StringIO
from pathlib import Path
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from data.risk_factors import RiskFactor, RiskFactorType


# ---------- Endpoints ----------

ECB_API = "https://data-api.ecb.europa.eu/service/data"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

YC_KEY = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{maturity}Y"
EXR_KEY = "D.{ccy}.EUR.SP00.A"

# FRED series IDs
FRED_BAA_10Y_SPREAD = "BAA10Y"         # Moody's BAA - 10Y Treasury, daily, full history back to 1986
FRED_EURO_HY_OAS = "BAMLHE00EHYIOAS"   # ICE BofA Euro HY OAS — NOTE: restricted to 3y on FRED from April 2026
FRED_US_HY_OAS_ICE = "BAMLH0A0HYM2"    # ICE BofA US HY OAS — same 3y restriction; kept for reference
FRED_UST_2Y      = "DGS2"
FRED_UST_5Y      = "DGS5"
FRED_UST_10Y     = "DGS10"


class MarketDataFetcher:
    """Pulls and caches historical series, returning RiskFactor objects."""

    def __init__(self, cache_dir: str = "data_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, rf_name: str) -> Path:
        return self.cache_dir / f"{rf_name}.pkl"

    def _load_cached(self, rf_name: str) -> RiskFactor | None:
        path = self._cache_path(rf_name)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warnings.warn(f"Cache load failed for {rf_name}: {e}")
            return None

    def _save_cached(self, rf: RiskFactor) -> None:
        with open(self._cache_path(rf.name), "wb") as f:
            pickle.dump(rf, f)

    # ----- Equity (yfinance) -----

    def fetch_equity(
        self, ticker: str, rf_name: str, start: str, end: str, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        if yf is None:
            raise ImportError("yfinance not installed")

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"yfinance returned empty data for {ticker}")

        prices = df["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.dropna().astype(float)
        prices.name = rf_name

        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.EQUITY_SPOT, series=prices)
        self._save_cached(rf)
        return rf

    def fetch_equity_vol_proxy(
        self, start: str, end: str, rf_name: str = "SX5E_RV",
        ewma_lambda: float = 0.94, use_cache: bool = True,
    ) -> RiskFactor:
        """EU equity vol proxy: EWMA realized vol on SX5E (RiskMetrics λ=0.94).
        VSTOXX history is not freely available; this is the standard substitute."""
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        if yf is None:
            raise ImportError("yfinance not installed")

        start_buffer = pd.Timestamp(start) - pd.DateOffset(years=2)
        df = yf.download(
            "^STOXX50E", start=start_buffer.strftime("%Y-%m-%d"), end=end,
            progress=False, auto_adjust=True,
        )
        if df.empty:
            raise RuntimeError("yfinance returned no SX5E data for vol proxy")

        prices = df["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices = prices.dropna().astype(float)

        log_ret = np.log(prices / prices.shift(1)).dropna()
        ewma_var_daily = (log_ret ** 2).ewm(alpha=(1 - ewma_lambda), adjust=False).mean()
        ewma_vol_ann = np.sqrt(ewma_var_daily * 252)

        ewma_vol_ann.index = pd.to_datetime(ewma_vol_ann.index).tz_localize(None)
        series = ewma_vol_ann.loc[start:end].dropna()
        series.name = rf_name

        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.EQUITY_VOL, series=series)
        self._save_cached(rf)
        return rf

    # ----- ECB SDW -----

    @staticmethod
    def _fetch_ecb_csv(dataset: str, key: str, start: str, end: str) -> pd.Series:
        url = f"{ECB_API}/{dataset}/{key}"
        params = {"format": "csvdata", "startPeriod": start, "endPeriod": end}
        r = requests.get(url, params=params, headers={"Accept": "text/csv"}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "TIME_PERIOD" not in df.columns:
            raise RuntimeError(f"ECB SDW returned no data for {dataset}/{key}")
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        return df.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index().astype(float)

    def fetch_ecb_yield(
        self, maturity_years: int, rf_name: str,
        start: str, end: str, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        key = YC_KEY.format(maturity=maturity_years)
        series = (self._fetch_ecb_csv("YC", key, start, end) / 100.0).dropna()
        series.name = rf_name
        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.RATE, series=series)
        self._save_cached(rf)
        return rf

    def fetch_ecb_fx(
        self, ccy: str, rf_name: str, start: str, end: str, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        key = EXR_KEY.format(ccy=ccy)
        series = self._fetch_ecb_csv("EXR", key, start, end).dropna()
        series.name = rf_name
        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.FX_SPOT, series=series)
        self._save_cached(rf)
        return rf

    # ----- FRED (with API key auto-detection) -----

    def _fetch_fred(
        self, series_id: str, start: str | None, end: str | None,
        max_retries: int = 3,
    ) -> pd.Series:
        """Generic FRED fetcher. Auto-detects FRED_API_KEY env var:
          - With key: uses official API endpoint (full history).
          - Without: falls back to fredgraph.csv (capped ~3 years).
        """
        import os
        import time

        api_key = os.environ.get("FRED_API_KEY")

        # ----- Path 1: official API (full history) -----
        if api_key:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
            }
            if start:
                params["observation_start"] = start
            if end:
                params["observation_end"] = end

            last_err: Exception | None = None
            for attempt in range(max_retries):
                try:
                    r = requests.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    obs = r.json().get("observations", [])
                    if not obs:
                        raise RuntimeError(f"FRED API returned no observations for {series_id}")
                    df = pd.DataFrame(obs)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    return df.set_index("date")["value"].sort_index().dropna()
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * (2 ** attempt))
            raise RuntimeError(f"FRED API fetch failed for {series_id}: {last_err}")

        # ----- Path 2: CSV fallback (truncated history) -----
        if not getattr(self, "_fred_csv_warned", False):
            warnings.warn(
                "FRED_API_KEY not set: falling back to fredgraph.csv, which "
                "caps history at ~3 years. Set FRED_API_KEY (free key from "
                "fred.stlouisfed.org/docs/api/api_key.html) for full history."
            )
            self._fred_csv_warned = True

        params_csv: dict[str, str] = {"id": series_id}
        if start:
            params_csv["cosd"] = start
        if end:
            params_csv["coed"] = end

        endpoints: list[tuple[str, dict | None]] = [
            (FRED_CSV, params_csv),
            (f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv", None),
        ]

        last_err = None
        text: str | None = None
        for url, params in endpoints:
            for attempt in range(max_retries):
                try:
                    r = requests.get(url, params=params, timeout=30)
                    r.raise_for_status()
                    text = r.text
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.5 * (2 ** attempt))
            if text is not None:
                break
        if text is None:
            raise RuntimeError(f"FRED CSV fetch failed for {series_id}: {last_err}")

        df = pd.read_csv(StringIO(text))
        date_col, value_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        return df.set_index(date_col)[value_col].sort_index().dropna()

    def fetch_fred_credit_spread(
        self, rf_name: str = "EUR_HY_OAS", series_id: str = FRED_EURO_HY_OAS,
        start: str | None = None, end: str | None = None, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        series = (self._fetch_fred(series_id, start, end) / 100.0)
        series.name = rf_name
        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.CREDIT_SPREAD, series=series)
        self._save_cached(rf)
        return rf

    def fetch_fred_yield(
        self, series_id: str, rf_name: str,
        start: str | None = None, end: str | None = None, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        series = (self._fetch_fred(series_id, start, end) / 100.0)
        series.name = rf_name
        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.RATE, series=series)
        self._save_cached(rf)
        return rf

    # ----- US equity vol and FX -----

    def fetch_vix(
        self, start: str, end: str, rf_name: str = "VIX", use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        if yf is None:
            raise ImportError("yfinance not installed")

        df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError("yfinance returned empty data for ^VIX")
        prices = df["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = (prices.dropna().astype(float) / 100.0)
        prices.name = rf_name

        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.EQUITY_VOL, series=prices)
        self._save_cached(rf)
        return rf

    def fetch_yahoo_fx(
        self, ticker: str, rf_name: str,
        start: str, end: str, use_cache: bool = True,
    ) -> RiskFactor:
        if use_cache and (cached := self._load_cached(rf_name)) is not None:
            return cached
        if yf is None:
            raise ImportError("yfinance not installed")

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"yfinance returned empty data for {ticker}")
        prices = df["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.dropna().astype(float)
        prices.name = rf_name

        rf = RiskFactor(name=rf_name, rf_type=RiskFactorType.FX_SPOT, series=prices)
        self._save_cached(rf)
        return rf


# ---------- High-level builders ----------

def build_eu_risk_factor_set(
    start: str = "2019-01-01",
    end: str | None = None,
    fetcher: MarketDataFetcher | None = None,
    use_cache: bool = True,
) -> "RiskFactorSet":
    from data.risk_factors import RiskFactorSet

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if fetcher is None:
        fetcher = MarketDataFetcher()

    factors: list[RiskFactor] = []

    equity_map = {
        "SAP_SPOT":  "SAP.DE",  "ASML_SPOT": "ASML.AS",
        "TTE_SPOT":  "TTE.PA",  "MC_SPOT":   "MC.PA",
        "ALV_SPOT":  "ALV.DE",
    }
    for rf_name, ticker in equity_map.items():
        factors.append(fetcher.fetch_equity(ticker, rf_name, start, end, use_cache))

    factors.append(fetcher.fetch_equity_vol_proxy(start, end, use_cache=use_cache))

    for maturity in (2, 5, 10):
        factors.append(
            fetcher.fetch_ecb_yield(maturity, f"EUR_{maturity}Y", start, end, use_cache)
        )

    for ccy, rf_name in [("USD", "EURUSD"), ("GBP", "EURGBP"), ("JPY", "EURJPY")]:
        factors.append(fetcher.fetch_ecb_fx(ccy, rf_name, start, end, use_cache))

    factors.append(
        fetcher.fetch_fred_credit_spread(
            rf_name="EUR_HY_OAS", series_id=FRED_EURO_HY_OAS,
            start=start, end=end, use_cache=use_cache,
        )
    )

    return RiskFactorSet(factors)


def build_us_risk_factor_set(
    start: str = "2019-01-01",
    end: str | None = None,
    fetcher: MarketDataFetcher | None = None,
    use_cache: bool = True,
) -> "RiskFactorSet":
    from data.risk_factors import RiskFactorSet

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if fetcher is None:
        fetcher = MarketDataFetcher()

    factors: list[RiskFactor] = []

    equity_map = {
        "AAPL_SPOT":  "AAPL",  "MSFT_SPOT":  "MSFT",
        "NVDA_SPOT":  "NVDA",  "GOOGL_SPOT": "GOOGL",
        "JPM_SPOT":   "JPM",
    }
    for rf_name, ticker in equity_map.items():
        factors.append(fetcher.fetch_equity(ticker, rf_name, start, end, use_cache))

    factors.append(fetcher.fetch_vix(start, end, use_cache=use_cache))

    rate_map = {
        "UST_2Y":  FRED_UST_2Y,
        "UST_5Y":  FRED_UST_5Y,
        "UST_10Y": FRED_UST_10Y,
    }
    for rf_name, series_id in rate_map.items():
        factors.append(
            fetcher.fetch_fred_yield(series_id, rf_name, start, end, use_cache)
        )

    fx_map = {
        "EURUSD":  "EURUSD=X",
        "USDJPY":  "JPY=X",
        "GBPUSD":  "GBPUSD=X",
    }
    for rf_name, ticker in fx_map.items():
        factors.append(fetcher.fetch_yahoo_fx(ticker, rf_name, start, end, use_cache))

    factors.append(
        fetcher.fetch_fred_credit_spread(
            rf_name="US_IG_SPREAD", series_id=FRED_BAA_10Y_SPREAD,
            start=start, end=end, use_cache=use_cache,
        )
    )

    return RiskFactorSet(factors)


# Back-compat alias
build_full_risk_factor_set = build_eu_risk_factor_set
