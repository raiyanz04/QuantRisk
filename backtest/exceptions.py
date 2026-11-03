"""Rolling VaR exception generation for backtesting.

A backtest needs a TIME SERIES of (VaR forecast, next-day realized P&L)
pairs, not a single VaR number. This module produces that series by
rolling a fixed estimation window across the P&L history: for each day
t, VaR_t is estimated from the window ending at t-1, then compared with
the realized P&L on day t. An exception (breach) is a day on which the
loss exceeds the forecast VaR.

Two design points that matter:

  1. One VaR model, multiple realized streams. Under FRTB the desk's
     single 1-day VaR is tested against BOTH actual and hypothetical
     P&L. We mirror that: the VaR forecast is estimated from one source
     series (the risk model's P&L) and can be tested against a different
     realized series. Build the VaR from RTPL and test it against HPL
     and you directly see the non-linearity gap — HPL breaches the
     RTPL-calibrated VaR precisely where the Taylor model under-prices
     the tail. With var_method="historical" the forecaster reproduces
     the Phase 3 historical-simulation VaR on a rolling basis.

  2. Estimation window vs evaluation window. The full exception series
     (everything after the first estimation window) feeds the
     statistical tests, where more observations mean more power. The
     regulatory traffic light instead looks only at the most recent 250
     exceptions. Both come from the same series; see traffic_light.py.

Sign convention: VaR is a positive loss number (consistent with the rest
of the engine). An exception is realized_pnl < -VaR.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


VAR_METHODS = ("historical", "gaussian")


@dataclass
class ExceptionStream:
    """Rolling-window exception series for one (realized stream, confidence)."""
    name: str                       # label, e.g. "HPL @ 99.0%"
    confidence: float
    window: int
    var_method: str
    var_forecast: pd.Series         # VaR_t (positive loss), indexed by date
    realized: pd.Series             # realized P&L on day t (aligned)
    exceptions: pd.Series           # 0/1 indicator, aligned
    n_obs: int
    n_exceptions: int

    @property
    def expected_rate(self) -> float:
        return 1.0 - self.confidence

    @property
    def observed_rate(self) -> float:
        return self.n_exceptions / self.n_obs if self.n_obs else float("nan")

    def recent(self, days: int = 250) -> "ExceptionStream":
        """A view over the most recent `days` observations (the regulatory
        evaluation window), returned as a new ExceptionStream."""
        exc = self.exceptions.iloc[-days:]
        return ExceptionStream(
            name=self.name, confidence=self.confidence, window=self.window,
            var_method=self.var_method,
            var_forecast=self.var_forecast.iloc[-days:],
            realized=self.realized.iloc[-days:],
            exceptions=exc,
            n_obs=len(exc), n_exceptions=int(exc.sum()),
        )

    def summary(self) -> pd.Series:
        return pd.Series({
            "stream":        self.name,
            "confidence":    self.confidence,
            "n_obs":         self.n_obs,
            "n_exceptions":  self.n_exceptions,
            "expected":      self.expected_rate * self.n_obs,
            "observed_rate": self.observed_rate,
            "expected_rate": self.expected_rate,
        })


def _forecast_var(window_pnl: np.ndarray, confidence: float, method: str) -> float:
    """One-step VaR forecast from a window of P&L (positive loss number)."""
    alpha = 1.0 - confidence
    if method == "historical":
        return -float(np.quantile(window_pnl, alpha))
    elif method == "gaussian":
        mu = float(window_pnl.mean())
        sd = float(window_pnl.std(ddof=1))
        return -float(mu + sd * norm.ppf(alpha))   # = -mu + sd * z(confidence)
    raise ValueError(f"Unknown var_method {method!r}; use one of {VAR_METHODS}")


def generate_exceptions(
    realized: pd.Series,
    confidence: float,
    window: int = 250,
    var_method: str = "historical",
    var_source: pd.Series | None = None,
    name: str | None = None,
) -> ExceptionStream:
    """Roll a `window`-day VaR forecast across the history and flag breaches.

    Parameters
    ----------
    realized
        The realized daily P&L stream being tested (e.g. HPL or RTPL).
    confidence
        VaR confidence (FRTB backtests at 0.975 and 0.99).
    window
        Estimation window length in days (rolling).
    var_method
        "historical" (empirical quantile) or "gaussian" (mean/vol).
    var_source
        Series used to ESTIMATE the VaR. Defaults to `realized` (a clean
        self-backtest). Pass the risk-model series (RTPL) here while
        testing against HPL to reproduce the FRTB "one VaR, two streams"
        setup and surface the non-linearity gap.
    name
        Optional label for reports.

    Returns
    -------
    ExceptionStream over all days after the first estimation window.
    """
    if var_source is None:
        var_source = realized

    # Align the two series on common dates so VaR_t and realized_t refer
    # to the same calendar day.
    df = pd.concat(
        [var_source.rename("src"), realized.rename("real")], axis=1
    ).dropna()
    src = df["src"].values
    real = df["real"].values
    idx = df.index
    n = len(df)

    if n <= window:
        raise ValueError(f"Need more than window={window} observations; got {n}.")

    var_fc = np.empty(n - window)
    for k in range(window, n):
        var_fc[k - window] = _forecast_var(src[k - window:k], confidence, var_method)

    realized_eval = real[window:]
    eval_idx = idx[window:]
    exc = (realized_eval < -var_fc).astype(int)

    return ExceptionStream(
        name=name or (getattr(realized, "name", None) or "pnl"),
        confidence=confidence, window=window, var_method=var_method,
        var_forecast=pd.Series(var_fc, index=eval_idx, name="var_forecast"),
        realized=pd.Series(realized_eval, index=eval_idx, name="realized"),
        exceptions=pd.Series(exc, index=eval_idx, name="exception"),
        n_obs=len(exc), n_exceptions=int(exc.sum()),
    )
