"""Filtered Historical Simulation VaR.

Hull's three-step compromise between pure HS and pure MC:
  1. Fit GARCH(1,1) to each RF change series; obtain σ_t.
  2. Standardise: z_t = x_t / σ_t. Residuals approximately i.i.d.
  3. Sample joint residual ROWS, rescale by current GARCH vol forecast.

Joint row sampling (rather than per-RF independent sampling) preserves
the empirical correlation structure — critical to avoid grossly
underestimating diversified-portfolio VaR.
"""

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from var.types import VaRResult, fill_var_es_from_distribution

try:
    from arch import arch_model
    _HAVE_ARCH = True
except ImportError:
    _HAVE_ARCH = False


def _fit_garch_filter(series: pd.Series) -> tuple[pd.Series, float]:
    """Fit GARCH(1,1) and return (standardised residuals, σ_{T+1|T})."""
    if not _HAVE_ARCH:
        raise ImportError("arch package required: pip install arch")

    scale = 100.0
    x = series.dropna().values * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)

    sigma = pd.Series(res.conditional_volatility, index=series.dropna().index)
    z = series.dropna() / (sigma / scale)
    forecast = res.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0])) / scale
    return z, sigma_next


def compute_filtered_hs_var(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    n_resamples: int = 10_000,
    seed: int = 42,
) -> VaRResult:
    if not _HAVE_ARCH:
        raise ImportError("FHS requires `arch`: pip install arch")

    rfs = portfolio.risk_factors
    rf_names = rfs.names
    changes = rfs.changes_matrix()

    z_columns: dict[str, pd.Series] = {}
    sigma_next_map: dict[str, float] = {}
    for name in rf_names:
        x = changes[name]
        if x.std() == 0:
            warnings.warn(f"FHS: risk factor {name} has zero variance, using identity filter")
            z_columns[name] = x.copy()
            sigma_next_map[name] = 1.0
            continue
        z, sigma_next = _fit_garch_filter(x)
        z_columns[name] = z
        sigma_next_map[name] = sigma_next

    Z = pd.concat(z_columns, axis=1).dropna()

    rng = np.random.default_rng(seed)
    n_obs = len(Z)
    idx = rng.integers(low=0, high=n_obs, size=n_resamples)
    Z_sample = Z.iloc[idx].values

    sigma_next = np.array([sigma_next_map[n] for n in Z.columns])
    x_sim = Z_sample * sigma_next[None, :]

    sim_changes = pd.DataFrame(
        x_sim,
        index=pd.RangeIndex(n_resamples, name="scenario"),
        columns=Z.columns,
    )

    hp = hypothetical_pnl(portfolio, sensitivities, changes=sim_changes)
    pnl = hp.total.rename("fhs_pnl")

    result = VaRResult(
        method="filtered_historical",
        n_scenarios=n_resamples,
        pnl_distribution=pnl,
    )
    fill_var_es_from_distribution(
        result,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
    )
    return result
