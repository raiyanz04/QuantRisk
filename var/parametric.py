"""Delta-normal (parametric) VaR.

Assumes joint-normal RF changes and linear P&L (delta only). Portfolio
variance via s' Σ s. Closed-form VaR via the normal quantile.

Why we still build it despite ignoring tails:
  - Speed: one matrix multiplication.
  - Decomposability: analytical component VaR (Phase 8).
  - Triangulation: HS - parametric gap is the fat-tail signal.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from var.types import VaRResult


def compute_parametric_var(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    n_synthetic_scenarios: int = 10_000,
    rng: np.random.Generator | None = None,
    window_days: int | None = 250,
) -> VaRResult:
    """Delta-normal VaR. ES uses the normal closed form φ(z)/(1-α)·σ.

    A synthetic normal P&L sample is materialised so the result carries
    a pnl_distribution like the other methods. VaR/ES values come from
    the analytical formula, not the sample.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    rfs = portfolio.risk_factors
    rf_names = rfs.names
    s = sensitivities.delta.reindex(columns=rf_names, fill_value=0.0).sum(axis=0).values

    # Same trailing window as HS / MC so the methods triangulate on one
    # sample period (see var_engine.run_all_methods).
    changes = rfs.changes_matrix()
    if window_days is not None and len(changes) > window_days:
        changes = changes.iloc[-window_days:]
    cov = changes.cov().reindex(index=rf_names, columns=rf_names).values

    port_var = float(s @ cov @ s)
    port_vol_1d = float(np.sqrt(max(port_var, 0.0)))

    result = VaRResult(
        method="delta_normal",
        n_scenarios=n_synthetic_scenarios,
        pnl_distribution=pd.Series(
            rng.normal(loc=0.0, scale=port_vol_1d, size=n_synthetic_scenarios),
            name="parametric_pnl",
        ),
    )
    for conf in confidence_levels:
        z = float(norm.ppf(conf))
        var_1d = z * port_vol_1d
        es_1d  = port_vol_1d * norm.pdf(z) / (1.0 - conf)
        for h in horizons_days:
            scale = np.sqrt(h)
            result.var[(conf, h)] = var_1d * scale
            result.es[(conf, h)]  = es_1d  * scale

    return result
