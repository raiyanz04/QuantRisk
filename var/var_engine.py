"""Unified VaR facade.

Single entry point that runs all four methods and produces a comparison.

Usage:
    from data.market_data import build_us_risk_factor_set
    from data.book_us import build_book
    from greeks.sensitivities import build_sensitivities
    from var.var_engine import run_all_methods

    rfs = build_us_risk_factor_set()
    pf = build_book(rfs)
    sens = build_sensitivities(pf)
    results, comparison = run_all_methods(pf, sens)
    print(comparison)
"""

from __future__ import annotations
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from var.types import VaRResult, compare_methods
from var.parametric import compute_parametric_var
from var.historical import compute_historical_var
from var.monte_carlo import compute_monte_carlo_var
from var.filtered_hs import compute_filtered_hs_var, _HAVE_ARCH


def run_all_methods(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    window_days: int = 250,
    mc_paths: int = 10_000,
    fhs_resamples: int = 10_000,
    include_fhs: bool = True,
    mc_distribution: str = "normal",
) -> tuple[list[VaRResult], pd.DataFrame]:
    """Run all four VaR methods on the same book.

    ``window_days`` is the common estimation window for delta-normal,
    historical simulation, and Monte Carlo, so their VaR/ES are directly
    comparable — one sample period, not full-history vs trailing-window.
    Filtered HS deliberately fits its GARCH on the FULL series: it is a
    conditional, current-vol-scaled method, so a long fit is correct and
    is the one legitimate exception to the common window.
    """
    results: list[VaRResult] = []

    results.append(compute_parametric_var(
        portfolio, sensitivities,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
        window_days=window_days,
    ))

    results.append(compute_historical_var(
        portfolio, sensitivities,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
        window_days=window_days,
    ))

    results.append(compute_monte_carlo_var(
        portfolio, sensitivities,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
        n_paths=mc_paths,
        distribution=mc_distribution,
        window_days=window_days,
    ))

    if include_fhs and _HAVE_ARCH:
        results.append(compute_filtered_hs_var(
            portfolio, sensitivities,
            confidence_levels=confidence_levels,
            horizons_days=horizons_days,
            n_resamples=fhs_resamples,
        ))

    comparison = compare_methods(results)
    return results, comparison
