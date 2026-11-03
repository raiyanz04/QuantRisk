"""Monte Carlo VaR.

Cholesky-decompose the empirical covariance, simulate correlated RF
changes, and run them through the Phase 2 hypothetical P&L pipeline.
Includes gamma/vega contributions; parametric is delta-only.

Two distributions:
  - normal:    g = Z @ Lᵀ, joint-normal changes with covariance Σ.
  - student_t: a genuine multivariate-t, built by mixing the joint-normal
    draw with a SINGLE shared chi-square per path, so joint tail
    dependence (factors blowing up together) is preserved — not the
    per-factor-independent t that only fattens the marginals.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from var.types import VaRResult, fill_var_es_from_distribution


def compute_monte_carlo_var(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    n_paths: int = 10_000,
    distribution: str = "normal",
    t_df: float = 6.0,
    seed: int = 42,
    window_days: int | None = 250,
) -> VaRResult:
    rfs = portfolio.risk_factors
    rf_names = rfs.names
    K = len(rf_names)

    # Calibrate covariance on the SAME trailing window as historical
    # simulation, so the methods triangulate on one sample period rather
    # than mixing full-history (parametric/MC) with windowed (HS).
    changes = rfs.changes_matrix()
    if window_days is not None and len(changes) > window_days:
        changes = changes.iloc[-window_days:]
    cov = changes.cov().reindex(index=rf_names, columns=rf_names).values
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(K))
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Cholesky failed on RF covariance: {e}. "
            "Likely cause: a risk factor has zero variance in-window."
        )

    rng = np.random.default_rng(seed)

    # Correlated standard normals via Cholesky: Cov(g) = L Lᵀ = Σ.
    g = rng.standard_normal((n_paths, K)) @ L.T

    if distribution == "normal":
        x = g
    elif distribution == "student_t":
        if t_df <= 2:
            raise ValueError("student_t MC needs t_df > 2 for finite variance")
        # True multivariate-t: ONE chi-square mixing draw per path, shared
        # across all factors. Sampling an independent t per factor (the old
        # approach) gives fat marginals but normal-like JOINT tails — it
        # loses the tail dependence that is the whole reason to use t.
        # Rescale by sqrt((ν-2)/ν) so simulated changes still match Σ:
        # an MV-t with ν df has covariance Σ·ν/(ν-2).
        w = rng.chisquare(t_df, size=n_paths)
        x = g * np.sqrt((t_df - 2) / t_df) / np.sqrt(w / t_df)[:, None]
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    sim_index = pd.RangeIndex(n_paths, name="scenario")
    sim_changes = pd.DataFrame(x, index=sim_index, columns=rf_names)

    hp = hypothetical_pnl(portfolio, sensitivities, changes=sim_changes)
    pnl = hp.total.rename("mc_pnl")

    method_name = "monte_carlo_normal" if distribution == "normal" else f"monte_carlo_t{int(t_df)}"
    result = VaRResult(
        method=method_name,
        n_scenarios=n_paths,
        pnl_distribution=pnl,
    )
    fill_var_es_from_distribution(
        result,
        confidence_levels=confidence_levels,
        horizons_days=horizons_days,
    )
    return result
