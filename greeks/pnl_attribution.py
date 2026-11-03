"""Risk-theoretical P&L reconstruction via Taylor expansion.

For a portfolio with sensitivities S = (Δ, Γ, V, ρ, θ) and risk factor
changes dX (already in matching native scales):

    P&L = Δ·dX + ½·Γ·dX² + V·dσ + ρ·dr + θ·dt

NAMING (FRTB MAR32). This Taylor / sensitivities stream is the RTPL
(risk-theoretical P&L) in the PLA test — what the risk model reconstructs.
The FRTB "hypothetical P&L" (HPL) is the FULL-REVALUATION stream and lives
in frtb/actual_pnl.py. The function name ``hypothetical_pnl`` is retained
in its standard VaR sense (today's book revalued under historical or
simulated moves), with ``risk_theoretical_pnl`` provided as the
FRTB-correct alias.

This module produces both the portfolio total and the by-Greek /
by-position / by-RF decompositions. The total series is the input to:
  - Historical simulation VaR (Phase 3)
  - Filtered historical VaR (Phase 3)
  - FRTB Expected Shortfall (Phase 4)
  - P&L Attribution test, as the RTPL stream (Phase 5)
  - Component / marginal VaR attribution (Phase 8)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from data.risk_factors import RiskFactorType
from greeks.sensitivities import SensitivityMatrix


@dataclass
class HypotheticalPnL:
    """Hypothetical P&L decomposed by Greek and aggregated."""
    total: pd.Series
    by_greek: pd.DataFrame
    by_position: pd.DataFrame
    by_risk_factor: pd.DataFrame

    def summary(self) -> pd.Series:
        return pd.Series({
            "n_days":            len(self.total),
            "mean_daily":        self.total.mean(),
            "std_daily":         self.total.std(),
            "annualised_vol":    self.total.std() * np.sqrt(252),
            "min":               self.total.min(),
            "max":               self.total.max(),
            "skew":              self.total.skew(),
            "kurtosis":          self.total.kurtosis(),
        })


def hypothetical_pnl(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    changes: pd.DataFrame | None = None,
) -> HypotheticalPnL:
    """Compute hypothetical P&L over the joint date index of the RFS,
    or over an arbitrary supplied changes matrix (used in Phase 3 MC
    and Phase 7 stress)."""
    rfs = portfolio.risk_factors
    if changes is None:
        changes = rfs.changes_matrix()

    rf_cols = list(changes.columns)
    delta = sensitivities.delta.reindex(columns=rf_cols, fill_value=0.0)
    gamma = sensitivities.gamma.reindex(columns=rf_cols, fill_value=0.0)
    vega  = sensitivities.vega.reindex(columns=rf_cols, fill_value=0.0)
    rho   = sensitivities.rho.reindex(columns=rf_cols, fill_value=0.0)
    theta = sensitivities.theta

    rf_types = pd.Series({n: rfs[n].rf_type for n in rf_cols})
    is_spot = rf_types.isin([RiskFactorType.EQUITY_SPOT, RiskFactorType.FX_SPOT])
    is_vol  = rf_types.eq(RiskFactorType.EQUITY_VOL)
    is_rate = rf_types.eq(RiskFactorType.RATE)

    dX = changes.values
    dX_spot = changes.loc[:, is_spot].values
    dX_vol  = changes.loc[:, is_vol].values
    dX_rate = changes.loc[:, is_rate].values

    delta_pnl_pos = dX @ delta.T.values
    gamma_pnl_pos = 0.5 * (dX_spot ** 2) @ gamma.loc[:, is_spot].T.values
    vega_pnl_pos  = dX_vol  @ vega.loc[:,  is_vol].T.values
    rho_pnl_pos   = dX_rate @ rho.loc[:,  is_rate].T.values

    n_days = len(changes)
    theta_pnl_pos = np.broadcast_to(theta.values, (n_days, len(theta))).copy()

    pos_names = delta.index.tolist()
    total_pos = (
        delta_pnl_pos + gamma_pnl_pos + vega_pnl_pos + rho_pnl_pos + theta_pnl_pos
    )
    by_position = pd.DataFrame(total_pos, index=changes.index, columns=pos_names)

    by_greek = pd.DataFrame({
        "delta": delta_pnl_pos.sum(axis=1),
        "gamma": gamma_pnl_pos.sum(axis=1),
        "vega":  vega_pnl_pos.sum(axis=1),
        "rho":   rho_pnl_pos.sum(axis=1),
        "theta": theta_pnl_pos.sum(axis=1),
    }, index=changes.index)

    by_rf = changes * delta.sum(axis=0).reindex(changes.columns).values

    total = by_greek.sum(axis=1)
    total.name = "hypothetical_pnl"

    return HypotheticalPnL(
        total=total, by_greek=by_greek,
        by_position=by_position, by_risk_factor=by_rf,
    )


# FRTB-correct aliases: this Taylor stream IS the RTPL in the PLA test.
RiskTheoreticalPnL = HypotheticalPnL
risk_theoretical_pnl = hypothetical_pnl


def actual_pnl_proxy(portfolio: Portfolio) -> pd.Series:
    """Crude actual-P&L proxy on linear positions only — sanity check
    that hypothetical P&L tracks actual MtM for cash equity/FX/credit/bonds."""
    rfs = portfolio.risk_factors
    changes = rfs.changes_matrix()
    pnl = pd.Series(0.0, index=changes.index)

    from data.portfolio import EquitySpot, FXSpot, CreditIndex, GovBond

    for pos in portfolio.positions:
        inst = pos.instrument
        sn = pos.signed_notional
        if isinstance(inst, (EquitySpot, FXSpot)):
            rf_name = getattr(inst, "rf_spot", None) or getattr(inst, "rf_fx", None)
            pnl += sn * changes[rf_name]
        elif isinstance(inst, CreditIndex):
            credit_dur = 4.0
            pnl += -credit_dur * sn * 1e-4 * changes[inst.rf_spread]
        elif isinstance(inst, GovBond):
            mat = inst.maturity_years
            avg_dur = mat / (1 + 0.04)
            for node in inst.rf_curve_nodes:
                pnl += -(avg_dur / len(inst.rf_curve_nodes)) * sn * 1e-4 * changes[node]

    pnl.name = "actual_pnl_proxy"
    return pnl
