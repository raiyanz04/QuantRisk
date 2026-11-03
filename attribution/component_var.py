"""Phase 8 — VaR attribution by position.

Aggregate VaR tells you the size of the risk; attribution tells you where
it lives and what a trade would do to it. Two decompositions, both summing
to total VaR:

  - Component VaR (Euler / Jorion): CVaR_i = z * cov(P&L_i, P&L_p) / sigma_p.
    The additive share of portfolio VaR carried by position i; risk hogs are
    the large positive contributions, hedges are negative.

  - Marginal VaR: the derivative of portfolio VaR w.r.t. position size,
    reported per 1m of added (signed) notional = CVaR_i / notional_i * 1e6 —
    the number a trade-approval desk reads ("what does +1m here do to VaR?").

Plus two non-additive views:
  - Standalone VaR_i: the position's VaR in isolation (no diversification).
  - Incremental VaR_i: VaR_p - VaR_without_i, the discrete effect of removing
    the whole position (distinct from component VaR, the marginal share).

Two estimators, mirroring the engine's parametric-vs-historical split:
  - parametric_var_attribution: delta-normal, exact closed-form Euler. Options
    enter via delta only (consistent with var/parametric.py); their convexity
    is not in this linear decomposition.
  - historical_var_attribution: allocates the empirical HS VaR by each
    position's average P&L in the loss tail, so it DOES carry option convexity
    and fat-tail composition that the Gaussian decomposition cannot.

Both use the same trailing window as the Phase 3 VaR engines, so the totals
reconcile with compute_parametric_var / compute_historical_var.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl


@dataclass
class VaRAttribution:
    method: str                          # "parametric" | "historical"
    confidence: float
    horizon_days: int
    portfolio_var: float                 # reconciles with the Phase 3 VaR
    portfolio_vol: float
    component_var: pd.Series             # CVaR_i, sums to portfolio_var
    marginal_var_per_1m: pd.Series       # dVaR per +1m signed notional
    standalone_var: pd.Series            # VaR_i in isolation
    incremental_var: pd.Series           # VaR_p - VaR_without_i
    pct_contribution: pd.Series          # CVaR_i / portfolio_var

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "component_var": self.component_var.round(0),
            "pct_of_total":  (self.pct_contribution * 100).round(1),
            "marginal_/1m":  self.marginal_var_per_1m.round(0),
            "standalone":    self.standalone_var.round(0),
            "incremental":   self.incremental_var.round(0),
        })
        return df.sort_values("component_var", ascending=False)

    def summary(self) -> str:
        chk = float(self.component_var.sum())
        return (
            f"{self.method} VaR attribution @ {self.confidence:.1%}, "
            f"{self.horizon_days}d  (portfolio VaR {self.portfolio_var:,.0f})\n"
            + self.table().to_string()
            + f"\n  [check] sum(component) = {chk:,.0f}  vs  VaR {self.portfolio_var:,.0f}"
        )


def parametric_var_attribution(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence: float = 0.99,
    horizon_days: int = 1,
    window_days: int | None = 250,
) -> VaRAttribution:
    """Delta-normal Euler decomposition. Exact and additive.

    sigma_p = sqrt(s' Σ s) with s the net delta and Σ the (windowed) RF
    covariance; CVaR_i = z * (δ_i' Σ s) / sigma_p * sqrt(h), which sums to
    z * sigma_p * sqrt(h) = portfolio VaR.
    """
    rfs = portfolio.risk_factors
    rf_names = rfs.names
    pos_names = [p.instrument.name for p in portfolio.positions]

    D = sensitivities.delta.reindex(index=pos_names, columns=rf_names, fill_value=0.0).values
    changes = rfs.changes_matrix()
    if window_days is not None and len(changes) > window_days:
        changes = changes.iloc[-window_days:]
    Sigma = changes.cov().reindex(index=rf_names, columns=rf_names).values

    s = D.sum(axis=0)
    z = float(norm.ppf(confidence))
    sqrt_h = float(np.sqrt(horizon_days))

    Sigma_s = Sigma @ s
    sigma_p = float(np.sqrt(max(float(s @ Sigma_s), 0.0)))
    VaR_p = z * sigma_p * sqrt_h
    idx = pd.Index(pos_names, name="position")

    if sigma_p == 0.0:
        zeros = pd.Series(0.0, index=idx)
        return VaRAttribution(
            "parametric", confidence, horizon_days, 0.0, 0.0,
            zeros, zeros, zeros, zeros, zeros)

    cov_i_p = D @ Sigma_s                                       # cov(P&L_i, P&L_p)
    cvar = z * cov_i_p / sigma_p * sqrt_h                       # sums to VaR_p
    standalone = z * np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", D, Sigma, D), 0.0)) * sqrt_h

    notionals = np.array([float(p.signed_notional) for p in portfolio.positions])
    with np.errstate(divide="ignore", invalid="ignore"):
        mvar_1m = np.where(notionals != 0.0, cvar / notionals * 1e6, 0.0)

    ivar = np.empty(len(pos_names))
    for i in range(len(pos_names)):
        s_wo = s - D[i]
        ivar[i] = VaR_p - z * np.sqrt(max(float(s_wo @ Sigma @ s_wo), 0.0)) * sqrt_h

    return VaRAttribution(
        method="parametric", confidence=confidence, horizon_days=horizon_days,
        portfolio_var=VaR_p, portfolio_vol=sigma_p * sqrt_h,
        component_var=pd.Series(cvar, idx),
        marginal_var_per_1m=pd.Series(mvar_1m, idx),
        standalone_var=pd.Series(standalone, idx),
        incremental_var=pd.Series(ivar, idx),
        pct_contribution=pd.Series(cvar / VaR_p, idx),
    )


def historical_var_attribution(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    confidence: float = 0.99,
    horizon_days: int = 1,
    window_days: int | None = 250,
) -> VaRAttribution:
    """Empirical decomposition: allocate HS VaR by each position's average
    P&L in the loss tail (P&L at or beyond the VaR quantile), rescaled so the
    contributions sum exactly to the portfolio HS VaR. Carries option
    convexity and fat-tail composition the Gaussian view misses.
    """
    hp = hypothetical_pnl(portfolio, sensitivities)
    by_pos, total = hp.by_position, hp.total
    if window_days is not None and len(total) > window_days:
        by_pos = by_pos.iloc[-window_days:]
        total = total.iloc[-window_days:]

    pos_names = list(by_pos.columns)
    idx = pd.Index(pos_names, name="position")
    sqrt_h = float(np.sqrt(horizon_days))
    alpha = 1.0 - confidence
    var_q = float(total.quantile(alpha))
    VaR_p = -var_q * sqrt_h

    tail = total <= var_q
    if int(tail.sum()) == 0:                      # tiny sample / coarse quantile
        tail = total <= total.quantile(alpha, interpolation="higher")

    raw = -(by_pos.loc[tail].mean()) * sqrt_h     # tail-conditional contributions
    raw = raw.reindex(idx)
    total_raw = float(raw.sum())
    cvar = raw * (VaR_p / total_raw) if total_raw != 0 else raw   # rescale: sum = VaR_p

    standalone = pd.Series(
        {c: -float(by_pos[c].quantile(alpha)) * sqrt_h for c in pos_names}, index=idx)

    ivar = pd.Series({
        c: VaR_p - (-float((total - by_pos[c]).quantile(alpha)) * sqrt_h)
        for c in pos_names
    }, index=idx)

    notionals = pd.Series(
        {p.instrument.name: float(p.signed_notional) for p in portfolio.positions}
    ).reindex(idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        mvar_1m = (cvar / notionals * 1e6).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return VaRAttribution(
        method="historical", confidence=confidence, horizon_days=horizon_days,
        portfolio_var=VaR_p, portfolio_vol=float("nan"),
        component_var=cvar,
        marginal_var_per_1m=mvar_1m,
        standalone_var=standalone,
        incremental_var=ivar,
        pct_contribution=(cvar / VaR_p if VaR_p else cvar),
    )
