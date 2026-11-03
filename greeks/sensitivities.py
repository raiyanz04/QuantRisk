"""Per-position analytical Greeks.

Produces a single SensitivityMatrix that the rest of the engine consumes:

    SensitivityMatrix.delta : DataFrame (positions x risk_factors)
    SensitivityMatrix.gamma : DataFrame (positions x risk_factors)
    SensitivityMatrix.vega  : DataFrame (positions x risk_factors)
    SensitivityMatrix.rho   : DataFrame (positions x risk_factors)
    SensitivityMatrix.theta : Series   (positions)

Units (locked in here once and forever):
  - Delta on EQUITY_SPOT / FX_SPOT : monetary units per LOG RETURN of spot.
       For equity/FX outrights this equals the position's signed notional.
  - Delta on RATE                  : DV01 in monetary units per BP.
  - Delta on CREDIT_SPREAD         : CS01 in monetary units per BP.
  - Delta on EQUITY_VOL            : vega in monetary units per VOL POINT.
  - Gamma on EQUITY_SPOT           : monetary units per (LOG RETURN)^2.
  - Theta                          : monetary units per CALENDAR DAY.

In Phase 3 the VaR engines build P&L_hypo = X @ S.T where X is the
changes matrix (already in matching native scales). The matrix multiply
just works because units are locked here.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.portfolio import (
    Portfolio, Position,
    EquitySpot, EquityOption, GovBond, FXSpot, CreditIndex,
)
from data.risk_factors import RiskFactorSet, RiskFactorType


@dataclass
class SensitivityMatrix:
    """Sensitivities per (position, risk factor) pair plus per-position theta."""
    positions: list[Position]
    rf_names: list[str]
    delta: pd.DataFrame
    gamma: pd.DataFrame
    vega:  pd.DataFrame
    rho:   pd.DataFrame
    theta: pd.Series

    @property
    def position_names(self) -> list[str]:
        return [p.instrument.name for p in self.positions]

    def summary(self) -> pd.DataFrame:
        """One row per position with absolute totals across all RFs."""
        out = pd.DataFrame(index=self.position_names)
        out["|delta|"] = self.delta.abs().sum(axis=1)
        out["|gamma|"] = self.gamma.abs().sum(axis=1)
        out["|vega|"]  = self.vega.abs().sum(axis=1)
        out["|rho|"]   = self.rho.abs().sum(axis=1)
        out["theta"]   = self.theta
        return out


# ---------- Black-Scholes ----------

def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _bs_greeks_per_share(
    S: float, K: float, T: float, r: float, sigma: float, opt_type: str,
) -> dict[str, float]:
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if opt_type == "call":
        delta = norm.cdf(d1)
        rho = K * T * disc * norm.cdf(d2)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * disc * norm.cdf(d2)
    elif opt_type == "put":
        delta = norm.cdf(d1) - 1.0
        rho = -K * T * disc * norm.cdf(-d2)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * disc * norm.cdf(-d2)
    else:
        raise ValueError(f"Unknown option type: {opt_type!r}")

    return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho, "theta": theta}


# ---------- Bond pricing: cashflow-by-cashflow ----------

def _node_maturity(node_name: str) -> float:
    """Extract maturity in years from a node name like 'EUR_5Y' or 'UST_10Y'."""
    tag = node_name.split("_")[-1]
    return float(tag.rstrip("Y"))


def _bond_price_pv(
    face: float, coupon: float, maturity_years: float,
    curve_nodes: dict[str, float],
    coupons_per_year: int = 1,
) -> float:
    """Clean-price PV of a vanilla fixed-coupon bullet bond, discounting
    each cashflow at its own interpolated zero rate from the touched
    curve nodes.

    Cashflow-by-cashflow pricing means bumping a non-maturity node (e.g.
    the 2Y node when pricing a 10Y bond) correctly affects the
    intermediate coupon discount factors, producing non-trivial KRDs at
    every touched node.
    """
    if maturity_years <= 0:
        return face

    items = sorted(curve_nodes.items(), key=lambda kv: _node_maturity(kv[0]))
    mats = np.array([_node_maturity(n) for n, _ in items])
    yields = np.array([y for _, y in items])

    n = max(1, int(round(maturity_years * coupons_per_year)))
    c = coupon * face / coupons_per_year

    pv = 0.0
    for k in range(1, n + 1):
        t = k / coupons_per_year
        y_t = float(np.interp(t, mats, yields))
        df = (1 + y_t / coupons_per_year) ** (-k)
        pv += c * df
    t_final = n / coupons_per_year
    y_final = float(np.interp(t_final, mats, yields))
    pv += face * (1 + y_final / coupons_per_year) ** (-n)
    return pv


def _bond_krd(
    face: float, coupon: float, maturity_years: float,
    curve_nodes: dict[str, float],
    bump_bp: float = 1.0,
) -> dict[str, float]:
    """Key-rate DV01 per touched node via 1bp bump-and-reprice."""
    base_pv = _bond_price_pv(face, coupon, maturity_years, curve_nodes)

    krds: dict[str, float] = {}
    bump = bump_bp * 1e-4
    for node, y in curve_nodes.items():
        bumped = dict(curve_nodes)
        bumped[node] = y + bump
        new_pv = _bond_price_pv(face, coupon, maturity_years, bumped)
        krds[node] = (new_pv - base_pv) / bump_bp
    return krds


# ---------- Main entry point ----------

def build_sensitivities(
    portfolio: Portfolio,
    spot_overrides: dict[str, float] | None = None,
    vol_overrides: dict[str, float] | None = None,
    rate_for_options: float = 0.04,
    credit_duration: float = 4.0,
) -> SensitivityMatrix:
    """Compute the full sensitivity matrix."""
    rfs = portfolio.risk_factors
    pos_names = [p.instrument.name for p in portfolio.positions]
    rf_names = rfs.names

    z = lambda: pd.DataFrame(0.0, index=pos_names, columns=rf_names)
    delta = z()
    gamma = z()
    vega  = z()
    rho   = z()
    theta = pd.Series(0.0, index=pos_names)

    def spot(name: str) -> float:
        if spot_overrides and name in spot_overrides:
            return spot_overrides[name]
        return rfs[name].current

    def vol(name: str) -> float:
        if vol_overrides and name in vol_overrides:
            return vol_overrides[name]
        return rfs[name].current

    for pos in portfolio.positions:
        inst = pos.instrument
        sn = pos.signed_notional
        pname = inst.name

        if isinstance(inst, EquitySpot):
            delta.loc[pname, inst.rf_spot] = sn

        elif isinstance(inst, FXSpot):
            delta.loc[pname, inst.rf_fx] = sn

        elif isinstance(inst, CreditIndex):
            cs01 = -credit_duration * sn * 1e-4
            delta.loc[pname, inst.rf_spread] = cs01

        elif isinstance(inst, GovBond):
            curve_levels = {n: rfs[n].current for n in inst.rf_curve_nodes}
            face = 100.0
            krds = _bond_krd(face, inst.coupon, inst.maturity_years, curve_levels)
            scale = sn / face
            for node, k in krds.items():
                delta.loc[pname, node] += k * scale

        elif isinstance(inst, EquityOption):
            S = spot(inst.rf_spot)
            sigma = vol(inst.rf_vol)
            K = inst.strike
            T = inst.tenor_years
            r = rate_for_options
            g = _bs_greeks_per_share(S, K, T, r, sigma, inst.option_type)

            num_shares = sn / S

            delta.loc[pname, inst.rf_spot] += num_shares * g["delta"] * S
            gamma.loc[pname, inst.rf_spot] += num_shares * g["gamma"] * S * S
            vega.loc[pname, inst.rf_vol]   += num_shares * g["vega"] / 100.0

            rho_node = _pick_rho_node(rfs, inst.tenor_years)
            if rho_node is not None:
                rho.loc[pname, rho_node] += num_shares * g["rho"] / 1e4

            theta.loc[pname] += num_shares * g["theta"] / 365.0

        else:
            raise TypeError(f"Unknown instrument type: {type(inst)}")

    return SensitivityMatrix(
        positions=portfolio.positions,
        rf_names=rf_names,
        delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta,
    )


def _pick_rho_node(rfs: RiskFactorSet, tenor_years: float) -> str | None:
    rate_nodes = rfs.by_type(RiskFactorType.RATE)
    if not rate_nodes:
        return None
    closest = min(rate_nodes, key=lambda rf: abs(_node_maturity(rf.name) - tenor_years))
    return closest.name
