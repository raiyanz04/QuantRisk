"""Hypothetical P&L (HPL) via full revaluation under shocked-from-today state.

NAMING (FRTB MAR32). This module produces HPL: the clean,
frozen-position, FULL-REVALUATION P&L. It is NOT APL (actual/booked
P&L, which would include intraday trades, fees and reserves). The
sensitivities-based Taylor stream in greeks/pnl_attribution.py is the
RTPL (risk-theoretical P&L). The PLA test compares RTPL vs HPL — see
frtb/pla_test.py. The ``compute_actual_pnl`` / ``ActualPnL`` names are
kept as legacy aliases; prefer ``compute_hpl`` / ``HPLResult``.

Construction: today's portfolio is frozen at today's spots/yields/vols,
and each historical day's RF CHANGES are applied as a 1-day shock to
today's state. We then full-reprice each position under the shocked
state and report the delta from the unshocked baseline as that day's
HPL.

Why this rather than the "fixed quantity" interpretation:
  - RTPL (Phase 2) uses today's Greeks for every historical day. That's
    an implicit "shock today's portfolio" interpretation too.
  - PLA tests whether the Taylor approximation tracks the full reval;
    this only makes sense if both streams are reconstructions of the
    SAME counterfactual ("today's book shocked by day-t's moves").
  - For linear positions (cash equity, FX, credit-via-duration),
    HPL == RTPL exactly under this convention. PLA divergence therefore
    isolates the contribution of non-linearity from bonds and options —
    which is exactly what FRTB MAR32 wants the test to measure.

Performance: 14 positions × ~3,700 days = ~50k position-reprices. Bond
reprice is O(maturity_years) and BS is O(1), so wall-clock is < 1 second.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.portfolio import (
    Portfolio,
    EquitySpot, EquityOption, GovBond, FXSpot, CreditIndex,
)
from greeks.sensitivities import _bs_d1_d2, _bond_price_pv, _pick_rho_node


def _bs_price_per_share(
    S: float, K: float, T: float, r: float, sigma: float, opt_type: str,
) -> float:
    """Black-Scholes price per share, q=0."""
    if T <= 0:
        if opt_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if sigma <= 0:
        disc = np.exp(-r * T)
        fwd = S * np.exp(r * T)
        if opt_type == "call":
            return max(fwd - K, 0.0) * disc
        return max(K - fwd, 0.0) * disc

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    disc = np.exp(-r * T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif opt_type == "put":
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"Unknown option type: {opt_type!r}")


@dataclass
class HPLResult:
    total: pd.Series
    by_position: pd.DataFrame
    reference_date: pd.Timestamp


# Legacy alias (pre-FRTB-correct naming).
ActualPnL = HPLResult


def compute_hpl(
    portfolio: Portfolio,
    rate_for_options: float = 0.04,
    credit_duration: float = 4.0,
    reference_date: pd.Timestamp | None = None,
) -> HPLResult:
    """HPL: today's portfolio shocked by each historical day's RF
    changes, then full-repriced.

    Each row of the output is a 1-day P&L number representing what
    today's book would have made/lost if today's RFs had moved by the
    amounts observed on date t.

    For linear positions this equals RTPL exactly (under the log-return
    convention). For bonds, convexity introduces a small wedge. For
    options, BS gamma/vega/vanna introduce a larger wedge — which is the
    content of the PLA test.
    """
    rfs = portfolio.risk_factors
    changes = rfs.changes_matrix()   # native-scale changes
    rf_names = rfs.names

    if reference_date is None:
        reference_date = changes.index[-1]

    # Today's RF levels — the "unshocked" baseline state.
    today_levels: dict[str, float] = {name: rfs[name].current for name in rf_names}

    pos_names = [p.instrument.name for p in portfolio.positions]
    n_days = len(changes)
    pnl_array = np.zeros((n_days, len(pos_names)))

    # Pre-compute today's option baseline values, num_shares.
    # And today's bond PVs (one per bond position) so the "unshocked"
    # reference is consistent.
    today_bond_pv: dict[int, float] = {}    # pos_index -> baseline PV
    today_bond_scale: dict[int, float] = {}
    today_option_baseline: dict[int, float] = {}  # pos_index -> baseline value
    today_option_num_shares: dict[int, float] = {}
    today_option_rate_node: dict[int, str | None] = {}  # nearest curve node

    for j, pos in enumerate(portfolio.positions):
        inst = pos.instrument
        if isinstance(inst, GovBond):
            curve = {n: today_levels[n] for n in inst.rf_curve_nodes}
            face = 100.0
            scale = pos.signed_notional / face
            base_pv = _bond_price_pv(face, inst.coupon, inst.maturity_years, curve)
            today_bond_pv[j] = base_pv * scale
            today_bond_scale[j] = scale
        elif isinstance(inst, EquityOption):
            S0 = today_levels[inst.rf_spot]
            sigma0 = today_levels[inst.rf_vol]
            num_shares = pos.signed_notional / S0
            base_price = _bs_price_per_share(
                S=S0, K=inst.strike, T=inst.tenor_years,
                r=rate_for_options, sigma=sigma0,
                opt_type=inst.option_type,
            )
            today_option_num_shares[j] = num_shares
            today_option_baseline[j] = num_shares * base_price
            # Same nearest-node mapping the sensitivity matrix uses for rho,
            # so the rate shock below is consistent with RTPL's rho term.
            today_option_rate_node[j] = _pick_rho_node(rfs, inst.tenor_years)

    # For each historical day, apply that day's RF changes to today's
    # state and reprice each position.
    for i in range(n_days):
        day_changes: dict[str, float] = {
            name: float(changes.iat[i, k])
            for k, name in enumerate(rf_names)
        }

        for j, pos in enumerate(portfolio.positions):
            inst = pos.instrument
            sn = pos.signed_notional

            if isinstance(inst, EquitySpot):
                # log_return shock; P&L = sn * log_return (matches hypo)
                pnl_array[i, j] = sn * day_changes[inst.rf_spot]

            elif isinstance(inst, FXSpot):
                pnl_array[i, j] = sn * day_changes[inst.rf_fx]

            elif isinstance(inst, CreditIndex):
                # d_spread in bps; P&L = -duration * sn * 1e-4 * d_spread_bps
                pnl_array[i, j] = -credit_duration * sn * 1e-4 * day_changes[inst.rf_spread]

            elif isinstance(inst, GovBond):
                # Shock curve: bp changes → decimal shift
                shocked = {
                    n: today_levels[n] + day_changes[n] * 1e-4
                    for n in inst.rf_curve_nodes
                }
                face = 100.0
                shocked_pv = _bond_price_pv(face, inst.coupon, inst.maturity_years, shocked) * today_bond_scale[j]
                pnl_array[i, j] = shocked_pv - today_bond_pv[j]

            elif isinstance(inst, EquityOption):
                # Spot shock = exp(log_return); vol shock = + d_vol_points/100.
                # Rate shock: bump the discount rate by the change of the same
                # curve node the option's analytical rho maps to, so HPL and
                # RTPL treat the option's rate sensitivity consistently
                # (otherwise RTPL carries a rho·dr term that HPL never sees).
                S_shocked = today_levels[inst.rf_spot] * np.exp(day_changes[inst.rf_spot])
                sigma_shocked = today_levels[inst.rf_vol] + day_changes[inst.rf_vol] / 100.0
                rate_node = today_option_rate_node[j]
                r_shocked = rate_for_options
                if rate_node is not None and rate_node in day_changes:
                    r_shocked = rate_for_options + day_changes[rate_node] * 1e-4
                shocked_price = _bs_price_per_share(
                    S=S_shocked, K=inst.strike, T=inst.tenor_years,
                    r=r_shocked, sigma=max(sigma_shocked, 1e-6),
                    opt_type=inst.option_type,
                )
                shocked_value = today_option_num_shares[j] * shocked_price
                pnl_array[i, j] = shocked_value - today_option_baseline[j]

            else:
                raise TypeError(f"Unknown instrument: {type(inst)}")

    by_position = pd.DataFrame(pnl_array, index=changes.index, columns=pos_names)
    total = by_position.sum(axis=1)
    total.name = "hpl"

    return HPLResult(
        total=total,
        by_position=by_position,
        reference_date=reference_date,
    )


# Legacy alias (pre-FRTB-correct naming): prefer compute_hpl.
compute_actual_pnl = compute_hpl
