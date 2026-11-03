"""Apply stress scenarios to the book.

For each scenario we report TWO numbers and their gap:

  - full_reval_pnl: the true loss, from a complete revaluation of every
    position under the shocked state (true simple-return move for cash
    equity/FX, full Black-Scholes reprice for options, full curve reprice
    for bonds). This is the headline, because stress moves are large and
    that is exactly where a delta-gamma ladder mis-states the answer.

  - greek_pnl: what the Phase 2 sensitivity ladder predicts — delta + half
    gamma * shock^2 + vega + rho, with no theta (a scenario is
    instantaneous). It decomposes cleanly by Greek and by risk factor, so
    you can see WHERE the loss comes from.

  - residual = full_reval_pnl - greek_pnl: the part the ladder misses. It
    blends genuine convexity (option gamma/vanna, bond convexity) with the
    linearization error of mapping a large move through a per-log-return
    delta. A material residual is the whole justification for full-reval
    stress testing.

Credit is modeled via a flat spread duration (CS01), so it is linear in
the engine and contributes zero residual — consistent with how the rest
of the engine treats credit.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.portfolio import (
    Portfolio, EquitySpot, EquityOption, GovBond, FXSpot, CreditIndex,
)
from data.risk_factors import RiskFactorSet
from greeks.sensitivities import SensitivityMatrix, _bond_price_pv, _pick_rho_node
from frtb.actual_pnl import _bs_price_per_share
from stress.scenarios import Scenario


@dataclass
class ScenarioResult:
    scenario: str
    description: str
    source: str
    full_reval_pnl: float            # headline true loss
    greek_pnl: float                 # delta-gamma-vega-rho estimate (no theta)
    residual: float                  # full - greek (convexity + linearization)
    by_greek: dict[str, float]       # {delta, gamma, vega, rho}
    by_position: pd.Series           # full-reval P&L per position (true)
    by_risk_factor: pd.Series        # Greek P&L attributed per RF (sums to greek_pnl)
    shock: pd.Series                 # resolved native shock vector

    def summary(self) -> str:
        g = self.by_greek
        worst = self.by_position.nsmallest(3)
        worst_str = ", ".join(f"{k} {v:,.0f}" for k, v in worst.items())
        return (
            f"{self.scenario} [{self.source}]\n"
            f"  Full reval P&L:   {self.full_reval_pnl:>14,.0f}\n"
            f"  Greek estimate:   {self.greek_pnl:>14,.0f}  "
            f"(delta {g['delta']:,.0f}, gamma {g['gamma']:,.0f}, "
            f"vega {g['vega']:,.0f}, rho {g['rho']:,.0f})\n"
            f"  Residual:         {self.residual:>14,.0f}  "
            f"({self.residual / self.full_reval_pnl * 100:+.1f}% of full)\n"
            f"  Worst positions:  {worst_str}"
        )


def _full_reval_pnl(
    portfolio: Portfolio,
    shock: dict[str, float],
    rate_for_options: float = 0.04,
    credit_duration: float = 4.0,
) -> pd.Series:
    """True per-position P&L under an instantaneous native-unit shock."""
    rfs = portfolio.risk_factors
    levels = {n: rfs[n].current for n in rfs.names}
    out: dict[str, float] = {}

    for pos in portfolio.positions:
        inst = pos.instrument
        sn = pos.signed_notional

        if isinstance(inst, EquitySpot):
            out[inst.name] = sn * (np.exp(shock.get(inst.rf_spot, 0.0)) - 1.0)

        elif isinstance(inst, FXSpot):
            out[inst.name] = sn * (np.exp(shock.get(inst.rf_fx, 0.0)) - 1.0)

        elif isinstance(inst, CreditIndex):
            out[inst.name] = -credit_duration * sn * 1e-4 * shock.get(inst.rf_spread, 0.0)

        elif isinstance(inst, GovBond):
            face, scale = 100.0, sn / 100.0
            base = {n: levels[n] for n in inst.rf_curve_nodes}
            shocked = {n: levels[n] + shock.get(n, 0.0) * 1e-4 for n in inst.rf_curve_nodes}
            base_pv = _bond_price_pv(face, inst.coupon, inst.maturity_years, base) * scale
            shock_pv = _bond_price_pv(face, inst.coupon, inst.maturity_years, shocked) * scale
            out[inst.name] = shock_pv - base_pv

        elif isinstance(inst, EquityOption):
            S0, sig0 = levels[inst.rf_spot], levels[inst.rf_vol]
            num_shares = sn / S0
            S1 = S0 * np.exp(shock.get(inst.rf_spot, 0.0))
            sig1 = sig0 + shock.get(inst.rf_vol, 0.0) / 100.0   # vol points -> decimal
            node = _pick_rho_node(rfs, inst.tenor_years)
            r1 = rate_for_options + (shock.get(node, 0.0) * 1e-4 if node else 0.0)
            base_px = _bs_price_per_share(
                S0, inst.strike, inst.tenor_years, rate_for_options, sig0, inst.option_type)
            shock_px = _bs_price_per_share(
                S1, inst.strike, inst.tenor_years, r1, max(sig1, 1e-6), inst.option_type)
            out[inst.name] = num_shares * (shock_px - base_px)

        else:
            raise TypeError(f"Unknown instrument: {type(inst)}")

    return pd.Series(out)


def _greek_attribution(
    sens: SensitivityMatrix,
    shock: pd.Series,
) -> tuple[dict[str, float], pd.Series]:
    """Greek P&L by component and by risk factor (no theta).

    Per RF: delta*shock + 0.5*gamma*shock^2 + vega*shock + rho*shock, summed
    over positions. Gamma is non-zero only on spot RFs, vega only on vol RFs,
    rho only on rate RFs, so the type restriction is implicit.
    """
    delta_rf = sens.delta.sum(axis=0).mul(shock, fill_value=0.0)
    gamma_rf = 0.5 * sens.gamma.sum(axis=0).mul(shock ** 2, fill_value=0.0)
    vega_rf = sens.vega.sum(axis=0).mul(shock, fill_value=0.0)
    rho_rf = sens.rho.sum(axis=0).mul(shock, fill_value=0.0)

    by_greek = {
        "delta": float(delta_rf.sum()),
        "gamma": float(gamma_rf.sum()),
        "vega":  float(vega_rf.sum()),
        "rho":   float(rho_rf.sum()),
    }
    by_rf = (
        delta_rf.add(gamma_rf, fill_value=0.0)
        .add(vega_rf, fill_value=0.0)
        .add(rho_rf, fill_value=0.0)
        .sort_values()
    )
    return by_greek, by_rf


def apply_scenario(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    scenario: Scenario,
    rate_for_options: float = 0.04,
    credit_duration: float = 4.0,
) -> ScenarioResult:
    """Run one scenario: full reval (headline) + Greek decomposition + residual."""
    shock = scenario.shocks_native
    full_by_pos = _full_reval_pnl(
        portfolio, shock.to_dict(), rate_for_options, credit_duration)
    by_greek, by_rf = _greek_attribution(sensitivities, shock)

    full_total = float(full_by_pos.sum())
    greek_total = sum(by_greek.values())
    return ScenarioResult(
        scenario=scenario.name, description=scenario.description, source=scenario.source,
        full_reval_pnl=full_total, greek_pnl=greek_total,
        residual=full_total - greek_total,
        by_greek=by_greek, by_position=full_by_pos.sort_values(),
        by_risk_factor=by_rf, shock=shock,
    )


def run_scenario_suite(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    scenarios: dict[str, Scenario],
    rate_for_options: float = 0.04,
    credit_duration: float = 4.0,
) -> tuple[list[ScenarioResult], pd.DataFrame]:
    """Run a library of scenarios and return results plus a comparison table."""
    results = [
        apply_scenario(portfolio, sensitivities, sc, rate_for_options, credit_duration)
        for sc in scenarios.values()
    ]
    table = pd.DataFrame([{
        "scenario":   r.scenario,
        "source":     r.source,
        "full_reval": round(r.full_reval_pnl, 0),
        "greek_est":  round(r.greek_pnl, 0),
        "residual":   round(r.residual, 0),
        "delta":      round(r.by_greek["delta"], 0),
        "gamma":      round(r.by_greek["gamma"], 0),
        "vega":       round(r.by_greek["vega"], 0),
        "rho":        round(r.by_greek["rho"], 0),
    } for r in results]).sort_values("full_reval").reset_index(drop=True)
    return results, table
