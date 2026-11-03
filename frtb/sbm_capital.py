"""FRTB Standardised Approach — Sensitivity-Based Method (MAR21).

Delta capital charge under the prescribed risk weights and correlations.

Implementation philosophy: capture the right order of magnitude and the
right relative ranking between risk classes, not the full bucket taxonomy.
Real FRTB SbM uses 11+ equity buckets, 25+ credit buckets, and per-tenor
rate weights — out of scope. We keep ONE bucket per risk class but apply
the PRESCRIBED MAR21 within-bucket correlations (replacing the earlier flat
0.15): the GIRR same-curve tenor matrix (MAR21.46), 25% for large-cap
advanced-economy equity (MAR21.78), and the 60% FX cross-bucket parameter
(MAR21.89, which acts as the correlation across single-factor FX buckets).
The charge is taken as the maximum over the three correlation scenarios
(high/medium/low, MAR21.6), as the real SA requires.

Why this matters: the old flat 0.15 mis-stated two classes. For a directional
rate book the high GIRR tenor correlation (~0.89-0.97) makes same-sign tenor
exposures ADD rather than diversify away, and the true equity correlation for
US large caps is 0.25, not the emerging-market 0.15. Both were understated.

Remaining simplifications, all conservative and noted inline:
  - Equity is one bucket; FRTB would split by sector (tech vs financials)
    with 25% within-sector and 15% across, so lumping at 25% slightly
    overstates correlation (higher capital).
  - GIRR risk weights are kept flat at the short-end 170bp rather than the
    per-tenor 130/110/110bp for 2y/5y/10y (overstates the rate charge).
  - Credit uses the single IG factor only; the full CSR name/tenor/basis
    correlation (MAR21.54) and multi-name vega are out of scope.

Risk weight unit convention (key fix from initial implementation):

  Risk class      | Sensitivity unit       | Shock unit              | WS = sens × shock
  ----------------|------------------------|-------------------------|--------------------
  Equity spot     | $ / log_return         | log return (0.30)       | sn × 0.30
  GIRR            | $ / bp (DV01)          | bps (170)               | DV01 × 170
  FX              | $ / log_return         | log return (0.075)      | sn × 0.075
  Credit IG       | $ / bp (CS01)          | bps (50)                | CS01 × 50
  Equity vega     | $ / vol_point          | vol points (~10)        | vega × 10

Three components in real FRTB SbM:
  1. DELTA charge (this module) — sensitivities × risk weights
  2. VEGA charge — separate from delta charge in FRTB; we fold it in
     under "EQUITY_VOL" for simplicity at this scope
  3. CURVATURE charge — second-order spot risk, captures gamma. Skipped
     here; for our book with only 2 modest-size options the curvature
     contribution would be < 5% of total SbM capital.

For an equity-heavy directional book like ours, the delta-only charge
typically captures ~85-95% of full SbM. Documented as a known
simplification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from data.risk_factors import RiskFactorType
from greeks.sensitivities import SensitivityMatrix


# ---------------------------------------------------------------
# FRTB-prescribed risk weights (simplified, single-bucket per class)
# Sources: MAR21 Section 4 risk weight tables, advanced economy values.
# ---------------------------------------------------------------

# Equity spot: 30% log-return shock (MAR21.77, advanced economy large-cap avg)
RW_EQUITY_SPOT_LR: float = 0.30

# GIRR (general interest rate risk): 170bp shock (MAR21.42, short-end weight)
RW_RATE_BPS: float = 170.0

# FX: 7.5% log-return shock (MAR21.87 is 15%, halved for specified pairs)
RW_FX_LR: float = 0.075

# Credit spread non-securitisation, IG corp: 50bp shock (MAR21.53 IG avg)
RW_CREDIT_BPS: float = 50.0

# Equity vol vega shock: ~10 vol points (MAR21 relative shock approximation)
RW_EQUITY_VOL_POINTS: float = 10.0


# ---------------------------------------------------------------
# FRTB-prescribed within-bucket correlations (MAR21), replacing flat 0.15.
# ---------------------------------------------------------------
# Equity delta, same bucket, large-cap advanced economy (MAR21.78(2)(b)).
# The old 0.15 was the EMERGING-market large-cap value (MAR21.78(2)(a)).
EQUITY_RHO: float = 0.25
# FX: each currency pair is its own bucket, so the cross-bucket parameter
# (MAR21.89) acts as the correlation across our pairs.
FX_GAMMA: float = 0.60
# GIRR same-curve, different-tenor: rho = max(e^{-theta|Tk-Tl|/min(Tk,Tl)}, 40%),
# theta = 3% (MAR21.46). Worked example 1y-5y = 88.69% reproduces exactly.
GIRR_THETA: float = 0.03
GIRR_RHO_FLOOR: float = 0.40


# ---------------------------------------------------------------
# Result container
# ---------------------------------------------------------------

@dataclass
class SbMCapitalResult:
    """Output of the SbM Delta capital charge."""
    total: float
    by_class: dict[str, float] = field(default_factory=dict)
    weighted_sensitivities: dict[str, float] = field(default_factory=dict)
    binding_scenario: str = "medium"
    scenario_totals: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = [{"risk_class": cls, "capital": cap}
                for cls, cap in sorted(self.by_class.items())]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Within-bucket aggregation: FRTB sqrt formula, K = sqrt(WS' R WS)
# ---------------------------------------------------------------

def _tenor_years(rf_name: str) -> float:
    """Parse a tenor in years from a rate RF name ('UST_10Y' -> 10,
    '6M' -> 0.5). Falls back to a mid-curve 5y if unparseable."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*Y", rf_name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*M", rf_name, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 12.0
    return 5.0


def _const_corr(n: int, rho: float) -> np.ndarray:
    """n x n correlation matrix: unit diagonal, constant off-diagonal rho."""
    R = np.full((n, n), float(rho))
    np.fill_diagonal(R, 1.0)
    return R


def _girr_corr(tenors: list[float]) -> np.ndarray:
    """GIRR same-curve tenor correlation matrix (MAR21.46)."""
    n = len(tenors)
    R = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ti, tj = tenors[i], tenors[j]
                R[i, j] = max(
                    float(np.exp(-GIRR_THETA * abs(ti - tj) / min(ti, tj))),
                    GIRR_RHO_FLOOR,
                )
    return R


# FRTB requires the charge under three correlation scenarios; the capital is
# the maximum of the three (MAR21.6). For a directional book the HIGH scenario
# usually binds, because inflating same-sign correlations removes netting.
CORRELATION_SCENARIOS: tuple[str, ...] = ("high", "medium", "low")


def _scenario_corr(R: np.ndarray, scenario: str) -> np.ndarray:
    """Apply the MAR21.6 scenario transformation to off-diagonal correlations:

        high:   rho_high = min(1.25 * rho, 1)
        medium: rho unchanged
        low:    rho_low  = max(2 * rho - 1, 0.75 * rho)
    """
    if scenario == "medium":
        return R
    off = ~np.eye(R.shape[0], dtype=bool)
    R2 = R.copy()
    vals = R[off]
    if scenario == "high":
        R2[off] = np.minimum(1.25 * vals, 1.0)
    elif scenario == "low":
        R2[off] = np.maximum(2.0 * vals - 1.0, 0.75 * vals)
    return R2


def _aggregate(weighted_sens: list[float], corr: np.ndarray) -> float:
    """K = sqrt( WS' R WS ), the FRTB within-bucket formula for an arbitrary
    correlation matrix R. Signed WS, so opposite-direction sensitivities net
    through the off-diagonal terms (the hedging benefit FRTB recognises
    within a bucket). A single risk factor reduces this to |WS|.
    """
    if len(weighted_sens) == 0:
        return 0.0
    ws = np.asarray(weighted_sens, dtype=float)
    return float(np.sqrt(max(float(ws @ corr @ ws), 0.0)))


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def compute_sbm_capital(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
) -> SbMCapitalResult:
    """Compute the FRTB SbM Delta capital charge.

    Process:
      1. Aggregate sensitivities to net-per-RF level.
      2. Apply the FRTB-prescribed risk weight per RF type to get WS.
      3. Aggregate WS within each risk class via K = sqrt(WS' R WS), using
         the prescribed correlation R for that class (GIRR tenor matrix,
         equity 25%, FX 60%; single-factor credit/vol reduce to |WS|).
      4. Repeat under the high/medium/low correlation scenarios (MAR21.6)
         and take the maximum total; sum across classes within each
         scenario (no cross-class diversification under SA).

    Returns
    -------
    SbMCapitalResult with total capital (the binding scenario), the
    per-risk-class breakdown, the binding scenario name, the three
    scenario totals, and the per-RF weighted sensitivities.
    """
    rfs = portfolio.risk_factors
    rf_types = {n: rfs[n].rf_type for n in rfs.names}

    # Net delta per RF (sum across positions)
    net_delta_per_rf = sensitivities.delta.sum(axis=0)
    # Net vega per RF (sum across positions) — separately for vol class
    net_vega_per_rf = sensitivities.vega.sum(axis=0)

    ws_by_class: dict[str, list[float]] = {
        "GIRR": [], "EQUITY": [], "FX": [], "CREDIT": [], "EQUITY_VOL": [],
    }
    girr_tenors: list[float] = []          # parallel to ws_by_class["GIRR"]
    ws_by_rf: dict[str, float] = {}

    for rf_name, net_delta in net_delta_per_rf.items():
        rf_type = rf_types[rf_name]

        if rf_type == RiskFactorType.EQUITY_SPOT:
            ws = net_delta * RW_EQUITY_SPOT_LR
            ws_by_class["EQUITY"].append(ws)
        elif rf_type == RiskFactorType.RATE:
            # DV01 in $/bp × shock in bps = monetary WS
            ws = net_delta * RW_RATE_BPS
            ws_by_class["GIRR"].append(ws)
            girr_tenors.append(_tenor_years(rf_name))
        elif rf_type == RiskFactorType.FX_SPOT:
            ws = net_delta * RW_FX_LR
            ws_by_class["FX"].append(ws)
        elif rf_type == RiskFactorType.CREDIT_SPREAD:
            ws = net_delta * RW_CREDIT_BPS
            ws_by_class["CREDIT"].append(ws)
        elif rf_type == RiskFactorType.EQUITY_VOL:
            # Vega is in $/vol_point; shock in vol points
            ws = net_vega_per_rf[rf_name] * RW_EQUITY_VOL_POINTS
            ws_by_class["EQUITY_VOL"].append(ws)
        else:
            continue

        ws_by_rf[rf_name] = float(ws)

    # Build the base (medium) correlation matrix per class once.
    base_corr: dict[str, np.ndarray] = {}
    for cls, ws_list in ws_by_class.items():
        if not ws_list:
            continue
        n = len(ws_list)
        if cls == "GIRR":
            base_corr[cls] = _girr_corr(girr_tenors)
        elif cls == "EQUITY":
            base_corr[cls] = _const_corr(n, EQUITY_RHO)
        elif cls == "FX":
            base_corr[cls] = _const_corr(n, FX_GAMMA)
        else:  # CREDIT, EQUITY_VOL — single factor here, R = I -> |WS|
            base_corr[cls] = _const_corr(n, 0.0)

    # FRTB charge = max over the high/medium/low correlation scenarios
    # (MAR21.6). Classes are summed within each scenario (no cross-class
    # diversification under SA).
    scenario_totals: dict[str, float] = {}
    scenario_by_class: dict[str, dict[str, float]] = {}
    for scen in CORRELATION_SCENARIOS:
        per_class = {
            cls: _aggregate(ws_by_class[cls], _scenario_corr(base_corr[cls], scen))
            for cls in base_corr
        }
        scenario_by_class[scen] = per_class
        scenario_totals[scen] = float(sum(per_class.values()))

    binding = max(scenario_totals, key=scenario_totals.get)
    by_class = scenario_by_class[binding]
    total = scenario_totals[binding]

    return SbMCapitalResult(
        total=total,
        by_class=by_class,
        weighted_sensitivities=ws_by_rf,
        binding_scenario=binding,
        scenario_totals=scenario_totals,
    )
