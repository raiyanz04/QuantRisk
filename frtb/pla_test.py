"""FRTB P&L Attribution test (MAR32.40-46).

The gating test for Internal Models Approach eligibility. Compares two
frozen-position daily P&L streams (FRTB MAR32 terminology):

  - RTPL (risk-theoretical P&L): the risk model's reconstruction — here
    the Phase 2 sensitivities/Taylor expansion (greeks/pnl_attribution).
  - HPL (hypothetical P&L): the clean FULL-REVALUATION P&L
    (frtb/actual_pnl.compute_hpl). NB: in FRTB, "hypothetical" is the
    full-reval stream, NOT the Taylor one; both are distinct from APL
    (actual booked P&L).

The test compares RTPL against HPL via two statistics:

  1. Spearman rank correlation
     Tests whether the model orders days correctly — same days are
     winners and losers under both streams. Robust to scale differences
     between hypo and actual.

  2. Kolmogorov-Smirnov statistic
     Tests whether the two distributions have the same SHAPE. Catches
     cases where ranks line up but tails are mis-sized (e.g. hypo
     systematically under-estimates extreme moves).

FRTB zone thresholds (MAR32 Annex B):

  Spearman   |   KS    |   Zone
  ---------- | ------- | -------
  >= 0.80    | <= 0.09 | GREEN  (IMA approved, no surcharge)
  >= 0.70    | <= 0.12 | AMBER  (IMA approved with capital surcharge)
  <  0.70    |  > 0.12 | RED    (IMA REVOKED, fall back to SA)

A desk needs BOTH statistics in the green band to be green overall;
falls to the worst zone of the two otherwise.

What this means for capital. A red-zone desk must use the Standardised
Approach (SbM) for that desk's capital. SbM is typically 30-50% more
capital-intensive than ES-based IMA. Re-passing PLA reinstates IMA but
desks that fall red are watched closely — repeated failures can mean
regulator-mandated permanent SA migration.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp


# FRTB MAR32 Annex B PLA thresholds
PLA_SPEARMAN_GREEN: float = 0.80
PLA_SPEARMAN_AMBER: float = 0.70
PLA_KS_GREEN:       float = 0.09
PLA_KS_AMBER:       float = 0.12


@dataclass
class PLAResult:
    """Output of the PLA test."""
    spearman_rho: float
    spearman_pvalue: float
    ks_statistic: float
    ks_pvalue: float
    zone: str                       # 'green' | 'amber' | 'red'
    spearman_zone: str
    ks_zone: str
    n_observations: int
    ima_eligible: bool              # True if green or amber
    surcharge_required: bool        # True if amber

    def summary(self) -> str:
        return (
            f"PLA Test ({self.n_observations} observations)\n"
            f"  Spearman: {self.spearman_rho:+.4f}  (zone: {self.spearman_zone})\n"
            f"  KS stat:  {self.ks_statistic:.4f}   (zone: {self.ks_zone})\n"
            f"  Overall zone: {self.zone.upper()}\n"
            f"  IMA eligible: {self.ima_eligible}"
            f"{' (with capital surcharge)' if self.surcharge_required else ''}"
        )


def _classify_spearman(rho: float) -> str:
    if rho >= PLA_SPEARMAN_GREEN:
        return "green"
    elif rho >= PLA_SPEARMAN_AMBER:
        return "amber"
    return "red"


def _classify_ks(ks: float) -> str:
    if ks <= PLA_KS_GREEN:
        return "green"
    elif ks <= PLA_KS_AMBER:
        return "amber"
    return "red"


def _aggregate_zones(zone_a: str, zone_b: str) -> str:
    """Overall zone = worst of the two statistic zones."""
    order = {"green": 0, "amber": 1, "red": 2}
    return zone_a if order[zone_a] >= order[zone_b] else zone_b


def run_pla_test(
    rtpl: pd.Series,
    hpl: pd.Series,
) -> PLAResult:
    """Run the FRTB PLA test (RTPL vs HPL).

    Parameters
    ----------
    rtpl
        Risk-theoretical P&L — the Phase 2 sensitivities/Taylor stream
        (greeks.pnl_attribution.hypothetical_pnl(...).total).
    hpl
        Hypothetical P&L — the full-revaluation stream
        (frtb.actual_pnl.compute_hpl(...).total).

    Both series are aligned on common dates; NaNs and mismatched indices
    are dropped before computing statistics. The two PLA statistics are
    symmetric in their arguments, so argument order does not change the
    result.

    Returns
    -------
    PLAResult with both raw statistics, individual zones, the overall
    aggregated zone, and IMA-eligibility / surcharge flags.
    """
    # Align on shared dates and drop NaNs (a reval stream's first
    # observation can be NaN because of the diff).
    aligned = pd.concat([
        rtpl.rename("rtpl"),
        hpl.rename("hpl"),
    ], axis=1).dropna()

    if len(aligned) < 30:
        raise ValueError(
            f"PLA test requires at least 30 paired observations; got {len(aligned)}"
        )

    rho, rho_p = spearmanr(aligned["rtpl"].values, aligned["hpl"].values)

    # KS metric: the largest gap between the empirical CDFs of RTPL and
    # HPL (MAR32.46) — the two-sample KS statistic.
    ks_stat, ks_p = ks_2samp(
        aligned["rtpl"].values, aligned["hpl"].values,
    )

    sp_zone = _classify_spearman(rho)
    ks_zone = _classify_ks(ks_stat)
    overall = _aggregate_zones(sp_zone, ks_zone)

    return PLAResult(
        spearman_rho=float(rho),
        spearman_pvalue=float(rho_p),
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        zone=overall,
        spearman_zone=sp_zone,
        ks_zone=ks_zone,
        n_observations=len(aligned),
        ima_eligible=(overall != "red"),
        surcharge_required=(overall == "amber"),
    )
