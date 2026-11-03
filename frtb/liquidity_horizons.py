"""FRTB liquidity horizon partitioning.

Maps each risk factor to its FRTB liquidity horizon (MAR33.12 Table 2),
then partitions the RF set into the bands needed for the FRTB ES
aggregation formula.

The standard FRTB LH bands are: 10, 20, 40, 60, 120 days. This module
exposes only the four bands actually present in our universe (10, 20, 40,
60); 120-day RFs are typically EM equity and structured credit, which
we don't carry.

Each band j contains all RFs with LH >= LH_j. So the 10-day band has
every RF; the 20-day band excludes 10-day RFs; etc. This nested-subset
structure is what makes the FRTB square-root aggregation work — each
incremental band measures the *additional* risk from extending the
liquidation window.
"""

from __future__ import annotations
from dataclasses import dataclass

from data.risk_factors import RiskFactorSet, RiskFactorType, LIQUIDITY_HORIZONS


# Standard FRTB LH bands present in our universe (days).
# Excluded from this list: 120-day (EM equity, structured credit).
FRTB_LH_BANDS: tuple[int, ...] = (10, 20, 40, 60)


@dataclass(frozen=True)
class LHPartition:
    """One band of the FRTB LH partition.

    Attributes
    ----------
    horizon_days : int
        The band's lower-bound LH in days (10, 20, 40, 60).
    rf_names : tuple[str, ...]
        Risk factors with LH >= horizon_days. Note this is CUMULATIVE
        upward: the 10d band has everything, the 20d band excludes
        only the 10d RFs, etc.
    """
    horizon_days: int
    rf_names: tuple[str, ...]


def build_lh_partitions(rfs: RiskFactorSet) -> list[LHPartition]:
    """Partition the risk factor set into FRTB LH bands.

    Returns a list of LHPartition objects, one per band, in ascending
    horizon order. Bands with no risk factors are omitted (so a book
    with no credit positions won't carry a 40d band).
    """
    partitions: list[LHPartition] = []
    for band in FRTB_LH_BANDS:
        members = tuple(
            name for name, rf in rfs.factors.items()
            if rf.liquidity_horizon >= band
        )
        if members:
            partitions.append(LHPartition(horizon_days=band, rf_names=members))
    return partitions


def summarize_lh_assignment(rfs: RiskFactorSet) -> dict[str, dict[str, int | str]]:
    """Diagnostic: per-RF LH assignment, useful for the dashboard.

    Returns {rf_name: {"type": str, "lh_days": int}}.
    """
    return {
        name: {"type": rf.rf_type.value, "lh_days": rf.liquidity_horizon}
        for name, rf in rfs.factors.items()
    }
