"""FRTB Expected Shortfall at 97.5% with liquidity horizon scaling.

Pipeline:
  1. For each LH band j, build a "band P&L distribution" by running the
     hypothetical P&L pipeline with ONLY the band's RFs (zeroing the
     changes of all others). This isolates the contribution of that band.
  2. Compute ES_j at 97.5% on each band's P&L distribution, then scale it
     from the 1-day P&L horizon up to the FRTB base liquidity horizon of
     LH_1 = 10 days (sqrt-time, MAR33). Without this step the base band is a
     1-DAY ES (which reads ~= the 1-day 99% VaR) and understates capital by
     a factor of sqrt(10).
  3. Aggregate via the FRTB square-root formula:

        ES = sqrt( ES_1^2 + sum_{j>1} [(LH_j - LH_{j-1}) / LH_1] * ES_j^2 )

     where LH_1 = 10 (the base horizon) and every ES_j is now expressed at
     that base horizon. The increment factor (LH_j - LH_{j-1}) / LH_1
     measures how much *additional* time the long-horizon RFs need beyond
     the previous band, normalised to base.

  4. Return the aggregated FRTB ES, plus per-band ES for the dashboard.

Sign convention: ES is reported as a positive loss number, consistent
with the rest of the engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from frtb.liquidity_horizons import LHPartition, build_lh_partitions


# FRTB ES is calibrated at 97.5% (one-tailed), MAR33.2
FRTB_ES_CONFIDENCE: float = 0.975
FRTB_BASE_HORIZON_DAYS: int = 10


@dataclass
class FRTBESResult:
    """Output of the FRTB ES calculation."""
    es_aggregated: float                      # Final FRTB ES, monetary units (loss)
    es_per_band: dict[int, float] = field(default_factory=dict)
    n_rfs_per_band: dict[int, int] = field(default_factory=dict)
    base_horizon_days: int = FRTB_BASE_HORIZON_DAYS
    confidence: float = FRTB_ES_CONFIDENCE
    pnl_distribution: pd.Series | None = None   # Full-set P&L for reference

    def summary(self) -> pd.DataFrame:
        """Per-band breakdown for the dashboard."""
        rows = []
        for h, es in sorted(self.es_per_band.items()):
            rows.append({
                "horizon_d": h,
                "n_rfs":     self.n_rfs_per_band.get(h, 0),
                "ES_band":   es,
            })
        return pd.DataFrame(rows)


def _compute_es_at_confidence(pnl: pd.Series, confidence: float) -> float:
    """ES at given confidence: mean of losses beyond the (1-α) quantile.

    Returned as a positive number. If the tail is empty (degenerate
    P&L distribution), falls back to -quantile.
    """
    arr = pnl.values
    if len(arr) == 0:
        return 0.0
    alpha = 1.0 - confidence
    threshold = np.quantile(arr, alpha)
    tail = arr[arr <= threshold]
    if len(tail) == 0:
        return -float(threshold)
    return -float(tail.mean())


def compute_frtb_es(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    changes: pd.DataFrame | None = None,
    confidence: float = FRTB_ES_CONFIDENCE,
    base_horizon_days: int = FRTB_BASE_HORIZON_DAYS,
    input_horizon_days: int = 1,
) -> FRTBESResult:
    """Compute FRTB ES at 97.5% with LH scaling.

    Parameters
    ----------
    portfolio
        The book.
    sensitivities
        Output of Phase 2 build_sensitivities().
    changes
        Optional pre-computed RF changes matrix. If None, uses the
        portfolio's RFS aligned history. Phase 4 stressed calibration
        passes a windowed subset of this through.
    confidence
        FRTB sets 97.5%; exposed for sensitivity analysis only.
    base_horizon_days
        LH of the base band — FRTB sets 10d.
    input_horizon_days
        Holding-period horizon of the supplied P&L. The Phase 2 pipeline
        produces 1-day P&L, so this is 1; the band ES are scaled by
        sqrt(base_horizon_days / input_horizon_days) to reach the base LH.

    Returns
    -------
    FRTBESResult with:
      - es_aggregated: the FRTB ES at the base liquidity horizon
        (positive loss, monetary units)
      - es_per_band: ES at each LH band, also at the base horizon
        (positive losses)
      - n_rfs_per_band: how many RFs sit in each band (diagnostic)
    """
    rfs = portfolio.risk_factors
    if changes is None:
        changes = rfs.changes_matrix()

    partitions = build_lh_partitions(rfs)
    if not partitions:
        return FRTBESResult(es_aggregated=0.0)

    # ---- Step 1+2: per-band P&L distribution → per-band ES ----
    es_per_band: dict[int, float] = {}
    n_rfs_per_band: dict[int, int] = {}

    # We also keep the full-RF P&L distribution for the dashboard's
    # reference panel (Phase 9).
    hp_full = hypothetical_pnl(portfolio, sensitivities, changes=changes)
    pnl_full = hp_full.total

    # Holding-period scaling: lift each band's 1-day ES to the FRTB base
    # liquidity horizon (sqrt-time, MAR33). With input_horizon_days=1 and
    # base_horizon_days=10 this is sqrt(10). Applying it to every band ES
    # scales the aggregate by the same factor (the formula is homogeneous
    # of degree 1 in the ES terms).
    hp_scale = float(np.sqrt(base_horizon_days / input_horizon_days))

    for part in partitions:
        # Mask: zero out changes for RFs NOT in this band so the Phase 2
        # pipeline runs as if those RFs were frozen.
        masked = changes.copy()
        keep = set(part.rf_names)
        for col in masked.columns:
            if col not in keep:
                masked[col] = 0.0

        hp_band = hypothetical_pnl(portfolio, sensitivities, changes=masked)
        es_per_band[part.horizon_days] = _compute_es_at_confidence(
            hp_band.total, confidence
        ) * hp_scale
        n_rfs_per_band[part.horizon_days] = len(part.rf_names)

    # ---- Step 3: FRTB square-root aggregation ----
    # ES = sqrt( ES_1^2  +  Σ_{j>1} [(LH_j - LH_{j-1}) / LH_1] · ES_j^2 )
    bands_sorted = sorted(es_per_band.keys())
    es_1 = es_per_band[bands_sorted[0]]
    total_sq = es_1 ** 2

    for i, band in enumerate(bands_sorted):
        if i == 0:
            continue
        prev_band = bands_sorted[i - 1]
        increment = (band - prev_band) / base_horizon_days
        total_sq += increment * (es_per_band[band] ** 2)

    es_aggregated = float(np.sqrt(max(total_sq, 0.0)))

    return FRTBESResult(
        es_aggregated=es_aggregated,
        es_per_band=es_per_band,
        n_rfs_per_band=n_rfs_per_band,
        base_horizon_days=base_horizon_days,
        confidence=confidence,
        pnl_distribution=pnl_full,
    )
