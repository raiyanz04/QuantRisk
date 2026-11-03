"""FRTB stressed-ES calibration.

FRTB MAR33 requires Expected Shortfall be calibrated to a "period of
significant financial stress" — operationally defined as the 12-month
(=250 trading day) window in the historical data that maximises ES
on the CURRENT portfolio's sensitivities.

This is the regulator's structural answer to the procyclicality of
unstressed VaR: when markets are calm, unstressed VaR shrinks, capital
falls, and banks lever up — exactly when systemic risk is building.
Stressed calibration locks capital to a tail period regardless of
current conditions.

Implementation:
  1. Take the full RF changes matrix from history.
  2. Slide a 250-day window across the timeline (step optional for speed).
  3. For each window position, compute FRTB ES (with LH scaling) using
     ONLY that window's RF changes and today's sensitivities.
  4. Return the window with the maximum ES, plus the calibrated capital
     ES, plus a diagnostic series of ES-by-window-start-date.

With full FRED history (2010 onward), the search typically lands on
Feb-Apr 2020 (COVID crash) or Aug 2011 (euro sovereign crisis) as the
most stressed period for an equity-heavy, IG-credit-short book.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from frtb.expected_shortfall import compute_frtb_es, FRTBESResult


@dataclass
class StressedESResult:
    """Output of the rolling-window stressed-ES calibration."""
    stressed_es: float                   # Max ES across all windows (loss, +ve)
    stressed_window_start: pd.Timestamp  # Start date of the worst window
    stressed_window_end: pd.Timestamp    # End date of the worst window
    current_es: float                    # ES on the latest 250d window
    stress_amplification: float          # stressed_es / current_es
    es_by_window: pd.Series              # ES per window-end-date (diagnostic)
    stressed_band_breakdown: FRTBESResult | None = None
    # Per-band ES at the worst window — useful to show "what drove the stress"


def compute_stressed_es(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    window_days: int = 250,
    step_days: int = 5,
) -> StressedESResult:
    """Rolling-window search for the stressed period.

    Parameters
    ----------
    window_days
        Length of each rolling window. FRTB sets 250 (12 trading months).
    step_days
        How many days to advance between windows. step=1 evaluates every
        window; step=5 evaluates one window per week. Step=5 gives ~750
        evaluations on 15 years of data with negligible accuracy loss
        (the 12-month windows overlap heavily anyway).

    Returns
    -------
    StressedESResult with the maximum ES, the window that produced it,
    and the diagnostic ES-by-window series.
    """
    rfs = portfolio.risk_factors
    changes = rfs.changes_matrix()
    n = len(changes)

    if n < window_days:
        # Not enough history for a single window
        result = compute_frtb_es(portfolio, sensitivities, changes=changes)
        idx = changes.index
        return StressedESResult(
            stressed_es=result.es_aggregated,
            stressed_window_start=idx[0],
            stressed_window_end=idx[-1],
            current_es=result.es_aggregated,
            stress_amplification=1.0,
            es_by_window=pd.Series([result.es_aggregated], index=[idx[-1]]),
            stressed_band_breakdown=result,
        )

    # Slide windows across the timeline
    es_series: dict[pd.Timestamp, float] = {}
    last_band_breakdown: FRTBESResult | None = None

    for start in range(0, n - window_days + 1, step_days):
        end = start + window_days
        window_changes = changes.iloc[start:end]
        window_end_date = window_changes.index[-1]

        result = compute_frtb_es(
            portfolio, sensitivities, changes=window_changes,
        )
        es_series[window_end_date] = result.es_aggregated

    # Also force-evaluate the most recent window so we always have it
    # for the current_es / amplification comparison even if step doesn't
    # land on the last index.
    last_window_changes = changes.iloc[-window_days:]
    last_result = compute_frtb_es(
        portfolio, sensitivities, changes=last_window_changes,
    )
    current_es = last_result.es_aggregated
    es_series[changes.index[-1]] = current_es

    es_by_window = pd.Series(es_series).sort_index()
    worst_end_date = es_by_window.idxmax()
    stressed_es = float(es_by_window.max())

    # Get the band breakdown for the worst window (for diagnostics)
    worst_end_idx = changes.index.get_loc(worst_end_date)
    worst_start_idx = max(0, worst_end_idx - window_days + 1)
    worst_window_changes = changes.iloc[worst_start_idx : worst_end_idx + 1]
    stressed_band_breakdown = compute_frtb_es(
        portfolio, sensitivities, changes=worst_window_changes,
    )

    return StressedESResult(
        stressed_es=stressed_es,
        stressed_window_start=worst_window_changes.index[0],
        stressed_window_end=worst_window_changes.index[-1],
        current_es=current_es,
        stress_amplification=stressed_es / current_es if current_es > 0 else 1.0,
        es_by_window=es_by_window,
        stressed_band_breakdown=stressed_band_breakdown,
    )
