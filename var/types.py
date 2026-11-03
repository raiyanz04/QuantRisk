"""Shared types and utilities for the VaR engines."""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class VaRResult:
    """Unified output of every VaR method."""
    method: str
    n_scenarios: int
    pnl_distribution: pd.Series
    var: dict[tuple[float, int], float] = field(default_factory=dict)
    es:  dict[tuple[float, int], float] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        keys = sorted(self.var.keys())
        rows = []
        for conf, h in keys:
            rows.append({
                "confidence": conf,
                "horizon_d":  h,
                "VaR":        self.var.get((conf, h), np.nan),
                "ES":         self.es.get((conf, h), np.nan),
            })
        return pd.DataFrame(rows)


def fill_var_es_from_distribution(
    result: VaRResult,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    horizons_days: tuple[int, ...] = (1, 10),
    base_horizon_days: int = 1,
) -> None:
    """Compute VaR and ES at every (confidence, horizon) and fill in-place.

    Multi-day VaR/ES via sqrt-time scaling (regulatory convention).
    Sign convention: VaR and ES reported as POSITIVE LOSS NUMBERS.
    """
    pnl = result.pnl_distribution.values
    if len(pnl) == 0:
        return

    for conf in confidence_levels:
        alpha = 1.0 - conf
        var_quantile = np.quantile(pnl, alpha)
        var_1d = -var_quantile

        tail = pnl[pnl <= var_quantile]
        es_1d = -tail.mean() if len(tail) > 0 else var_1d

        for h in horizons_days:
            scale = np.sqrt(h / base_horizon_days)
            result.var[(conf, h)] = var_1d * scale
            result.es[(conf, h)]  = es_1d * scale


def compare_methods(results: list[VaRResult]) -> pd.DataFrame:
    """Side-by-side VaR/ES across methods, indexed by (conf, horizon)."""
    rows = []
    for r in results:
        for (conf, h), v in sorted(r.var.items()):
            rows.append({
                "method":     r.method,
                "confidence": conf,
                "horizon_d":  h,
                "VaR":        v,
                "ES":         r.es.get((conf, h), np.nan),
                "n_scen":     r.n_scenarios,
            })
    df = pd.DataFrame(rows)
    return df.pivot_table(
        index=["confidence", "horizon_d"],
        columns="method",
        values=["VaR", "ES"],
    ).round(0)
