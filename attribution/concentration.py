"""Concentration metrics on a VaR attribution.

Where aggregate VaR is the size of the risk and component VaR is its
location, concentration is how a CRO sets and polices limits. All metrics
are computed on the component VaR of a VaRAttribution:

  - Herfindahl-Hirschman Index (HHI): sum of squared risk shares, on the
    ABSOLUTE component contributions (so a hedge with negative component VaR
    contributes risk mass rather than cancelling). 1/n = perfectly spread,
    1.0 = a single position carries everything.
  - Effective number of positions: 1/HHI, the diversified-equivalent count.
  - Diversification ratio: sum of standalone VaRs / portfolio VaR. 1.0 means
    no diversification (perfectly correlated); higher means more benefit.
  - Top-1 / Top-3 share of total absolute component VaR, and the single
    largest contributor by name.
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from attribution.component_var import VaRAttribution


@dataclass
class ConcentrationMetrics:
    n_positions: int
    hhi: float
    effective_n: float
    diversification_ratio: float
    top1_share: float
    top3_share: float
    largest_contributor: str

    def summary(self) -> str:
        return (
            "Concentration (on component VaR)\n"
            f"  positions:            {self.n_positions}\n"
            f"  HHI:                  {self.hhi:.3f}  "
            f"(effective {self.effective_n:.1f} positions)\n"
            f"  diversification ratio: {self.diversification_ratio:.2f}x\n"
            f"  top-1 / top-3 share:  {self.top1_share:.1%} / {self.top3_share:.1%}\n"
            f"  largest contributor:  {self.largest_contributor}"
        )


def concentration(attr: VaRAttribution) -> ConcentrationMetrics:
    cvar = attr.component_var
    abs_c = cvar.abs()
    total_abs = float(abs_c.sum())

    if total_abs == 0.0:
        return ConcentrationMetrics(len(cvar), float("nan"), float("nan"),
                                    float("nan"), 0.0, 0.0, "")

    shares = abs_c / total_abs
    hhi = float((shares ** 2).sum())
    ranked = abs_c.sort_values(ascending=False)
    div_ratio = (float(attr.standalone_var.sum()) / attr.portfolio_var
                 if attr.portfolio_var else float("nan"))

    return ConcentrationMetrics(
        n_positions=len(cvar),
        hhi=hhi,
        effective_n=1.0 / hhi,
        diversification_ratio=div_ratio,
        top1_share=float(ranked.iloc[:1].sum() / total_abs),
        top3_share=float(ranked.iloc[:3].sum() / total_abs),
        largest_contributor=str(ranked.index[0]),
    )
