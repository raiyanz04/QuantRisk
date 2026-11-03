"""Risk factor primitives.

A risk factor is the lowest-level driver of P&L: an equity spot price, a
yield-curve key rate, an FX rate, a credit spread, or an implied vol surface
point. Instruments are *consumers* of risk factors; every position in the book
maps to a sensitivity vector across the risk factors it depends on.

This module defines:
    - RiskFactorType : the five families covered by the engine
    - RiskFactor     : a single risk factor with its historical series
    - RiskFactorSet  : a collection that produces aligned joint matrices

The downstream contract is: every VaR / ES / stress engine consumes daily
*changes* of risk factors (not levels), in the natural scale for each type
(log returns for spots, bps for rates and spreads, vol points for vols).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


class RiskFactorType(str, Enum):
    """The five risk factor families covered by the engine."""
    EQUITY_SPOT = "equity_spot"          # changes computed as log returns
    EQUITY_VOL = "equity_vol"            # changes computed as vol-point diffs
    RATE = "rate"                        # changes computed as bp diffs
    FX_SPOT = "fx_spot"                  # changes computed as log returns
    CREDIT_SPREAD = "credit_spread"      # changes computed as bp diffs


# FRTB liquidity horizons (BCBS d457 MAR33.12), in business days.
LIQUIDITY_HORIZONS: dict[RiskFactorType, int] = {
    RiskFactorType.EQUITY_SPOT: 10,      # large-cap; small-cap = 20
    RiskFactorType.EQUITY_VOL: 20,
    RiskFactorType.RATE: 10,             # EUR/USD/GBP/JPY; EM = 60
    RiskFactorType.FX_SPOT: 10,          # majors; non-majors = 20
    RiskFactorType.CREDIT_SPREAD: 40,    # IG; HY = 60
}


@dataclass
class RiskFactor:
    """A single risk factor and its historical series.

    Rates and credit spreads stored in *decimal* (0.0245 for 2.45%), not bps.
    Vols stored in decimal (0.18 for 18%).
    """
    name: str
    rf_type: RiskFactorType
    series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def current(self) -> float:
        return float(self.series.iloc[-1])

    @property
    def liquidity_horizon(self) -> int:
        return LIQUIDITY_HORIZONS[self.rf_type]

    def changes(self) -> pd.Series:
        """Daily changes in the natural scale for this risk factor type.

        - Spot prices (equity, FX): log returns, dimensionless.
        - Rates and credit spreads: absolute differences in basis points.
        - Implied vols: absolute differences in vol points.
        """
        if self.rf_type in (RiskFactorType.EQUITY_SPOT, RiskFactorType.FX_SPOT):
            return np.log(self.series / self.series.shift(1)).dropna()
        elif self.rf_type in (RiskFactorType.RATE, RiskFactorType.CREDIT_SPREAD):
            return (self.series.diff() * 10000).dropna()
        elif self.rf_type == RiskFactorType.EQUITY_VOL:
            return (self.series.diff() * 100).dropna()
        else:
            raise ValueError(f"Unknown risk factor type: {self.rf_type}")


class RiskFactorSet:
    """Collection of risk factors with aligned date index."""

    def __init__(self, factors: list[RiskFactor]) -> None:
        self.factors: dict[str, RiskFactor] = {f.name: f for f in factors}

    def __getitem__(self, name: str) -> RiskFactor:
        return self.factors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.factors

    def __len__(self) -> int:
        return len(self.factors)

    @property
    def names(self) -> list[str]:
        return list(self.factors.keys())

    def by_type(self, rf_type: RiskFactorType) -> list[RiskFactor]:
        return [f for f in self.factors.values() if f.rf_type == rf_type]

    def changes_matrix(self) -> pd.DataFrame:
        """Joint daily changes for all risk factors, aligned on the
        intersection of available dates."""
        return pd.concat(
            {name: f.changes() for name, f in self.factors.items()},
            axis=1,
        ).dropna()

    def covariance(self) -> pd.DataFrame:
        return self.changes_matrix().cov()

    def correlation(self) -> pd.DataFrame:
        return self.changes_matrix().corr()
