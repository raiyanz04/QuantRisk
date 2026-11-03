"""Instruments, positions, and the portfolio container."""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from data.risk_factors import RiskFactor, RiskFactorSet


class Direction(int, Enum):
    LONG = 1
    SHORT = -1


# ---------- Instruments ----------

@dataclass
class Instrument:
    """Base instrument. Subclasses encode which risk factors drive value."""
    name: str

    def risk_factor_names(self) -> list[str]:
        raise NotImplementedError


@dataclass
class EquitySpot(Instrument):
    rf_spot: str

    def risk_factor_names(self) -> list[str]:
        return [self.rf_spot]


@dataclass
class EquityOption(Instrument):
    rf_spot: str
    rf_vol: str
    strike: float
    tenor_years: float
    option_type: str  # 'call' or 'put'

    def risk_factor_names(self) -> list[str]:
        return [self.rf_spot, self.rf_vol]


@dataclass
class GovBond(Instrument):
    issuer: str
    coupon: float
    maturity_years: float
    rf_curve_nodes: list[str] = field(default_factory=list)

    def risk_factor_names(self) -> list[str]:
        return list(self.rf_curve_nodes)


@dataclass
class FXSpot(Instrument):
    rf_fx: str

    def risk_factor_names(self) -> list[str]:
        return [self.rf_fx]


@dataclass
class CreditIndex(Instrument):
    rf_spread: str

    def risk_factor_names(self) -> list[str]:
        return [self.rf_spread]


# ---------- Positions & Portfolio ----------

@dataclass
class Position:
    instrument: Instrument
    notional: float
    direction: Direction = Direction.LONG

    @property
    def signed_notional(self) -> float:
        return self.notional * self.direction.value


@dataclass
class Portfolio:
    positions: list[Position]
    risk_factors: RiskFactorSet

    def __post_init__(self) -> None:
        missing: list[tuple[str, str]] = []
        for p in self.positions:
            for rf_name in p.instrument.risk_factor_names():
                if rf_name not in self.risk_factors:
                    missing.append((p.instrument.name, rf_name))
        if missing:
            raise ValueError(
                "Positions reference risk factors not in the RiskFactorSet:\n"
                + "\n".join(f"  - {inst}: missing {rf}" for inst, rf in missing)
            )

    def __len__(self) -> int:
        return len(self.positions)

    def risk_factor_names(self) -> set[str]:
        used: set[str] = set()
        for p in self.positions:
            used.update(p.instrument.risk_factor_names())
        return used

    def summary(self) -> dict[str, dict[str, float]]:
        gross: dict[str, float] = defaultdict(float)
        net: dict[str, float] = defaultdict(float)
        for p in self.positions:
            for rf_name in p.instrument.risk_factor_names():
                rf_type = self.risk_factors[rf_name].rf_type.value
                gross[rf_type] += abs(p.signed_notional)
                net[rf_type] += p.signed_notional
        return {"gross": dict(gross), "net": dict(net)}
