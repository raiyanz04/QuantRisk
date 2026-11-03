"""EU trading book: 14 positions across 5 risk factor families."""

from __future__ import annotations

from data.portfolio import (
    Portfolio, Position, Direction,
    EquitySpot, EquityOption, GovBond, FXSpot, CreditIndex,
)
from data.risk_factors import RiskFactorSet


def build_book(risk_factors: RiskFactorSet) -> Portfolio:
    """Build the EU trading book against a RiskFactorSet from
    build_eu_risk_factor_set()."""

    positions: list[Position] = []

    # ----- Equity directional book -----
    positions.append(Position(
        instrument=EquitySpot(name="SAP long",   rf_spot="SAP_SPOT"),
        notional=15_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="ASML long",  rf_spot="ASML_SPOT"),
        notional=12_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="TTE long",   rf_spot="TTE_SPOT"),
        notional=10_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="LVMH long",  rf_spot="MC_SPOT"),
        notional=10_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="Allianz short", rf_spot="ALV_SPOT"),
        notional=8_000_000, direction=Direction.SHORT,
    ))

    # ----- Equity options -----
    positions.append(Position(
        instrument=EquityOption(
            name="SAP 6M ATM call",
            rf_spot="SAP_SPOT", rf_vol="SX5E_RV",
            strike=145.0, tenor_years=0.5, option_type="call",
        ),
        notional=5_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquityOption(
            name="ASML 1Y 105% call",
            rf_spot="ASML_SPOT", rf_vol="SX5E_RV",
            strike=735.0, tenor_years=1.0, option_type="call",
        ),
        notional=5_000_000, direction=Direction.LONG,
    ))

    # ----- Rates curve play -----
    positions.append(Position(
        instrument=GovBond(
            name="DE 5Y long", issuer="DE",
            coupon=0.025, maturity_years=5.0,
            rf_curve_nodes=["EUR_5Y"],
        ),
        notional=20_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=GovBond(
            name="FR 10Y long", issuer="FR",
            coupon=0.030, maturity_years=10.0,
            rf_curve_nodes=["EUR_2Y", "EUR_5Y", "EUR_10Y"],
        ),
        notional=20_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=GovBond(
            name="IT 10Y short", issuer="IT",
            coupon=0.035, maturity_years=10.0,
            rf_curve_nodes=["EUR_2Y", "EUR_5Y", "EUR_10Y"],
        ),
        notional=15_000_000, direction=Direction.SHORT,
    ))

    # ----- FX -----
    positions.append(Position(
        instrument=FXSpot(name="EUR/USD short", rf_fx="EURUSD"),
        notional=10_000_000, direction=Direction.SHORT,
    ))
    positions.append(Position(
        instrument=FXSpot(name="EUR/GBP long",  rf_fx="EURGBP"),
        notional=5_000_000,  direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=FXSpot(name="EUR/JPY long",  rf_fx="EURJPY"),
        notional=5_000_000,  direction=Direction.LONG,
    ))

    # ----- Credit -----
    positions.append(Position(
        instrument=CreditIndex(name="Euro HY short", rf_spread="EUR_HY_OAS"),
        notional=10_000_000, direction=Direction.SHORT,
    ))

    return Portfolio(positions=positions, risk_factors=risk_factors)
