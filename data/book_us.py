"""US trading book: 14 positions across 5 risk factor families."""

from __future__ import annotations

from data.portfolio import (
    Portfolio, Position, Direction,
    EquitySpot, EquityOption, GovBond, FXSpot, CreditIndex,
)
from data.risk_factors import RiskFactorSet


def build_book(risk_factors: RiskFactorSet) -> Portfolio:
    """Build the US trading book against a RiskFactorSet from
    build_us_risk_factor_set()."""

    positions: list[Position] = []

    # ----- Equity directional book -----
    positions.append(Position(
        instrument=EquitySpot(name="AAPL long",  rf_spot="AAPL_SPOT"),
        notional=15_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="MSFT long",  rf_spot="MSFT_SPOT"),
        notional=12_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="NVDA long",  rf_spot="NVDA_SPOT"),
        notional=10_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="GOOGL long", rf_spot="GOOGL_SPOT"),
        notional=10_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquitySpot(name="JPM short",  rf_spot="JPM_SPOT"),
        notional=8_000_000,  direction=Direction.SHORT,
    ))

    # ----- Equity options -----
    positions.append(Position(
        instrument=EquityOption(
            name="AAPL 6M ATM call",
            rf_spot="AAPL_SPOT", rf_vol="VIX",
            strike=230.0, tenor_years=0.5, option_type="call",
        ),
        notional=5_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=EquityOption(
            name="MSFT 1Y 105% call",
            rf_spot="MSFT_SPOT", rf_vol="VIX",
            strike=460.0, tenor_years=1.0, option_type="call",
        ),
        notional=5_000_000, direction=Direction.LONG,
    ))

    # ----- Rates curve play (steepener) -----
    positions.append(Position(
        instrument=GovBond(
            name="UST 2Y short", issuer="US",
            coupon=0.045, maturity_years=2.0,
            rf_curve_nodes=["UST_2Y"],
        ),
        notional=15_000_000, direction=Direction.SHORT,
    ))
    positions.append(Position(
        instrument=GovBond(
            name="UST 5Y long", issuer="US",
            coupon=0.040, maturity_years=5.0,
            rf_curve_nodes=["UST_2Y", "UST_5Y"],
        ),
        notional=20_000_000, direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=GovBond(
            name="UST 10Y long", issuer="US",
            coupon=0.042, maturity_years=10.0,
            rf_curve_nodes=["UST_2Y", "UST_5Y", "UST_10Y"],
        ),
        notional=20_000_000, direction=Direction.LONG,
    ))

    # ----- FX -----
    positions.append(Position(
        instrument=FXSpot(name="EUR/USD short", rf_fx="EURUSD"),
        notional=10_000_000, direction=Direction.SHORT,
    ))
    positions.append(Position(
        instrument=FXSpot(name="USD/JPY long",  rf_fx="USDJPY"),
        notional=5_000_000,  direction=Direction.LONG,
    ))
    positions.append(Position(
        instrument=FXSpot(name="GBP/USD short", rf_fx="GBPUSD"),
        notional=5_000_000,  direction=Direction.SHORT,
    ))

    # ----- Credit -----
    positions.append(Position(
        instrument=CreditIndex(name="US IG short", rf_spread="US_IG_SPREAD"),
        notional=10_000_000, direction=Direction.SHORT,
    ))

    return Portfolio(positions=positions, risk_factors=risk_factors)
