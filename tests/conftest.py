"""Shared fixtures for the market-risk-engine tests.

Everything is synthetic and in-memory: the real data layer pulls from
Yahoo/FRED, but the engine's *maths* (VaR/ES, the FRTB ES aggregation, the
Euler attribution, the coverage tests) is data-agnostic, so we drive it with
fixed-seed random walks and constructed P&L. That keeps the suite hermetic
and fast, with no API keys or network.

Books are all-equity-spot on purpose: spot positions have zero gamma/vega/rho/
theta, so the hypothetical-P&L pipeline reduces to delta * return and the
algebraic identities under test (additivity, sqrt-time, LH aggregation) are
exact rather than approximate.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.risk_factors import RiskFactor, RiskFactorSet, RiskFactorType
from data.portfolio import EquitySpot, Position, Portfolio, Direction
from greeks.sensitivities import build_sensitivities


def _equity_rfs(n=3, days=400, seed=0):
    idx = pd.bdate_range("2022-01-01", periods=days)
    rng = np.random.default_rng(seed)
    factors = []
    for i in range(n):
        rets = rng.normal(0.0, 0.012, size=days)
        px = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name=f"EQ{i}")
        factors.append(RiskFactor(name=f"EQ{i}", rf_type=RiskFactorType.EQUITY_SPOT,
                                  series=px))
    return RiskFactorSet(factors)


def _book(notionals, seed=0):
    rfs = _equity_rfs(n=len(notionals), seed=seed)
    positions = [
        Position(EquitySpot(name=f"P{i}", rf_spot=f"EQ{i}"),
                 notional=float(abs(notl)), direction=Direction.LONG)
        for i, notl in enumerate(notionals)
    ]
    pf = Portfolio(positions, rfs)
    sens = build_sensitivities(pf)
    return pf, sens


@pytest.fixture
def balanced_book():
    """Three equal-notional long equity positions in uncorrelated names."""
    return _book([1_000_000, 1_000_000, 1_000_000], seed=1)


@pytest.fixture
def concentrated_book():
    """One position ~9x the others, so concentration metrics should spike."""
    return _book([9_000_000, 1_000_000, 1_000_000], seed=1)


@pytest.fixture(scope="session")
def normal_pnl():
    """A large standard-ish P&L sample (mean 0) for VaR/ES distribution tests."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0, 100_000.0, size=200_000), name="pnl")
