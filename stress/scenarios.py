"""Stress scenarios: prescribed shock vectors applied to the current book.

VaR and ES measure risk over a recent window; stress testing exists
because that window may simply not contain the relevant tail event. A
scenario applies a *known* set of shocks — historical (what 2008 did) or
hypothetical (a supervisory "+200bp parallel") — to today's positions and
asks what the book would lose. This complements frtb/stressed_calibration
(the empirical worst-250-day-window search): that finds the worst episode
in your data, this replays episodes that may pre-date your data (GFC) or
that you want to impose by hand.

A Scenario resolves to a shock vector in the engine's NATIVE change units
(log returns for spot/FX, bps for rates/credit, vol points for vol) — the
same units risk_factors.changes() produces — so the scenario engine can
reuse the existing pricing and sensitivity machinery unchanged.

The named historical magnitudes below are stylized representative
peak-to-trough moves, not exact prints. Where the episode is inside your
data (COVID, 2022) prefer from_historical_window, which extracts the
actual cumulative move; GFC (pre-2010) is only available as a prescribed
scenario.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from data.risk_factors import RiskFactorSet, RiskFactorType


_TENOR_RE = re.compile(r"(\d+)\s*Y", re.IGNORECASE)


def _tenor_key(rf_name: str) -> str | None:
    """Extract a tenor token like '2Y' / '10Y' from a rate RF name."""
    m = _TENOR_RE.search(rf_name)
    return f"{int(m.group(1))}Y" if m else None


@dataclass
class Scenario:
    """A resolved stress scenario: native-unit shocks per risk factor."""
    name: str
    description: str
    source: str                       # "historical" | "hypothetical"
    shocks_native: pd.Series          # rf_name -> change in native units

    def covered(self) -> list[str]:
        """Risk factors actually shocked (non-zero)."""
        return list(self.shocks_native[self.shocks_native != 0.0].index)


def from_spec(
    rfs: RiskFactorSet,
    name: str,
    description: str,
    *,
    equity: float = 0.0,              # simple return, e.g. -0.35 for -35%
    fx: float = 0.0,                  # simple return on the FX rate
    vol_points: float = 0.0,          # additive vol points (VIX 15->80 = +65)
    credit_bp: float = 0.0,           # additive bps
    rates_bp: float | dict[str, float] = 0.0,  # scalar=parallel, or {tenor: bp}
    source: str = "hypothetical",
) -> Scenario:
    """Build a scenario from intuitive shocks, resolved to native units
    against the given risk-factor set.

    Conversions: equity/FX simple returns -> log returns ln(1+r); vol points
    and bps are already native; rates accept a parallel scalar or a per-tenor
    dict keyed like {'2Y': -250, '10Y': -150}.
    """
    shocks = {n: 0.0 for n in rfs.names}
    eq_log = float(np.log1p(equity)) if equity else 0.0
    fx_log = float(np.log1p(fx)) if fx else 0.0

    for rf in rfs.factors.values():
        if rf.rf_type == RiskFactorType.EQUITY_SPOT:
            shocks[rf.name] = eq_log
        elif rf.rf_type == RiskFactorType.FX_SPOT:
            shocks[rf.name] = fx_log
        elif rf.rf_type == RiskFactorType.EQUITY_VOL:
            shocks[rf.name] = float(vol_points)
        elif rf.rf_type == RiskFactorType.CREDIT_SPREAD:
            shocks[rf.name] = float(credit_bp)
        elif rf.rf_type == RiskFactorType.RATE:
            if isinstance(rates_bp, dict):
                key = _tenor_key(rf.name)
                shocks[rf.name] = float(rates_bp.get(key, 0.0))
            else:
                shocks[rf.name] = float(rates_bp)

    return Scenario(name, description, source, pd.Series(shocks, name=name))


def from_historical_window(
    rfs: RiskFactorSet,
    start: str,
    end: str,
    name: str | None = None,
    description: str | None = None,
) -> Scenario:
    """Extract the cumulative RF move over [start, end] as a scenario.

    Native changes are additive within their unit (log returns sum to the
    cumulative log return; bp and vol-point diffs sum to the net change), so
    summing daily changes over the window gives the episode's peak-to-trough
    shock when the window is set to the episode's start/end dates.
    """
    shocks = {}
    for n, rf in rfs.factors.items():
        ch = rf.changes().loc[start:end]
        shocks[n] = float(ch.sum()) if len(ch) else 0.0
    if not any(v != 0.0 for v in shocks.values()):
        raise ValueError(f"No risk-factor changes found in window [{start}, {end}].")
    nm = name or f"Historical {start}->{end}"
    desc = description or f"Cumulative RF move observed over {start} to {end}."
    return Scenario(nm, desc, "historical", pd.Series(shocks, name=nm))


def named_scenarios(rfs: RiskFactorSet) -> dict[str, Scenario]:
    """Standard library of prescribed scenarios resolved against `rfs`.

    Historical magnitudes are stylized peak-to-trough moves. The direction
    detail matters for the book: GFC and COVID are flight-to-quality (rates
    RALLY, a tailwind for long-duration bonds), whereas 2022 is a rate SHOCK
    (rates rise sharply, hitting the same bonds) — the engine surfaces that
    contrast in the by-risk-factor attribution.
    """
    return {
        "GFC_2008": from_spec(
            rfs, "GFC 2008 (acute)",
            "Lehman-phase flight to quality: equities and credit collapse, "
            "rates rally, vol spikes, USD bid.",
            equity=-0.45, rates_bp={"2Y": -250, "5Y": -200, "10Y": -150},
            credit_bp=+450, vol_points=+60, fx=-0.10, source="historical",
        ),
        "COVID_2020": from_spec(
            rfs, "COVID March 2020",
            "Pandemic crash: ~-34% equities in 5 weeks, emergency cuts, "
            "credit blowout, record vol spike.",
            equity=-0.34, rates_bp={"2Y": -120, "5Y": -110, "10Y": -100},
            credit_bp=+270, vol_points=+65, fx=-0.03, source="historical",
        ),
        "RATES_2022": from_spec(
            rfs, "2022 rate shock",
            "Inflation/hiking cycle: sharp bear-flattening rate selloff, "
            "equity derating, strong USD, modest credit widening.",
            equity=-0.25, rates_bp={"2Y": +375, "5Y": +300, "10Y": +240},
            credit_bp=+75, vol_points=+12, fx=-0.13, source="historical",
        ),
        "PARALLEL_RATES_+200": from_spec(
            rfs, "Parallel +200bp",
            "Supervisory IRRBB-style parallel upward rate shock; "
            "other factors held flat.",
            rates_bp=+200, source="hypothetical",
        ),
        "RISK_OFF_MODERATE": from_spec(
            rfs, "Moderate risk-off",
            "Equities -20%, vol +10 points; a contained correction.",
            equity=-0.20, vol_points=+10, source="hypothetical",
        ),
    }
