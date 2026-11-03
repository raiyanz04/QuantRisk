"""Basel and FRTB backtesting traffic lights -> capital multiplier.

This is the step that makes backtesting DRIVE capital: the number of 99%
1-day VaR exceptions over the most recent 250 trading days sets a
multiplier applied to the capital measure.

Two regimes are implemented in parallel (the engine builds Basel III and
FRTB IMA side by side):

  Basel 2.5 (VaR-based; BCBS "Supervisory framework for the use of
  backtesting", bcbs22). Base multiplier 3.0 plus a plus-factor:

      exceptions   0-4   5     6     7     8     9     >=10
      plus-factor  0.00  0.40  0.50  0.65  0.75  0.85  1.00
      multiplier   3.00  3.40  3.50  3.65  3.75  3.85  4.00
      zone         green --------- yellow ---------    red

  FRTB IMA (ES-based; MAR33). Multiplier mc = 1.5 + add-on, add-on in
  [0, 0.5], same green/amber/red boundaries at <=4 / 5-9 / >=10. Per-count
  amber add-ons per the MAR33.44 table:

      exceptions   0-4   5     6     7     8     9     >=10
      add-on       0.00  0.20  0.26  0.33  0.38  0.42  0.50
      mc           1.50  1.70  1.76  1.83  1.88  1.92  2.00

  NOTE: the zone structure, the 1.5 base and the 0.5 cap are firm; the
  intermediate amber add-on steps are the part worth cross-checking
  against your current MAR33.44 text before relying on them for a filing.

FRTB hard IMA gates (separate from the multiplier): a desk with more than
12 exceptions at 99% OR more than 30 at 97.5% over the trailing 250 days
loses IMA and must use the Standardised Approach. Exceptions are counted
on each P&L stream separately and the BINDING count is the greater.
"""
from __future__ import annotations
from dataclasses import dataclass


# Basel 2.5 plus-factor (BCBS bcbs22), base multiplier 3.0.
BASEL_BASE_MULTIPLIER = 3.0
_BASEL_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

# FRTB MAR33.44 multiplier add-on, base mc 1.5.
FRTB_BASE_MC = 1.5
_FRTB_ADDON = {5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}

# FRTB hard IMA-eligibility limits over the trailing 250 days.
FRTB_MAX_EXCEPTIONS_99 = 12
FRTB_MAX_EXCEPTIONS_975 = 30


def _zone(n: int) -> str:
    if n <= 4:
        return "green"
    if n <= 9:
        return "amber"
    return "red"


@dataclass
class TrafficLightResult:
    n_exceptions: int            # binding 99% count over the regulatory window
    eval_window: int
    zone: str
    basel_plus_factor: float
    basel_multiplier: float
    frtb_addon: float
    frtb_mc: float

    def summary(self) -> str:
        return (
            f"Traffic light ({self.n_exceptions} exceptions / {self.eval_window}d, "
            f"zone: {self.zone.upper()})\n"
            f"  Basel 2.5:  m  = {self.basel_multiplier:.2f}  "
            f"(3.00 + {self.basel_plus_factor:.2f})\n"
            f"  FRTB IMA:   mc = {self.frtb_mc:.2f}  "
            f"(1.50 + {self.frtb_addon:.2f})"
        )


def traffic_light(n_exceptions_99: int, eval_window: int = 250) -> TrafficLightResult:
    """Map the trailing-window 99% exception count to both multipliers."""
    n = int(n_exceptions_99)
    zone = _zone(n)
    if zone == "green":
        plus, addon = 0.0, 0.0
    elif zone == "amber":
        plus, addon = _BASEL_PLUS[n], _FRTB_ADDON[n]
    else:  # red
        plus, addon = 1.00, 0.50
    return TrafficLightResult(
        n_exceptions=n, eval_window=eval_window, zone=zone,
        basel_plus_factor=plus, basel_multiplier=BASEL_BASE_MULTIPLIER + plus,
        frtb_addon=addon, frtb_mc=FRTB_BASE_MC + addon,
    )


@dataclass
class IMAEligibility:
    eligible: bool
    reason: str
    n_99: int
    n_975: int

    def summary(self) -> str:
        status = "IMA RETAINED" if self.eligible else "IMA REVOKED -> Standardised Approach"
        return (
            f"{status}: {self.reason} "
            f"(99%: {self.n_99}, 97.5%: {self.n_975} over trailing window)"
        )


def ima_eligibility(n_99: int, n_975: int) -> IMAEligibility:
    """FRTB hard gate: >12 at 99% or >30 at 97.5% over 250d forces SA."""
    over99 = n_99 > FRTB_MAX_EXCEPTIONS_99
    over975 = n_975 > FRTB_MAX_EXCEPTIONS_975
    if over99 or over975:
        bits = []
        if over99:
            bits.append(f"{n_99} > {FRTB_MAX_EXCEPTIONS_99} at 99%")
        if over975:
            bits.append(f"{n_975} > {FRTB_MAX_EXCEPTIONS_975} at 97.5%")
        return IMAEligibility(False, "; ".join(bits), n_99, n_975)
    return IMAEligibility(
        True,
        f"within limits ({FRTB_MAX_EXCEPTIONS_99} at 99%, {FRTB_MAX_EXCEPTIONS_975} at 97.5%)",
        n_99, n_975,
    )
