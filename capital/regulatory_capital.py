"""Phase 9 — regulatory capital: Basel 2.5 vs FRTB IMA, in parallel.

The closing of the loop. Every prior phase feeds one euro number:

  Basel 2.5 (VaR-based, MAR pre-FRTB internal models):
      K_basel = max(VaR, m_c * VaR_avg60) + max(SVaR, m_s * SVaR_avg60)
  with VaR and SVaR at 10-day / 99%, m = 3 + plus-factor (Phase 6).

  FRTB IMA (ES-based, MAR33):
      K_ima = max(IMCC, m_c * IMCC_avg60) + SES_NMRF
  with IMCC the stress-calibrated expected shortfall (97.5%, liquidity-
  horizon scaled, Phase 4/7), m_c = 1.5 + add-on (Phase 6), and SES the
  non-modellable-risk-factor add-on.

  FRTB SA (fallback): the Sensitivities-based Method charge (Phase 5).

Inputs are pulled from the existing engines, so the capital number is
exactly consistent with the rest of the system:
  VaR / SVaR  -> Phase 3 historical VaR + stressed window (Phase 7 dates)
  ES / SES    -> Phase 4 FRTB ES + Phase 7 stressed calibration
  multipliers -> Phase 6 backtesting traffic light
  IMA gate    -> Phase 6 eligibility
  SA          -> Phase 5 SbM

HORIZON NOTE. The FRTB ES now arrives at the 10-day base liquidity horizon
directly from Phase 4 — frtb/expected_shortfall.py scales the 1-day band ES
by sqrt(10) internally (MAR33). This module therefore consumes the current
and stressed ES as-is, with NO horizon scaling here. (Earlier versions
applied the sqrt(10) in this layer; it has moved to the source, so the two
files must stay in sync: if Phase 4's scaling is reverted, this layer would
under-scale the ES by sqrt(10).) The Basel VaR/SVaR are still scaled to
10-day here, since the historical-VaR engine returns them at 1-day.

SIMPLIFICATIONS to keep in mind when reading the comparison:
  - VaR_avg60 / IMCC_avg60 are proxied by the current value (no 60-day
    rolling average of the measure); since m > 1 the multiplier term binds,
    so K ~= m * (current measure), which is the standard project shortcut.
  - The SA number is the simplified single-bucket-per-class SbM from Phase 5
    (no curvature, flat within-bucket correlation), so it understates the
    true SA — do not over-read an SA-below-IMA ordering.
  - This book has no non-modellable risk factors (all liquid, observable),
    so SES_NMRF defaults to 0; the term is kept for completeness.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from frtb.actual_pnl import compute_hpl
from frtb.stressed_calibration import compute_stressed_es
from frtb.sbm_capital import compute_sbm_capital
from var.historical import compute_historical_var
from backtest.backtest_engine import run_backtest_suite

OUTPUT_FLOOR_PCT = 0.725          # Basel III endgame output floor on SA


@dataclass
class CapitalResult:
    # --- inputs (10-day / 99% for VaR, 97.5% LH-scaled for ES) ---
    var_10d: float
    svar_10d: float
    es_10d: float
    stressed_es_10d: float
    m_basel: float
    m_frtb: float
    nmrf: float
    sbm_sa: float
    ima_eligible: bool
    stress_window: tuple[str, str]
    # --- outputs ---
    basel_capital: float
    frtb_ima_capital: float
    frtb_sa_capital: float
    frtb_binding: float
    output_floor: float
    frtb_reported: float

    def comparison(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"regime": "Basel 2.5 (VaR+SVaR)", "multiplier": self.m_basel,
             "capital": round(self.basel_capital)},
            {"regime": "FRTB IMA (stressed ES)", "multiplier": self.m_frtb,
             "capital": round(self.frtb_ima_capital)},
            {"regime": "FRTB SA (SbM)", "multiplier": float("nan"),
             "capital": round(self.frtb_sa_capital)},
            {"regime": "FRTB reported (binding+floor)", "multiplier": float("nan"),
             "capital": round(self.frtb_reported)},
        ])

    def summary(self) -> str:
        ratio = self.frtb_ima_capital / self.basel_capital if self.basel_capital else float("nan")
        return (
            f"=== REGULATORY CAPITAL (10-day / 99% VaR basis) ===\n"
            f"Inputs:\n"
            f"  VaR (10d,99%):        {self.var_10d:>14,.0f}\n"
            f"  Stressed VaR (10d):   {self.svar_10d:>14,.0f}   "
            f"[worst window {self.stress_window[0]}->{self.stress_window[1]}]\n"
            f"  ES (10d,97.5%):       {self.es_10d:>14,.0f}\n"
            f"  Stressed ES / IMCC:   {self.stressed_es_10d:>14,.0f}\n"
            f"  m_basel={self.m_basel:.2f}  m_frtb={self.m_frtb:.2f}  "
            f"IMA={'eligible' if self.ima_eligible else 'REVOKED'}  NMRF={self.nmrf:,.0f}\n\n"
            f"Capital:\n"
            f"  Basel 2.5:            {self.basel_capital:>14,.0f}\n"
            f"  FRTB IMA:             {self.frtb_ima_capital:>14,.0f}   "
            f"({ratio:.0%} of Basel)\n"
            f"  FRTB SA (floor base): {self.frtb_sa_capital:>14,.0f}\n"
            f"  FRTB reported:        {self.frtb_reported:>14,.0f}   "
            f"(max of binding {('IMA' if self.ima_eligible else 'SA')} and "
            f"{OUTPUT_FLOOR_PCT:.1%} x SA = {self.output_floor:,.0f})"
        )


def compute_regulatory_capital(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    *,
    nmrf: float = 0.0,
    confidence_var: float = 0.99,
) -> CapitalResult:
    """Aggregate Basel 2.5 and FRTB IMA/SA capital from the existing engines."""
    rfs = portfolio.risk_factors
    rtpl = hypothetical_pnl(portfolio, sensitivities).total   # risk-theoretical (Taylor)
    hpl = compute_hpl(portfolio).total                        # hypothetical (full reval)

    # --- VaR (10-day, 99%) from Phase 3 historical simulation ---
    hv = compute_historical_var(
        portfolio, sensitivities,
        confidence_levels=(confidence_var,), horizons_days=(10,))
    var_10d = float(hv.var[(confidence_var, 10)])

    # --- Stressed period (Phase 7) -> stressed ES and stressed VaR ---
    ses = compute_stressed_es(portfolio, sensitivities)
    es_10d = ses.current_es                  # already 10-day base (Phase 4 scales)
    stressed_es_10d = ses.stressed_es        # IMCC, 10-day base

    sw = rtpl.loc[ses.stressed_window_start:ses.stressed_window_end]
    svar_1d = -float(sw.quantile(1.0 - confidence_var))
    svar_10d = svar_1d * float(np.sqrt(10))

    # --- multipliers and IMA gate (Phase 6) ---
    bt = run_backtest_suite(rtpl, hpl)
    m_basel = bt.traffic.basel_multiplier
    m_frtb = bt.traffic.frtb_mc
    ima_ok = bt.ima.eligible

    # --- SA fallback (Phase 5) ---
    sbm_sa = float(compute_sbm_capital(portfolio, sensitivities).total)

    # --- capital ---
    basel = max(var_10d, m_basel * var_10d) + max(svar_10d, m_basel * svar_10d)
    frtb_ima = max(stressed_es_10d, m_frtb * stressed_es_10d) + nmrf
    frtb_sa = sbm_sa
    frtb_binding = frtb_ima if ima_ok else frtb_sa
    floor = OUTPUT_FLOOR_PCT * frtb_sa
    frtb_reported = max(frtb_binding, floor)

    return CapitalResult(
        var_10d=var_10d, svar_10d=svar_10d, es_10d=es_10d,
        stressed_es_10d=stressed_es_10d,
        m_basel=m_basel, m_frtb=m_frtb, nmrf=nmrf, sbm_sa=sbm_sa,
        ima_eligible=ima_ok,
        stress_window=(str(ses.stressed_window_start)[:10],
                       str(ses.stressed_window_end)[:10]),
        basel_capital=basel, frtb_ima_capital=frtb_ima, frtb_sa_capital=frtb_sa,
        frtb_binding=frtb_binding, output_floor=floor, frtb_reported=frtb_reported,
    )
