"""Backtesting suite facade: dual-stream, multi-confidence.

Mirrors var_engine.run_all_methods. Runs the full FRTB-style backtest:
the desk's 1-day VaR (estimated from the risk-model / RTPL series) is
tested against BOTH the hypothetical (HPL, full-reval) and
risk-theoretical (RTPL) P&L streams, at 97.5% and 99%, producing:

  - per-(stream, confidence) coverage tests (Kupiec + Christoffersen),
  - the Basel and FRTB capital multipliers from the trailing-250 99%
    exception count (binding = greater of the two streams),
  - the FRTB hard IMA-eligibility verdict.

Why RTPL drives the VaR but both streams are tested: under FRTB there is
ONE desk VaR, checked against actual and hypothetical P&L. Here the VaR
model is the risk-model (RTPL) distribution; testing it against HPL
surfaces the non-linearity gap (HPL breaches an RTPL-calibrated VaR
wherever the Taylor model under-prices the tail). NB: the regulatory
pair is APL + HPL; a synthetic engine has no booked APL, so HPL is the
regulatory stream and RTPL is the second, informative one.
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from backtest.exceptions import ExceptionStream, generate_exceptions
from backtest.coverage_tests import CoverageResult, run_coverage_tests
from backtest.traffic_light import (
    TrafficLightResult, traffic_light, IMAEligibility, ima_eligibility,
)


@dataclass
class StreamBacktest:
    stream: str
    confidence: float
    exceptions: ExceptionStream
    coverage: CoverageResult
    recent_exceptions: int          # over the regulatory window


@dataclass
class BacktestSuiteResult:
    per_stream: list[StreamBacktest]
    traffic: TrafficLightResult
    ima: IMAEligibility
    eval_window: int

    def comparison(self) -> pd.DataFrame:
        rows = []
        for sb in self.per_stream:
            c = sb.coverage
            rows.append({
                "stream":       sb.stream,
                "conf":         sb.confidence,
                "n_obs":        c.n_obs,
                "exc":          c.n_exceptions,
                "obs_rate":     round(c.observed_rate, 4),
                "exp_rate":     round(c.expected_rate, 4),
                "LR_uc":        round(c.lr_uc, 3),
                "p_uc":         round(c.p_uc, 4),
                "LR_ind":       round(c.lr_ind, 3),
                "p_ind":        round(c.p_ind, 4),
                "LR_cc":        round(c.lr_cc, 3),
                "p_cc":         round(c.p_cc, 4),
                f"last{self.eval_window}": sb.recent_exceptions,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        return "\n".join([self.traffic.summary(), "", self.ima.summary()])


def run_backtest_suite(
    rtpl: pd.Series,
    hpl: pd.Series,
    confidence_levels: tuple[float, ...] = (0.975, 0.99),
    window: int = 250,
    var_method: str = "historical",
    eval_window: int = 250,
) -> BacktestSuiteResult:
    """Run the dual-stream backtest at each confidence level.

    The VaR is estimated from `rtpl` (the risk-model P&L) and tested
    against both `hpl` and `rtpl`.
    """
    streams = {"HPL": hpl, "RTPL": rtpl}
    per_stream: list[StreamBacktest] = []
    recent_99: dict[str, int] = {}
    recent_975: dict[str, int] = {}

    for conf in confidence_levels:
        for sname, realized in streams.items():
            es = generate_exceptions(
                realized=realized, confidence=conf, window=window,
                var_method=var_method, var_source=rtpl,
                name=f"{sname} @ {conf:.1%}",
            )
            cov = run_coverage_tests(es.exceptions, conf)
            recent = es.recent(eval_window)
            per_stream.append(StreamBacktest(
                stream=sname, confidence=conf, exceptions=es,
                coverage=cov, recent_exceptions=recent.n_exceptions,
            ))
            if abs(conf - 0.99) < 1e-9:
                recent_99[sname] = recent.n_exceptions
            if abs(conf - 0.975) < 1e-9:
                recent_975[sname] = recent.n_exceptions

    # FRTB: count each stream separately; the binding count is the greater.
    binding_99 = max(recent_99.values()) if recent_99 else 0
    binding_975 = max(recent_975.values()) if recent_975 else 0

    return BacktestSuiteResult(
        per_stream=per_stream,
        traffic=traffic_light(binding_99, eval_window=eval_window),
        ima=ima_eligibility(binding_99, binding_975),
        eval_window=eval_window,
    )
