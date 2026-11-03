"""Phase 6 — backtesting suite (coverage tests + capital multiplier)."""
from backtest.exceptions import ExceptionStream, generate_exceptions
from backtest.coverage_tests import (
    CoverageResult, run_coverage_tests, kupiec_pof, christoffersen_independence,
)
from backtest.traffic_light import (
    TrafficLightResult, traffic_light, IMAEligibility, ima_eligibility,
)
from backtest.backtest_engine import (
    BacktestSuiteResult, StreamBacktest, run_backtest_suite,
)

__all__ = [
    "ExceptionStream", "generate_exceptions",
    "CoverageResult", "run_coverage_tests", "kupiec_pof", "christoffersen_independence",
    "TrafficLightResult", "traffic_light", "IMAEligibility", "ima_eligibility",
    "BacktestSuiteResult", "StreamBacktest", "run_backtest_suite",
]
