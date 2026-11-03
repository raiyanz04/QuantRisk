"""Phase 7 — stress testing (prescribed/historical scenario engine).

Complements frtb/stressed_calibration.py (the empirical worst-250-day ES
calibration) with named scenario replays applied by full revaluation.
"""
from stress.scenarios import (
    Scenario, from_spec, from_historical_window, named_scenarios,
)
from stress.scenario_engine import (
    ScenarioResult, apply_scenario, run_scenario_suite,
)

__all__ = [
    "Scenario", "from_spec", "from_historical_window", "named_scenarios",
    "ScenarioResult", "apply_scenario", "run_scenario_suite",
]
