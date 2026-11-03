"""Backtesting battery: Kupiec POF, Christoffersen, traffic light, IMA gate."""
import numpy as np
import pytest
from scipy.stats import chi2

from backtest.coverage_tests import (kupiec_pof, christoffersen_independence,
                                     run_coverage_tests)
from backtest.traffic_light import traffic_light, ima_eligibility


# --------------------------------------------------------------------------
# Kupiec proportion-of-failures
# --------------------------------------------------------------------------
def test_kupiec_zero_when_rate_matches_var_level():
    """If the observed exception rate equals 1 - confidence, LR_uc = 0."""
    n, conf = 1000, 0.99
    exc = np.zeros(n, dtype=int)
    exc[:10] = 1                       # 10/1000 = 1% = 1 - 0.99
    lr, p = kupiec_pof(exc, conf)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_kupiec_rejects_excess_exceptions():
    n, conf = 1000, 0.99
    exc = np.zeros(n, dtype=int)
    exc[:50] = 1                       # 5% observed vs 1% expected -> reject
    lr, p = kupiec_pof(exc, conf)
    assert lr >= 0.0
    assert lr > chi2.ppf(0.95, 1)      # beyond the 5% critical value
    assert p < 0.05


# --------------------------------------------------------------------------
# Christoffersen independence + conditional-coverage decomposition
# --------------------------------------------------------------------------
def test_christoffersen_counts_and_cc_decomposition():
    rng = np.random.default_rng(7)
    exc = (rng.random(2000) < 0.01).astype(int)

    lr_ind, p_ind, counts = christoffersen_independence(exc)
    assert lr_ind >= 0.0
    assert counts["n00"] + counts["n01"] + counts["n10"] + counts["n11"] == len(exc) - 1

    res = run_coverage_tests(exc, confidence=0.99)
    # The textbook identity LR_cc = LR_uc + LR_ind must hold exactly.
    assert res.lr_cc == pytest.approx(res.lr_uc + res.lr_ind, rel=1e-12)


def test_christoffersen_zero_with_no_exceptions():
    exc = np.zeros(500, dtype=int)
    lr_ind, _, _ = christoffersen_independence(exc)
    assert lr_ind == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Basel/FRTB traffic light and the FRTB IMA hard gate
# --------------------------------------------------------------------------
def test_traffic_light_zones_and_multipliers():
    green = traffic_light(4)
    assert green.zone == "green"
    assert green.basel_multiplier == pytest.approx(3.00)
    assert green.frtb_mc == pytest.approx(1.50)

    amber6 = traffic_light(6)
    assert amber6.zone == "amber"
    assert amber6.basel_multiplier == pytest.approx(3.50)   # 3.00 + 0.50
    assert amber6.frtb_mc == pytest.approx(1.76)            # 1.50 + 0.26

    amber9 = traffic_light(9)
    assert amber9.zone == "amber"
    assert amber9.basel_multiplier == pytest.approx(3.85)   # 3.00 + 0.85
    assert amber9.frtb_mc == pytest.approx(1.92)            # 1.50 + 0.42

    red = traffic_light(10)
    assert red.zone == "red"
    assert red.basel_multiplier == pytest.approx(4.00)      # 3.00 + 1.00
    assert red.frtb_mc == pytest.approx(2.00)               # 1.50 + 0.50


def test_ima_eligibility_hard_limits():
    assert ima_eligibility(12, 30).eligible is True         # exactly at the limits
    assert ima_eligibility(13, 0).eligible is False         # breach at 99%
    assert ima_eligibility(0, 31).eligible is False         # breach at 97.5%
