"""VaR/ES distribution maths and the FRTB ES liquidity-horizon aggregation."""
import numpy as np
import pytest
from scipy.stats import norm

from var.types import VaRResult, fill_var_es_from_distribution
from frtb.expected_shortfall import compute_frtb_es, _compute_es_at_confidence
from greeks.pnl_attribution import hypothetical_pnl


# --------------------------------------------------------------------------
# Core VaR/ES from a P&L distribution
# --------------------------------------------------------------------------
def test_es_ge_var_monotone_and_sqrt_time(normal_pnl):
    res = VaRResult(method="hist", n_scenarios=len(normal_pnl), pnl_distribution=normal_pnl)
    fill_var_es_from_distribution(res, confidence_levels=(0.95, 0.99), horizons_days=(1, 10))

    for key in res.var:
        assert res.var[key] > 0.0
        assert res.es[key] >= res.var[key]                       # ES never below VaR

    # Higher confidence -> larger loss
    assert res.var[(0.99, 1)] > res.var[(0.95, 1)]
    # sqrt-time scaling is exact by construction
    assert res.var[(0.99, 10)] == pytest.approx(res.var[(0.99, 1)] * np.sqrt(10), rel=1e-12)
    assert res.es[(0.99, 10)] == pytest.approx(res.es[(0.99, 1)] * np.sqrt(10), rel=1e-12)


def test_es_var_ratio_matches_gaussian(normal_pnl):
    """On a Gaussian sample the empirical ES/VaR ratio matches phi(z)/((1-a)z)."""
    res = VaRResult(method="hist", n_scenarios=len(normal_pnl), pnl_distribution=normal_pnl)
    fill_var_es_from_distribution(res, confidence_levels=(0.99,), horizons_days=(1,))

    z = norm.ppf(0.99)
    expected_ratio = norm.pdf(z) / ((1 - 0.99) * z)
    empirical_ratio = res.es[(0.99, 1)] / res.var[(0.99, 1)]
    assert empirical_ratio == pytest.approx(expected_ratio, rel=0.05)


def test_compute_es_at_confidence_is_tail_mean():
    """_compute_es_at_confidence returns the positive mean of the lower tail."""
    pnl = __import__("pandas").Series(np.linspace(-100, 99, 200))  # -100..99
    # 5% tail of 200 obs = 10 worst = -100..-91 -> mean -95.5 -> ES = 95.5
    es = _compute_es_at_confidence(pnl, confidence=0.95)
    assert es == pytest.approx(95.5, rel=1e-9)


# --------------------------------------------------------------------------
# FRTB ES — liquidity-horizon aggregation
# --------------------------------------------------------------------------
def test_frtb_single_band_is_one_day_es_scaled_to_base_horizon(balanced_book):
    """An all-equity book lives in one 10-day LH band, so the FRTB ES is just
    the 1-day band ES lifted to the 10-day base horizon by sqrt(10)."""
    pf, sens = balanced_book
    res = compute_frtb_es(pf, sens, input_horizon_days=1)

    # Equity spot -> single liquidity-horizon band (10 days).
    assert set(res.es_per_band.keys()) == {10}

    pnl_full = hypothetical_pnl(pf, sens).total
    es_1d = _compute_es_at_confidence(pnl_full, res.confidence)
    assert res.es_per_band[10] == pytest.approx(es_1d * np.sqrt(10), rel=1e-9)

    # With one band the square-root aggregation is sqrt(ES_1^2) = ES_1.
    assert res.es_aggregated == pytest.approx(res.es_per_band[10], rel=1e-12)
    assert res.confidence == pytest.approx(0.975)
