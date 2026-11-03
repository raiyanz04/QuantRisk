"""Component-VaR (Euler) additivity and concentration metrics."""
import pytest

from attribution.component_var import parametric_var_attribution
from attribution.concentration import concentration


def test_component_var_sums_to_portfolio_var(balanced_book):
    """The delta-normal Euler decomposition is exact: sum of component VaR
    equals the portfolio VaR, and the percentage contributions sum to 1."""
    pf, sens = balanced_book
    attr = parametric_var_attribution(pf, sens, confidence=0.99, horizon_days=1)

    assert attr.component_var.sum() == pytest.approx(attr.portfolio_var, rel=1e-9)
    assert attr.pct_contribution.sum() == pytest.approx(1.0, rel=1e-9)


def test_concentration_metrics_well_formed_and_ordered(balanced_book, concentrated_book):
    pf_b, sens_b = balanced_book
    pf_c, sens_c = concentrated_book

    attr_b = parametric_var_attribution(pf_b, sens_b, 0.99, 1)
    attr_c = parametric_var_attribution(pf_c, sens_c, 0.99, 1)
    cm_b = concentration(attr_b)
    cm_c = concentration(attr_c)

    # Well-formed: HHI in (0, 1], effective_n = 1/HHI, shares <= 1.
    assert 0.0 < cm_b.hhi <= 1.0
    assert cm_b.effective_n == pytest.approx(1.0 / cm_b.hhi, rel=1e-9)
    assert cm_b.top1_share <= 1.0 + 1e-12

    # A balanced 3-name book sits near the 1/N = 1/3 diversified floor.
    assert cm_b.hhi == pytest.approx(1.0 / 3.0, abs=0.06)

    # Concentrating the book must raise HHI and the top-1 share.
    assert cm_c.hhi > cm_b.hhi
    assert cm_c.top1_share > cm_b.top1_share
    assert cm_c.largest_contributor == "P0"   # the 9x position
