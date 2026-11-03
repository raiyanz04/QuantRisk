"""Coverage backtests: Kupiec POF and Christoffersen independence / CC.

These are likelihood-ratio tests on the 0/1 exception series produced by
backtest.exceptions.

  - Kupiec POF (unconditional coverage). Does the exception RATE match the
    VaR level? H0: p = 1 - confidence. LR_uc ~ chi2(1).

  - Christoffersen independence. Are exceptions clustered in time? Fits a
    first-order Markov chain to the indicator and tests whether the
    probability of an exception depends on whether yesterday was an
    exception. H0: pi_01 = pi_11. LR_ind ~ chi2(1). This is the test most
    student projects skip, and the one regulators care about most: a model
    can have the right AVERAGE exception rate yet fail every time inside a
    single stress cluster — exactly the wrong time to fail.

  - Christoffersen conditional coverage. The joint test, and it decomposes
    EXACTLY as LR_cc = LR_uc + LR_ind ~ chi2(2). A desk can pass one
    component and fail the other; CC catches both at once.

All log-likelihood terms use scipy.special.xlogy, so the 0*log(0) = 0
boundary cases (no exceptions, no clustering) resolve cleanly rather than
producing NaNs. (As is standard, LR_uc is evaluated on all n observations
and LR_ind on the n-1 consecutive pairs; the additive decomposition is
the textbook form.)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import xlogy


def _as_array(exceptions) -> np.ndarray:
    if isinstance(exceptions, pd.Series):
        return exceptions.values.astype(int)
    return np.asarray(exceptions, dtype=int)


@dataclass
class CoverageResult:
    """Full coverage test battery on one exception series."""
    n_obs: int
    n_exceptions: int
    confidence: float
    expected_rate: float
    observed_rate: float
    lr_uc: float                    # Kupiec POF (unconditional coverage)
    p_uc: float
    lr_ind: float                   # Christoffersen independence
    p_ind: float
    lr_cc: float                    # conditional coverage (= uc + ind)
    p_cc: float
    n00: int                        # Markov transition counts (diagnostic)
    n01: int
    n10: int
    n11: int

    def verdict(self, alpha: float = 0.05) -> dict[str, str]:
        """Pass/reject at significance `alpha` for each test."""
        def crit(df):
            return chi2.ppf(1 - alpha, df)
        return {
            "kupiec_uc":    "reject" if self.lr_uc  > crit(1) else "pass",
            "independence": "reject" if self.lr_ind > crit(1) else "pass",
            "cond_cover":   "reject" if self.lr_cc  > crit(2) else "pass",
        }

    def summary(self) -> str:
        v = self.verdict()
        return (
            f"Coverage tests ({self.n_obs} obs, {self.n_exceptions} exceptions, "
            f"{self.observed_rate:.3%} vs {self.expected_rate:.3%} expected)\n"
            f"  Kupiec POF (uc):  LR={self.lr_uc:6.3f}  p={self.p_uc:.4f}  [{v['kupiec_uc']}]\n"
            f"  Independence:     LR={self.lr_ind:6.3f}  p={self.p_ind:.4f}  [{v['independence']}]\n"
            f"  Cond. coverage:   LR={self.lr_cc:6.3f}  p={self.p_cc:.4f}  [{v['cond_cover']}]"
        )


def kupiec_pof(exceptions, confidence: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR statistic and p-value. chi2(1)."""
    I = _as_array(exceptions)
    n = len(I)
    x = int(I.sum())
    p = 1.0 - confidence            # expected exception rate
    pi = x / n if n else 0.0        # observed rate (unconstrained MLE)

    ll_null = xlogy(x, p) + xlogy(n - x, 1 - p)        # rate fixed at p
    ll_mle = xlogy(x, pi) + xlogy(n - x, 1 - pi)       # rate = pi
    lr = max(-2.0 * (ll_null - ll_mle), 0.0)
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_independence(exceptions) -> tuple[float, float, dict[str, int]]:
    """Christoffersen Markov independence LR, p-value, and transition counts."""
    I = _as_array(exceptions)
    prev, cur = I[:-1], I[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}

    denom0, denom1 = n00 + n01, n10 + n11
    total = n00 + n01 + n10 + n11
    pi01 = n01 / denom0 if denom0 else 0.0
    pi11 = n11 / denom1 if denom1 else 0.0
    pi = (n01 + n11) / total if total else 0.0

    ll_markov = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
                 + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    ll_indep = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    lr = max(-2.0 * (ll_indep - ll_markov), 0.0)
    return float(lr), float(chi2.sf(lr, 1)), counts


def run_coverage_tests(exceptions, confidence: float) -> CoverageResult:
    """Run Kupiec POF + Christoffersen independence + conditional coverage."""
    I = _as_array(exceptions)
    n = len(I)
    x = int(I.sum())

    lr_uc, p_uc = kupiec_pof(I, confidence)
    lr_ind, p_ind, counts = christoffersen_independence(I)
    lr_cc = lr_uc + lr_ind                      # exact decomposition
    p_cc = float(chi2.sf(lr_cc, 2))

    return CoverageResult(
        n_obs=n, n_exceptions=x, confidence=confidence,
        expected_rate=1.0 - confidence,
        observed_rate=(x / n if n else float("nan")),
        lr_uc=lr_uc, p_uc=p_uc,
        lr_ind=lr_ind, p_ind=p_ind,
        lr_cc=lr_cc, p_cc=p_cc,
        **counts,
    )
