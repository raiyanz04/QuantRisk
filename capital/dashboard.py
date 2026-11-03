"""Phase 9 — six-panel risk dashboard.

Risk numbers without a view aren't usable on a desk. This renders the
whole engine onto one page, mirroring a sell-side market-risk report:

    1  Live risk        VaR (four methods) and ES, today          (Ph 3/4)
    2  P&L attribution  risk-theoretical vs hypothetical P&L       (Ph 2/5)
    3  Backtesting      rolling VaR vs realised P&L, breaches      (Ph 6)
    4  Capital          Basel 2.5 vs FRTB IMA vs SA                (Ph 9)
    5  Stress           named-scenario losses, worst first         (Ph 7)
    6  Attribution      component VaR by position                  (Ph 8)

build_dashboard runs the engines once and writes a PNG. Pass a precomputed
CapitalResult to avoid re-running the (slow) stressed-ES calibration.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from data.portfolio import Portfolio
from greeks.sensitivities import SensitivityMatrix
from greeks.pnl_attribution import hypothetical_pnl
from frtb.actual_pnl import compute_hpl
from var.var_engine import run_all_methods
from backtest.exceptions import generate_exceptions
from stress.scenarios import named_scenarios
from stress.scenario_engine import run_scenario_suite
from attribution.attribution_engine import run_attribution
from capital.regulatory_capital import CapitalResult, compute_regulatory_capital

# Palette
NAVY, STEEL, AMBER, RED, GREEN, GREY = (
    "#1f3a5f", "#4a7ba6", "#e0a458", "#c1462f", "#3f7d55", "#9aa5b1")
_MFMT = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")


def _panel_live_risk(ax, results):
    methods = [r.method.replace("_", "\n") for r in results]
    var_v = [r.var[(0.99, 1)] for r in results]
    es_v = [r.es[(0.99, 1)] for r in results]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, var_v, 0.4, label="VaR 99%", color=NAVY)
    ax.bar(x + 0.2, es_v, 0.4, label="ES 99%", color=STEEL)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=7)
    ax.yaxis.set_major_formatter(_MFMT)
    ax.set_title("1  Live risk — 1-day VaR/ES by method", fontsize=10, weight="bold")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(axis="y", alpha=0.3)


def _panel_pnl_attribution(ax, rtpl, hpl):
    from scipy.stats import spearmanr
    rho, _ = spearmanr(rtpl.values, hpl.values)
    ax.scatter(rtpl.values / 1e6, hpl.values / 1e6, s=4, alpha=0.25, color=STEEL)
    lim = max(abs(rtpl).max(), abs(hpl).max()) / 1e6
    ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1.0, ls="--")
    ax.set_xlabel("RTPL (Taylor) $M", fontsize=8)
    ax.set_ylabel("HPL (full reval) $M", fontsize=8)
    ax.set_title(f"2  P&L attribution — Spearman {rho:.4f}", fontsize=10, weight="bold")
    ax.grid(alpha=0.3)


def _panel_backtesting(ax, rtpl, n=500):
    es = generate_exceptions(realized=rtpl, confidence=0.99, window=250, var_source=rtpl)
    real = es.realized.iloc[-n:]
    vfc = es.var_forecast.iloc[-n:]
    exc = es.exceptions.iloc[-n:].astype(bool)
    ax.plot(real.index, real.values / 1e6, color=GREY, lw=0.5)
    ax.plot(vfc.index, -vfc.values / 1e6, color=NAVY, lw=1.1, label="-VaR 99%")
    ax.scatter(real.index[exc.values], real.values[exc.values] / 1e6,
               color=RED, s=14, zorder=5, label=f"breaches ({int(exc.sum())})")
    ax.set_title(f"3  Backtesting — last {n} days", fontsize=10, weight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelsize=7)


def _panel_capital(ax, cap: CapitalResult):
    labels = ["Basel 2.5", "FRTB IMA", "FRTB SA"]
    vals = [cap.basel_capital, cap.frtb_ima_capital, cap.frtb_sa_capital]
    colors = [GREY, NAVY, STEEL]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v/1e6:.0f}M",
                ha="center", va="bottom", fontsize=8, weight="bold")
    ax.yaxis.set_major_formatter(_MFMT)
    ax.set_title("4  Capital — regime comparison", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.18)


def _panel_stress(ax, table):
    t = table.sort_values("full_reval")          # worst (most negative) first
    names = t["scenario"].tolist()
    vals = (t["full_reval"] / 1e6).tolist()
    ax.barh(names, vals, color=[RED if v < 0 else GREEN for v in vals])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}M"))
    ax.set_title("5  Stress — scenario P&L (full reval)", fontsize=10, weight="bold")
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="x", alpha=0.3)


def _panel_attribution(ax, rep):
    cv = rep.parametric.component_var.sort_values()
    ax.barh(cv.index, cv.values / 1e6,
            color=[RED if v < 0 else NAVY for v in cv.values])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.1f}M"))
    ax.set_title(f"6  Attribution — component VaR  "
                 f"(HHI {rep.concentration.hhi:.2f})", fontsize=10, weight="bold")
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="x", alpha=0.3)


def build_dashboard(
    portfolio: Portfolio,
    sensitivities: SensitivityMatrix,
    save_path: str = "risk_dashboard.png",
    capital: CapitalResult | None = None,
) -> str:
    """Render the six-panel dashboard to `save_path`; returns the path."""
    rfs = portfolio.risk_factors
    rtpl = hypothetical_pnl(portfolio, sensitivities).total
    hpl = compute_hpl(portfolio).total

    results, _ = run_all_methods(
        portfolio, sensitivities, confidence_levels=(0.99,), horizons_days=(1,))
    _, stress_table = run_scenario_suite(portfolio, sensitivities, named_scenarios(rfs))
    attribution = run_attribution(portfolio, sensitivities, confidence=0.99)
    if capital is None:
        capital = compute_regulatory_capital(portfolio, sensitivities)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    fig.suptitle(
        f"Market Risk Dashboard  —  {len(portfolio)} positions  —  "
        f"FRTB IMA capital \\${capital.frtb_ima_capital/1e6:.0f}M  "
        f"(Basel 2.5 \\${capital.basel_capital/1e6:.0f}M)",
        fontsize=13, weight="bold")

    _panel_live_risk(axes[0, 0], results)
    _panel_pnl_attribution(axes[0, 1], rtpl, hpl)
    _panel_backtesting(axes[0, 2], rtpl)
    _panel_capital(axes[1, 0], capital)
    _panel_stress(axes[1, 1], stress_table)
    _panel_attribution(axes[1, 2], attribution)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
