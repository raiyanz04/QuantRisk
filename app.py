"""Market Risk Engine — interactive dashboard.

Renders the same six panels as `capital/dashboard.py` (the static PNG in the
README), but as a live, interactive app. Backed entirely by the existing
engine modules; the bundled `data_cache/` parquet/pkl files mean nothing
fetches from the network and no API keys are required.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- engine modules (same surface phase9_demo / capital.dashboard use) ---
from data.market_data import build_us_risk_factor_set
from data.book_us import build_book
from greeks.sensitivities import build_sensitivities
from greeks.pnl_attribution import hypothetical_pnl
from frtb.actual_pnl import compute_hpl
from frtb.pla_test import run_pla_test
from var.var_engine import run_all_methods
from backtest.exceptions import generate_exceptions
from stress.scenarios import named_scenarios
from stress.scenario_engine import run_scenario_suite
from attribution.attribution_engine import run_attribution
from capital.regulatory_capital import compute_regulatory_capital

# --- palette: matches the matplotlib dashboard PNG so the static and ---
# --- live views read as one product, not two ---
NAVY  = "#1f3a5f"
STEEL = "#4a7ba6"
AMBER = "#e0a458"
RED   = "#c1462f"
GREEN = "#3f7d55"
GREY  = "#9aa5b1"

PLOTLY_LAYOUT = dict(
    font=dict(family="Source Sans Pro, Helvetica, Arial, sans-serif", size=12),
    margin=dict(t=20, b=20, l=10, r=10),
    plot_bgcolor="white",
)
AXIS_STYLE = dict(showgrid=True, gridcolor="#e8eaed", zeroline=True, zerolinecolor=GREY)

st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================================
#  Cached engine state — heavy work happens once per session
# =========================================================================
@st.cache_resource(show_spinner="Loading risk factor set + portfolio + sensitivities …")
def load_engine(start: str = "2010-01-01", end: str = "2024-12-31"):
    rfs = build_us_risk_factor_set(start=start, end=end, use_cache=True)
    pf = build_book(rfs)
    sens = build_sensitivities(pf)
    return rfs, pf, sens


@st.cache_data(show_spinner="Computing P&L streams (RTPL + HPL) …")
def get_pnl_streams(_pf, _sens):
    rtpl = hypothetical_pnl(_pf, _sens).total
    hpl = compute_hpl(_pf).total
    return rtpl, hpl


@st.cache_data(show_spinner="Running VaR across all four methods …")
def get_var_results(_pf, _sens, confidence_levels: tuple, horizons: tuple):
    results, _ = run_all_methods(
        _pf, _sens,
        confidence_levels=confidence_levels,
        horizons_days=horizons,
    )
    return results


@st.cache_data(show_spinner="Calibrating regulatory capital (stressed-ES over 15y) …")
def get_capital(_pf, _sens):
    return compute_regulatory_capital(_pf, _sens)


@st.cache_data(show_spinner="Running stress scenario suite (full revaluation) …")
def get_stress_table(_pf, _sens, _rfs):
    _, table = run_scenario_suite(_pf, _sens, named_scenarios(_rfs))
    return table


@st.cache_data(show_spinner="Computing VaR attribution …")
def get_attribution(_pf, _sens, confidence: float = 0.99):
    return run_attribution(_pf, _sens, confidence=confidence)


@st.cache_data(show_spinner="Running PLA test …")
def get_pla(_rtpl, _hpl):
    return run_pla_test(rtpl=_rtpl, hpl=_hpl)


# =========================================================================
#  Helpers
# =========================================================================
def fmt_money(x, scale="M", sign=False):
    if pd.isna(x):
        return "—"
    fmt = f"${{:{'+' if sign else ''},.1f}}M"
    return fmt.format(x / 1e6)


ZONE_DOT = {"green": "🟢", "amber": "🟠", "red": "🔴"}


# =========================================================================
#  Header + KPIs
# =========================================================================
def render_header(pf, rfs, cap):
    st.title("Market Risk Engine")
    st.caption(
        "End-to-end sell-side market-risk system for a US trading book — "
        "VaR / ES across four methods, FRTB Expected Shortfall with liquidity horizons, "
        "Basel III vs FRTB capital, P&L attribution (PLA), backtesting, stress and "
        "component-VaR attribution. All numbers run from the bundled `data_cache/`; "
        "nothing leaves your machine."
    )

    c1, c2, c3, c4 = st.columns(4)
    ratio = cap.frtb_ima_capital / cap.basel_capital if cap.basel_capital else float("nan")
    binding = max(cap.frtb_ima_capital, 0.725 * cap.frtb_sa_capital)

    with c1:
        st.metric("Basel 2.5", fmt_money(cap.basel_capital),
                  help="3 × max(VaR, m·VaR̄₆₀) + 3 × max(SVaR, m·SVaR̄₆₀)")
    with c2:
        st.metric("FRTB IMA", fmt_money(cap.frtb_ima_capital),
                  delta=f"{ratio:.0%} of Basel", delta_color="off",
                  help="mc × max(ES, ES̄₆₀) — risk-sensitive stressed-ES capital")
    with c3:
        st.metric("FRTB SA (SbM)", fmt_money(cap.frtb_sa_capital),
                  help="Sensitivities-based delta + curvature + vega charges")
    with c4:
        st.metric("FRTB reported", fmt_money(binding),
                  help="max(IMA, 72.5% × SA) — Basel output floor")


# =========================================================================
#  Tabs
# =========================================================================
def tab_live_risk(pf, sens):
    st.subheader("VaR and ES — four methods, 1-day horizon")
    st.caption(
        "Historical, parametric (normal), Cornish-Fisher and filtered-historical-simulation. "
        "FHS is the FRTB-aligned method: GARCH-volatility-rescaled returns let the tail breathe "
        "with current market conditions."
    )
    confidences = (0.95, 0.975, 0.99)
    results = get_var_results(pf, sens, confidences, (1,))

    # Long-form table
    rows = []
    for r in results:
        for c in confidences:
            rows.append({"Method": r.method, "Confidence": f"{c:.1%}",
                         "VaR (1d)": r.var[(c, 1)], "ES (1d)": r.es[(c, 1)]})
    df = pd.DataFrame(rows)

    # Bar chart at user-selected confidence
    pick = st.select_slider(
        "Chart this confidence:",
        options=[f"{c:.1%}" for c in confidences],
        value="99.0%",
    )
    conf_val = float(pick.strip("%")) / 100
    bar_df = pd.DataFrame({
        "Method": [r.method for r in results],
        "VaR": [r.var[(conf_val, 1)] for r in results],
        "ES": [r.es[(conf_val, 1)] for r in results],
    })
    fig = go.Figure()
    fig.add_bar(name=f"VaR {pick}", x=bar_df["Method"], y=bar_df["VaR"], marker_color=NAVY)
    fig.add_bar(name=f"ES {pick}", x=bar_df["Method"], y=bar_df["ES"], marker_color=STEEL)
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=380,
        yaxis=dict(**AXIS_STYLE, title="Loss ($)", tickformat="$,.0s"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df.style.format({"VaR (1d)": "${:,.0f}", "ES (1d)": "${:,.0f}"}),
        use_container_width=True, hide_index=True,
    )


def tab_pla(pf, sens):
    st.subheader("P&L attribution — RTPL (Taylor) vs HPL (full revaluation)")
    st.caption(
        "Today-frozen design: positions and curves are anchored at today's date and shocked "
        "from there. Linear positions therefore match by construction, so the test signal "
        "is genuine convexity (gamma) and vega divergence — not portfolio drift."
    )
    rtpl, hpl = get_pnl_streams(pf, sens)
    pla = get_pla(rtpl, hpl)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Observations", f"{pla.n_observations:,}")
    with c2:
        st.metric("Spearman ρ", f"{pla.spearman_rho:+.4f}",
                  delta=pla.spearman_zone, delta_color="off")
    with c3:
        st.metric("KS statistic", f"{pla.ks_statistic:.4f}",
                  delta=pla.ks_zone, delta_color="off")
    with c4:
        zone = pla.zone.upper()
        st.metric("Overall zone", f"{ZONE_DOT.get(pla.zone, '')} {zone}",
                  delta="IMA eligible" if pla.ima_eligible else "SA migration",
                  delta_color="off")

    # Scatter
    lim = max(abs(rtpl).max(), abs(hpl).max()) / 1e6
    fig = go.Figure()
    fig.add_scatter(
        x=rtpl.values / 1e6, y=hpl.values / 1e6, mode="markers",
        marker=dict(size=4, color=STEEL, opacity=0.30), name="Daily",
    )
    fig.add_scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color=RED, width=1, dash="dash"), name="45°",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=460, showlegend=False,
        xaxis=dict(**AXIS_STYLE, title="RTPL (Taylor)  $M"),
        yaxis=dict(**AXIS_STYLE, title="HPL (full reval)  $M",
                   scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("FRTB MAR32 thresholds (reference)"):
        st.markdown(
            "| Metric    | Green      | Amber          | Red       |\n"
            "|-----------|------------|----------------|-----------|\n"
            "| Spearman  | ≥ 0.80     | ≥ 0.70         | < 0.70    |\n"
            "| KS stat   | ≤ 0.09     | ≤ 0.12         | > 0.12    |\n"
            "Overall zone is the worst of the two; amber retains IMA but adds a capital "
            "surcharge; red forces permanent SA migration for that desk."
        )


def tab_backtesting(pf, sens):
    st.subheader("Rolling 250-day VaR vs realised P&L — exception backtesting")
    st.caption(
        "Forecasts are quantiles of the trailing 250-day RTPL window vs next-day realised. "
        "Exception clusters surface in Christoffersen's conditional coverage test; recent "
        "250d sets the FRTB capital multiplier `mc` via the BCBS traffic light."
    )
    rtpl, hpl = get_pnl_streams(pf, sens)

    c1, c2 = st.columns(2)
    with c1:
        conf = st.select_slider(
            "Confidence:",
            options=[0.95, 0.975, 0.99],
            value=0.99, format_func=lambda x: f"{x:.1%}",
        )
    with c2:
        window_n = st.slider("Display window (days):",
                             min_value=250, max_value=2000, value=500, step=50)

    es = generate_exceptions(realized=rtpl, confidence=conf, window=250, var_source=rtpl)
    real = es.realized.iloc[-window_n:]
    vfc  = es.var_forecast.iloc[-window_n:]
    exc  = es.exceptions.iloc[-window_n:].astype(bool)
    n_breach = int(exc.sum())
    exp_breach = window_n * (1 - conf)

    # Trailing 250d for the FRTB capital multiplier reading
    trailing = es.exceptions.iloc[-250:].astype(bool).sum() if len(es.exceptions) >= 250 else int(exc.sum())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Breaches in window", f"{n_breach}",
                  delta=f"vs {exp_breach:.0f} expected", delta_color="off")
    with k2:
        st.metric("Realised rate", f"{n_breach / len(real):.2%}",
                  delta=f"target {1 - conf:.1%}", delta_color="off")
    with k3:
        # Traffic light zone for FRTB at 99% trailing 250d:
        # 0-4 green, 5-9 amber, 10+ red (BCBS).
        if conf >= 0.985:
            if trailing <= 4: zone, dot = "GREEN", "🟢"
            elif trailing <= 9: zone, dot = "AMBER", "🟠"
            else: zone, dot = "RED", "🔴"
            st.metric("Trailing 250d zone (99%)", f"{dot} {zone}",
                      delta=f"{trailing} exceptions", delta_color="off")
        else:
            st.metric("Trailing 250d exceptions", f"{trailing}",
                      delta="traffic light is 99% only", delta_color="off")

    # Chart
    fig = go.Figure()
    fig.add_scatter(x=real.index, y=real.values / 1e6, name="Realised P&L",
                    mode="lines", line=dict(color=GREY, width=0.7))
    fig.add_scatter(x=vfc.index, y=-vfc.values / 1e6, name=f"-VaR {conf:.1%}",
                    mode="lines", line=dict(color=NAVY, width=1.4))
    if exc.any():
        fig.add_scatter(
            x=real.index[exc.values], y=real.values[exc.values] / 1e6,
            name=f"Breaches ({n_breach})", mode="markers",
            marker=dict(size=8, color=RED, symbol="x"),
        )
    fig.update_layout(
        **PLOTLY_LAYOUT, height=420,
        yaxis=dict(**AXIS_STYLE, title="P&L  $M"),
        xaxis=dict(**AXIS_STYLE, title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Honest note:** on the *full* history at 99%, the unconditional coverage test "
        "rejects the model (more breaches than the nominal 1%). The 97.5% level is borderline. "
        "Recent 250d is fine, which is why capital stays in IMA — FRTB intentionally uses "
        "the rolling window for the multiplier, not the full sample.", icon="ℹ️",
    )


def tab_capital(cap):
    st.subheader("Regulatory capital — three regimes, one binding number")

    # The summary() output as-is — it's an institutional artifact people recognise.
    st.code(cap.summary(), language="text")

    st.divider()

    comp = cap.comparison()
    st.dataframe(
        comp.style.format({
            "multiplier": lambda v: "—" if pd.isna(v) else f"{v:.2f}",
            "capital":    "${:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    fig = go.Figure(go.Bar(
        x=comp["regime"], y=comp["capital"],
        marker_color=[GREY, NAVY, STEEL, AMBER],
        text=[fmt_money(v) for v in comp["capital"]], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=380,
        yaxis=dict(**AXIS_STYLE, title="Capital ($)", tickformat="$,.0s"),
        xaxis=dict(**AXIS_STYLE, title=None),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def tab_stress(pf, sens, rfs):
    st.subheader("Named scenarios — full revaluation, worst first")
    st.caption(
        "Historical replays (GFC, COVID, 2022 rates) and hypothetical regime shifts. "
        "`greek_est` is the Taylor approximation; `residual = full_reval − greek_est` "
        "quantifies how much non-linearity (gamma, vega) the acute moves carry."
    )
    table = get_stress_table(pf, sens, rfs).copy()
    table = table.sort_values("full_reval")  # worst (most negative) first

    fig = go.Figure(go.Bar(
        x=table["full_reval"] / 1e6, y=table["scenario"], orientation="h",
        marker_color=[RED if v < 0 else GREEN for v in table["full_reval"]],
        text=[fmt_money(v, sign=True) for v in table["full_reval"]],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=420,
        xaxis=dict(**AXIS_STYLE, title="Full-reval P&L  ($M)"),
        yaxis=dict(**AXIS_STYLE, title=None, autorange="reversed"),
        showlegend=False,
    )
    fig.update_layout(margin=dict(t=20, b=20, l=10, r=80))
    st.plotly_chart(fig, use_container_width=True)

    cols = ["scenario", "source", "full_reval", "greek_est", "residual",
            "delta", "gamma", "vega", "rho"]
    money_cols = [c for c in cols if c not in ("scenario", "source")]
    st.dataframe(
        table[cols].style.format({c: "${:,.0f}" for c in money_cols}),
        use_container_width=True, hide_index=True,
    )


def tab_attribution(pf, sens):
    st.subheader("VaR attribution — where the 99% loss comes from")
    st.caption(
        "Parametric and historical component VaR by position. Component VaRs sum exactly "
        "to portfolio VaR by construction (Euler decomposition). HHI = concentration of risk "
        "by absolute contribution; effective N = 1/HHI."
    )
    att = get_attribution(pf, sens, confidence=0.99)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Positions", att.concentration.n_positions)
    with k2: st.metric("HHI", f"{att.concentration.hhi:.3f}",
                       delta=f"effective {att.concentration.effective_n:.1f}",
                       delta_color="off")
    with k3: st.metric("Diversification", f"{att.concentration.diversification_ratio:.2f}×",
                       delta="ΣVaR / portfolio VaR", delta_color="off")
    with k4: st.metric("Top-3 share", f"{att.concentration.top3_share:.1%}",
                       delta=f"largest: {att.concentration.largest_contributor}",
                       delta_color="off")

    cv = att.parametric.component_var.sort_values()
    fig = go.Figure(go.Bar(
        x=cv.values / 1e6, y=cv.index, orientation="h",
        marker_color=[RED if v < 0 else NAVY for v in cv.values],
        text=[f"${v/1e6:+.2f}M" for v in cv.values], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=440,
        xaxis=dict(**AXIS_STYLE, title="Component VaR ($M)"),
        yaxis=dict(**AXIS_STYLE, title=None),
        showlegend=False,
    )
    fig.update_layout(margin=dict(t=20, b=20, l=10, r=80))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        att.comparison().style.format({
            "param_component": "${:,.0f}", "param_pct": "{:.1f}%",
            "hist_component":  "${:,.0f}", "hist_pct":  "{:.1f}%",
            "marginal_/1m":    "${:,.0f}",
        }),
        use_container_width=True,
    )


# =========================================================================
#  Main
# =========================================================================
def main():
    rfs, pf, sens = load_engine()
    cap = get_capital(pf, sens)

    # Sidebar
    with st.sidebar:
        st.markdown("### Book & data")
        st.metric("Positions", len(pf))
        st.metric("Risk factors", len(rfs.names))
        st.metric("Aligned days", len(rfs.changes_matrix()))
        st.markdown(
            "**Window:** 2010-01-01 → 2024-12-31  \n"
            "**Source:** bundled `data_cache/` — no network calls, no FRED key needed."
        )
        with st.expander("About this dashboard", expanded=False):
            st.markdown(
                "Mirrors the six panels of `capital/dashboard.py` (the static PNG in the README) "
                "but interactive. Every number is computed by the same engine modules under "
                "`var/`, `frtb/`, `backtest/`, `stress/`, `attribution/` and `capital/`. "
                "Heavy computations (capital, attribution, stress) are cached for the session."
            )
        with st.expander("Methodology pointers", expanded=False):
            st.markdown(
                "- **VaR/ES** — four methods (historical, parametric, Cornish-Fisher, FHS-GARCH).\n"
                "- **PLA test** — FRTB MAR32 Annex B (Spearman ≥ 0.80, KS ≤ 0.09 for green).\n"
                "- **Capital** — Basel 2.5 (VaR+SVaR, m = 3.85), FRTB IMA (stressed ES, mc),  \n"
                "  FRTB SA (SbM: delta + curvature + vega). Reported = max(IMA, 72.5%×SA).\n"
                "- **Stress** — historical (GFC, COVID, 2022 rates) and hypothetical regime shifts; "
                "  full revaluation with Greek-Taylor residual.\n"
                "- **Attribution** — Euler component VaR, parametric and historical, sums match VaR exactly."
            )

    render_header(pf, rfs, cap)
    st.divider()

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Live risk", "P&L attribution (PLA)", "Backtesting",
        "Capital detail", "Stress", "Attribution",
    ])
    with t1: tab_live_risk(pf, sens)
    with t2: tab_pla(pf, sens)
    with t3: tab_backtesting(pf, sens)
    with t4: tab_capital(cap)
    with t5: tab_stress(pf, sens, rfs)
    with t6: tab_attribution(pf, sens)


if __name__ == "__main__":
    main()
