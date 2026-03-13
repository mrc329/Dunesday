"""
app.py — Dunesday v5 Streamlit Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

from model.config import FILM_PARAMS, IMAX_CONFIG
from model.core import run_all_scenarios, imax_gap_summary, SCENARIOS
from model.signals import fetch_and_calibrate

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DUNESDAY · Box Office Model",
    page_icon="🏜️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PALETTES ───────────────────────────────────────────────────────────────────
#  dark  — near-black field, warm off-white type (current)
#  light — Tufte paper/linen, near-black ink
PALETTES: dict[str, dict] = {
    "dark": {
        "bg":           "#04060c",
        "sidebar_bg":   "#040710",
        "text":         "#cdc8be",
        "dim":          "#3c4e62",
        "card_rule":    "#141e2c",
        "axis":         "rgba(255,255,255,0.09)",
        "vline_ref":    "rgba(255,255,255,0.15)",
        "chart_font":   "#8a9daf",
        "mid_ref":      "#667788",
        "footer":       "#1e3048",
        "dune":         "#d4a030",   # Arrakis amber
        "av":           "#cc2030",   # Avengers crimson
        "cyan":         "#00b4c2",
        "conf_medium":  "#d48020",
        "fill_dune":    "rgba(212,160,48,0.12)",
        "fill_av":      "rgba(204,32,48,0.10)",
        "info_bg":      "rgba(0,180,194,0.07)",
    },
    "light": {
        "bg":           "#f5f1e8",   # aged paper
        "sidebar_bg":   "#ece8de",
        "text":         "#1a1816",   # near-black ink
        "dim":          "#7a6e62",   # warm muted brown
        "card_rule":    "#cdc8bc",
        "axis":         "rgba(0,0,0,0.12)",
        "vline_ref":    "rgba(0,0,0,0.15)",
        "chart_font":   "#4a4038",
        "mid_ref":      "#8a7a6a",
        "footer":       "#8a7a6a",
        "dune":         "#8c6400",   # amber darkened for light bg
        "av":           "#921520",   # crimson darkened for light bg
        "cyan":         "#006070",   # teal darkened for light bg
        "conf_medium":  "#9a6010",
        "fill_dune":    "rgba(140,100,0,0.10)",
        "fill_av":      "rgba(146,21,32,0.09)",
        "info_bg":      "rgba(0,96,112,0.07)",
    },
}

# ── THEME STATE — must init before any widget ──────────────────────────────────
if "theme" not in st.session_state:
    try:
        _base = st.get_option("theme.base")
        st.session_state.theme = "light" if _base == "light" else "dark"
    except Exception:
        st.session_state.theme = "dark"

P = PALETTES[st.session_state.theme]


# ── CSS ────────────────────────────────────────────────────────────────────────
def build_css(P: dict) -> str:
    return f"""
<style>
  /* ── global background ── */
  [data-testid="stApp"],
  .main,
  section.main > div.block-container {{
    background-color: {P['bg']} !important;
  }}
  .block-container {{
    padding-top: 0.6rem !important;
    padding-bottom: 0.5rem !important;
  }}

  /* ── body text ── */
  .stMarkdown p, .stMarkdown li, .stMarkdown span,
  [data-testid="stText"], [data-testid="stCaptionContainer"] p,
  label, .stCaption {{
    color: {P['text']} !important;
  }}

  /* ── metric cards — Tufte: top rule, no box ── */
  .stMetric {{
    background: transparent !important;
    border: none !important;
    border-top: 1px solid {P['card_rule']} !important;
    padding: 10px 0 6px !important;
    border-radius: 0 !important;
  }}
  [data-testid="stMetricValue"] {{
    font-size: 1.65rem !important;
    font-variant-numeric: tabular-nums !important;
    font-weight: 700 !important;
    color: {P['text']} !important;
    line-height: 1.15 !important;
    letter-spacing: -0.5px !important;
  }}
  [data-testid="stMetricLabel"] {{
    font-size: 0.58rem !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: {P['dim']} !important;
    font-weight: 400 !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none !important; }}
  [data-testid="stMetricDelta"] > div {{
    font-size: 0.68rem !important;
    font-variant-numeric: tabular-nums !important;
  }}

  /* ── verdict box ── */
  .verdict-box {{
    border-left: 2px solid {P['dune']};
    padding: 14px 20px;
    margin: 8px 0;
    line-height: 1.8;
    color: {P['text']};
  }}

  /* ── sidebar ── */
  [data-testid="stSidebar"] > div,
  div[data-testid="stSidebarContent"] {{
    background: {P['sidebar_bg']} !important;
  }}
  [data-testid="stSidebar"] .stMarkdown p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stCaption p {{
    color: {P['text']} !important;
  }}

  /* ── tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    padding: 6px 14px !important;
    background: transparent !important;
    color: {P['dim']} !important;
  }}
  .stTabs [aria-selected="true"] {{
    color: {P['text']} !important;
    border-bottom-color: {P['dune']} !important;
  }}

  /* ── dividers ── */
  hr {{
    border-color: {P['card_rule']} !important;
    opacity: 1 !important;
  }}

  /* ── info box ── */
  [data-testid="stInfo"] {{
    background: {P['info_bg']} !important;
    border-color: {P['cyan']} !important;
  }}
  [data-testid="stInfo"] p {{
    color: {P['text']} !important;
  }}

  /* ── expander ── */
  [data-testid="stExpander"] {{
    border-color: {P['card_rule']} !important;
    background: transparent !important;
  }}
  [data-testid="stExpander"] summary {{
    color: {P['text']} !important;
  }}

  /* ── dataframe ── */
  [data-testid="stDataFrame"] {{ font-size: 0.78rem !important; }}

  /* ── toggle / slider labels ── */
  [data-testid="stToggleLabel"] p,
  [data-testid="stSliderLabel"] p,
  [data-testid="stSelectSliderLabel"] p {{
    color: {P['text']} !important;
  }}
</style>
"""

st.markdown(build_css(P), unsafe_allow_html=True)


# ── CHART LAYOUT HELPER ────────────────────────────────────────────────────────
def _layout(P: dict, **kw) -> dict:
    """Tufte-style Plotly base layout — transparent bg, no gridlines."""
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=P["bg"],
        font=dict(color=P["chart_font"], size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10),
            orientation="h",
            y=1.04,
            x=0,
        ),
        margin=dict(t=20, b=10, l=4, r=8),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor=P["axis"],
            linewidth=0.5,
            ticks="",
            tickfont=dict(size=10),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=10),
        ),
    )
    base.update(kw)
    return base


# ── FETCH LIVE SIGNALS ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_signals(base_dune: int, base_av: int):
    return fetch_and_calibrate(base_dune_score=base_dune, base_av_score=base_av)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Theme toggle — top of sidebar, before everything else
    theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(theme_label, use_container_width=False):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Model Controls")

    st.markdown("**Base Audience Scores**")
    st.caption("Live signals auto-calibrate from these baselines")
    base_dune_aud = st.slider("Dune base score", 60, 100, 87)
    base_av_aud   = st.slider("Avengers base score", 60, 100, 88)

    st.divider()

    with st.spinner("Fetching live signals..."):
        signals = load_signals(base_dune_aud, base_av_aud)

    cal = signals["calibration"]
    confidence_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    conf_icon = confidence_icons.get(cal["signal_confidence"], "⚪")

    st.markdown(f"**{conf_icon} Live Calibration**")
    st.caption(f"Updated: {signals['last_updated']}")

    c1, c2 = st.columns(2)
    with c1:
        adj = cal["dune_adj"]
        st.metric("Dune score", f"{cal['dune_calibrated']:.0f}",
                  delta=f"{adj:+.1f}" if adj != 0 else "no adj")
    with c2:
        adj = cal["avengers_adj"]
        st.metric("Avengers score", f"{cal['avengers_calibrated']:.0f}",
                  delta=f"{adj:+.1f}" if adj != 0 else "no adj")

    st.caption(f"Sources: {' · '.join(cal['sources'])}")
    if cal.get("notes"):
        st.caption(f"📌 {cal['notes']}")

    st.divider()
    override = st.toggle("Manual score override", value=False)
    if override:
        dune_aud = st.slider("Dune (manual)", 60, 100, int(cal["dune_calibrated"]))
        av_aud   = st.slider("Avengers (manual)", 60, 100, int(cal["avengers_calibrated"]))
        st.caption("⚠️ Overriding live calibration")
    else:
        dune_aud = int(cal["dune_calibrated"])
        av_aud   = int(cal["avengers_calibrated"])

    st.divider()
    st.markdown("**International Multipliers**")
    dune_intl = st.slider("Dune intl mult", 0.8, 2.5, 1.48, 0.05)
    av_intl   = st.slider("Avengers intl mult", 1.0, 3.5, 2.18, 0.05)

    st.markdown("**Simulation**")
    n_trials = st.select_slider("MC trials", [500, 1000, 2000, 5000], value=1000)

    st.divider()
    st.markdown("**IMAX Config (Locked)**")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{P['dim']}; line-height:1.9;'>
    Dune exclusive: <b style='color:{P['dune']}'>Days 1–21</b><br>
    Dune screens: <b style='color:{P['dune']}'>400</b><br>
    Avengers day 1: <b style='color:{P['av']}'>0 screens</b><br>
    Avengers first IMAX: <b style='color:{P['av']}'>Jan 8</b>
    </div>
    """, unsafe_allow_html=True)


# ── RUN MODEL ─────────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo..."):
    results = run_all_scenarios(
        n=n_trials,
        dune_aud=dune_aud, av_aud=av_aud,
        dune_intl=dune_intl, av_intl=av_intl,
    )
    imax = imax_gap_summary()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:4px 0 12px; border-bottom:1px solid {P['card_rule']}; margin-bottom:14px;'>
  <span style='font-size:1.4rem; font-weight:700; letter-spacing:5px; color:{P['text']};'>
    <span style='color:{P['dune']}'>DUNE</span>SDAY
  </span>
  <span style='color:{P['dim']}; font-size:0.62rem; letter-spacing:3px; margin-left:20px;
               vertical-align:middle;'>
    BOX OFFICE MODEL &nbsp;·&nbsp; DEC 18 2026 &nbsp;·&nbsp; LIVE SIGNALS
  </span>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ───────────────────────────────────────────────────────────────────
st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2.5px; color:{P['dim']}; margin:4px 0 8px;'>VERDICT — BOTH HOLDING DEC 18</p>",
            unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
sc_a = results["A_Both_Hold"]

with k1:
    st.metric("Dune P50 Profit", f"${sc_a['DUNE']['p50']:.0f}M", delta="100% break-even")
with k2:
    st.metric("Avengers P50 Profit", f"${sc_a['AVENGERS']['p50']:.0f}M",
              delta=f"{sc_a['AVENGERS']['breakeven_pct']:.0f}% BE")
with k3:
    st.metric("IMAX Gap", f"${imax['gap']:.1f}M", delta="Dune advantage")
with k4:
    st.metric("Dune Xmas IMAX", f"${imax['xmas_day_dune']:.2f}M")
with k5:
    st.metric("Avengers Xmas IMAX", "$0.00M", delta="Zero screens", delta_color="inverse")
with k6:
    st.metric("Avengers Locked Out", "21 days", delta="Dec 18 – Jan 7", delta_color="off")

st.divider()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "SCENARIOS", "IMAX TIMELINE", "LIVE SIGNALS", "DISTRIBUTIONS", "DISNEY DECISION",
])


# ── TAB 1: SCENARIOS ──────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>NET PROFIT BY SCENARIO — P10 / P50 / P90</p>",
                unsafe_allow_html=True)

    sk_list   = list(SCENARIOS.keys())
    sc_labels = [SCENARIOS[sk]["label"] for sk in sk_list]
    dune_p50s = [results[sk]["DUNE"]["p50"]     for sk in sk_list]
    dune_p10s = [results[sk]["DUNE"]["p10"]     for sk in sk_list]
    dune_p90s = [results[sk]["DUNE"]["p90"]     for sk in sk_list]
    av_p50s   = [results[sk]["AVENGERS"]["p50"] for sk in sk_list]
    av_p10s   = [results[sk]["AVENGERS"]["p10"] for sk in sk_list]
    av_p90s   = [results[sk]["AVENGERS"]["p90"] for sk in sk_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Dune", x=sc_labels, y=dune_p50s,
        marker_color=P["dune"], offsetgroup=0,
        text=[f"${v:.0f}M" for v in dune_p50s],
        textposition="outside",
        textfont=dict(size=10, color=P["dune"]),
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50 for p90, p50 in zip(dune_p90s, dune_p50s)],
            arrayminus=[p50 - p10 for p50, p10 in zip(dune_p50s, dune_p10s)],
            color=P["dune"], thickness=1.5, width=5,
        ),
    ))
    fig.add_trace(go.Bar(
        name="Avengers", x=sc_labels, y=av_p50s,
        marker_color=P["av"], offsetgroup=1,
        text=[f"${v:.0f}M" for v in av_p50s],
        textposition="outside",
        textfont=dict(size=10, color=P["av"]),
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50 for p90, p50 in zip(av_p90s, av_p50s)],
            arrayminus=[p50 - p10 for p50, p10 in zip(av_p50s, av_p10s)],
            color=P["av"], thickness=1.5, width=5,
        ),
    ))
    fig.add_hline(y=0, line_width=0.5, line_color=P["axis"])
    fig.update_layout(**_layout(P, barmode="group", height=390, yaxis_title="Net Profit ($M)"))
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for sk in sk_list:
        rd, ra = results[sk]["DUNE"], results[sk]["AVENGERS"]
        rows.append({
            "Scenario":     SCENARIOS[sk]["label"],
            "Dune P50":     f"${rd['p50']:.0f}M",
            "Avengers P50": f"${ra['p50']:.0f}M",
            "Dune BE%":     f"{rd['breakeven_pct']:.0f}%",
            "Avengers BE%": f"{ra['breakeven_pct']:.0f}%",
            "Description":  SCENARIOS[sk]["description"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 2: IMAX TIMELINE ──────────────────────────────────────────────────────
with tab2:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>IMAX SCREEN ALLOCATION — DAYS 1–45</p>",
                unsafe_allow_html=True)

    days        = np.arange(45)
    open_date   = datetime.date(2026, 12, 18)
    date_labels = [(open_date + datetime.timedelta(days=int(d))).strftime("%b %d") for d in days]
    excl        = IMAX_CONFIG["dune_exclusive_days"]
    dune_screens = [400 if d < excl else 200 for d in days]
    av_screens   = [0   if d < excl else 200 for d in days]

    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Screen Allocation", "Daily IMAX Revenue ($M)"),
        vertical_spacing=0.14,
    )
    fig2.add_trace(go.Bar(x=date_labels, y=dune_screens,
                          name="Dune", marker_color=P["dune"]), row=1, col=1)
    fig2.add_trace(go.Bar(x=date_labels, y=av_screens,
                          name="Avengers", marker_color=P["av"]), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["dune_daily"],
        name="Dune IMAX rev",
        line=dict(color=P["dune"], width=2),
        fill="tozeroy", fillcolor=P["fill_dune"],
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["avengers_daily"],
        name="Avengers IMAX rev",
        line=dict(color=P["av"], width=2),
        fill="tozeroy", fillcolor=P["fill_av"],
    ), row=2, col=1)

    # add_vline with string (categorical) x-axes triggers a Plotly bug in newer
    # versions: annotation positioning calls _mean([str, str]) which raises
    # TypeError.  Use add_shape + add_annotation instead.
    for day, color, label in [
        (7,  P["vline_ref"], "Wk 1"),
        (14, P["vline_ref"], "Wk 2"),
        (21, P["cyan"],      "Day 21"),
    ]:
        x_val = date_labels[day]
        for xref, yref in [("x", "y domain"), ("x2", "y2 domain")]:
            fig2.add_shape(
                type="line",
                x0=x_val, x1=x_val, y0=0, y1=1,
                xref=xref, yref=yref,
                line=dict(dash="dot", color=color, width=0.8),
            )
        if label:
            fig2.add_annotation(
                x=x_val, y=1,
                xref="x", yref="y domain",
                text=label,
                font=dict(color=color, size=9),
                showarrow=False,
                yanchor="bottom",
                xanchor="center",
            )

    fig2.update_layout(**_layout(P, height=480, barmode="stack",
                                 margin=dict(t=32, b=10, l=4, r=8)))
    fig2.update_xaxes(tickangle=45, nticks=15, showgrid=False,
                      showline=True, linecolor=P["axis"], linewidth=0.5)
    fig2.update_yaxes(showgrid=False, showline=False, zeroline=False)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dune excl window",     f"${imax['dune_excl_rev']:.1f}M", "Days 1–21")
    c2.metric("Avengers excl window", "$0.0M", "Zero screens", delta_color="inverse")
    c3.metric("Dune 45-day IMAX",     f"${imax['dune_total']:.1f}M")
    c4.metric("Avengers 45-day IMAX", f"${imax['avengers_total']:.1f}M",
              delta=f"-${imax['gap']:.1f}M vs Dune", delta_color="inverse")


# ── TAB 3: LIVE SIGNALS ───────────────────────────────────────────────────────
with tab3:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>LIVE SIGNAL DASHBOARD</p>",
                unsafe_allow_html=True)

    conf       = cal["signal_confidence"]
    conf_label = {
        "high":   "HIGH — multiple live sources",
        "medium": "MEDIUM — partial live data",
        "low":    "LOW — fallback values only",
    }.get(conf, conf)
    conf_color = {
        "high":   P["dune"],
        "medium": P["conf_medium"],
        "low":    P["av"],
    }.get(conf, P["mid_ref"])

    st.markdown(f"""
    <div style='border-left:2px solid {conf_color}; padding:8px 14px; margin-bottom:14px;'>
      <span style='color:{conf_color}; font-size:0.6rem; letter-spacing:2px'>
        SIGNAL CONFIDENCE: {conf_label}
      </span><br>
      <span style='color:{P['dim']}; font-size:0.82rem'>{cal.get("notes", "")}</span>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<span style='color:{P['av']}; font-size:0.82rem; font-weight:600; letter-spacing:2px;'>AVENGERS: DOOMSDAY</span>",
                    unsafe_allow_html=True)

        av_sig  = signals["avengers"]
        teasers = av_sig.get("teaser_views_x_M", [])

        if teasers:
            labels = [f"T{i+1}" for i in range(len(teasers))]
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=labels, y=teasers,
                marker_color=P["av"],
                text=[f"{v:.0f}M" for v in teasers],
                textposition="outside",
                textfont=dict(size=10, color=P["av"]),
                showlegend=False,
            ))
            fig_d.add_trace(go.Scatter(
                x=labels, y=teasers,
                line=dict(color=P["av"], dash="dot", width=1),
                mode="lines", showlegend=False,
            ))
            t1 = teasers[0]
            fig_d.add_hline(
                y=t1 * 0.77, line_dash="dash", line_width=0.8,
                line_color=P["dune"], opacity=0.8,
                annotation_text="D&W held 77%",
                annotation_font_color=P["dune"], annotation_font_size=9,
            )
            fig_d.add_hline(
                y=t1 * 0.47, line_dash="dash", line_width=0.8,
                line_color=P["mid_ref"], opacity=0.8,
                annotation_text="L&T soft 47%",
                annotation_font_color=P["mid_ref"], annotation_font_size=9,
            )
            fig_d.update_layout(**_layout(
                P,
                title=dict(text="Teaser Decay — X/Twitter Views", font=dict(size=11), x=0),
                height=260, yaxis_title="Views (M)",
            ))
            st.plotly_chart(fig_d, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Trends interest",
                  f"{av_sig.get('trends_interest', '—')}/100",
                  delta="Google Trends US")
        m2.metric("YT trailer views",
                  f"{av_sig['yt_trailer_views']:,}" if av_sig.get("yt_trailer_views") else "—",
                  delta="Full trailer not released" if not av_sig.get("full_trailer_out") else "Live")

        if len(teasers) >= 2:
            decay_signal = cal.get("teaser_decay_signal", "neutral")
            decay_color  = {
                "strong":  P["dune"],
                "neutral": P["mid_ref"],
                "soft":    P["av"],
            }.get(decay_signal, P["mid_ref"])
            st.markdown(f"""
            <div style='border-left:2px solid {decay_color}; padding:7px 12px; margin-top:8px;'>
              <span style='color:{decay_color}; font-size:0.6rem; letter-spacing:2px'>
                DECAY: {decay_signal.upper()}
              </span><br>
              <span style='color:{P['dim']}; font-size:0.76rem'>
                T1→T2: {teasers[0]:.0f}M → {teasers[1]:.0f}M
                ({(teasers[1]/teasers[0]*100):.0f}% of T1)
                {'— matches Love&Thunder pattern' if (teasers[1]/teasers[0]) < 0.55 else '— tracking neutral'}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"<span style='color:{P['dune']}; font-size:0.82rem; font-weight:600; letter-spacing:2px;'>DUNE: PART THREE</span>",
                    unsafe_allow_html=True)

        dune_sig = signals["dune"]
        st.info("No trailer released. WB following Part Two marketing cadence — strategic delay.")

        m1, m2 = st.columns(2)
        m1.metric("Trends interest",
                  f"{dune_sig.get('trends_interest', '—')}/100",
                  delta=f"vs Avengers {av_sig.get('trends_interest', '?')}/100")
        m2.metric("Alamo poll", "#1 Most Anticipated", delta="14,000 respondents")

        av_t    = av_sig.get("trends_interest", 72)
        dune_t  = dune_sig.get("trends_interest", 13)
        total_t = av_t + dune_t or 1

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Bar(
            x=["Search Interest Share"],
            y=[av_t / total_t * 100],
            name="Avengers", marker_color=P["av"],
            text=[f"Avengers {av_t / total_t * 100:.0f}%"],
            textposition="inside",
            textfont=dict(size=10, color="white"),
        ))
        fig_ratio.add_trace(go.Bar(
            x=["Search Interest Share"],
            y=[dune_t / total_t * 100],
            name="Dune", marker_color=P["dune"],
            text=[f"Dune {dune_t / total_t * 100:.0f}%"],
            textposition="inside",
            textfont=dict(size=10, color=P["bg"]),
        ))
        fig_ratio.add_hline(
            y=18, line_dash="dot", line_width=0.8,
            line_color=P["dune"], opacity=0.7,
            annotation_text="Expected Dune baseline (no trailer)",
            annotation_font_color=P["dune"], annotation_font_size=9,
        )
        fig_ratio.update_layout(**_layout(P, barmode="stack", height=180, yaxis_title="%"))
        st.plotly_chart(fig_ratio, use_container_width=True)

        st.caption("Dune's 13/100 vs Avengers 72/100 is marketing stage, not demand. "
                   "Dune has released zero promotional materials.")

    st.divider()
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>HOW SIGNALS FEED THE MODEL</p>",
                unsafe_allow_html=True)

    rows_sig = [
        ["Google Trends", "Search interest ratio",
         f"Av {av_t} / Dune {dune_t}",
         f"Av {cal['avengers_adj']:+.1f}pt / Dune {cal['dune_adj']:+.1f}pt",
         "✓ Live" if signals.get("source") == "live" else "⚠ Fallback"],
        ["Teaser decay", "T1→T2 view retention",
         f"{teasers[0]:.0f}M → {teasers[1]:.0f}M ({teasers[1]/teasers[0]*100:.0f}%)" if len(teasers) >= 2 else "—",
         f"Av {cal['avengers_adj']:+.1f}pt (decay component)", "✓ Manual"],
        ["YouTube API", "Official trailer views", "—",
         "Ready — add YOUTUBE_API_KEY to secrets", "⚠ Key needed"],
        ["Fandango presales", "Purchase intent", "Not open yet",
         "Opens Sept 2026", "⏳ Pending"],
    ]
    st.dataframe(
        pd.DataFrame(rows_sig, columns=["Source", "Signal", "Current Value", "Model Impact", "Status"]),
        use_container_width=True, hide_index=True,
    )

    with st.expander("🔑 Set up YouTube API key (5 min)"):
        st.markdown("""
        1. Go to [console.cloud.google.com](https://console.cloud.google.com)
        2. Create project → Enable **YouTube Data API v3**
        3. Credentials → Create API Key (free, 10k units/day)
        4. In Streamlit Cloud: App Settings → Secrets → add:
        ```toml
        YOUTUBE_API_KEY = "your-key-here"
        ```
        Once added, view counts for official Marvel and WB trailers
        will update automatically on every page load and feed the audience
        score calibration.
        """)


# ── TAB 4: DISTRIBUTIONS ──────────────────────────────────────────────────────
with tab4:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>NET PROFIT DISTRIBUTIONS — SCENARIO A (BOTH HOLD)</p>",
                unsafe_allow_html=True)

    dune_profits = results["A_Both_Hold"]["DUNE"]["profits"]
    av_profits   = results["A_Both_Hold"]["AVENGERS"]["profits"]

    fig4 = go.Figure()
    for profits, name, color in [
        (dune_profits, "Dune",     P["dune"]),
        (av_profits,   "Avengers", P["av"]),
    ]:
        fig4.add_trace(go.Histogram(
            x=profits, nbinsx=60, name=name,
            marker_color=color, opacity=0.55,
            histnorm="probability density",
        ))
        p50 = float(np.median(profits))
        fig4.add_vline(
            x=p50, line_dash="dash", line_width=1, line_color=color,
            annotation_text=f"{name} P50 ${p50:.0f}M",
            annotation_font_color=color, annotation_font_size=9,
        )

    fig4.add_vline(
        x=0, line_width=0.5, line_color=P["axis"],
        annotation_text="Break-even",
        annotation_font_color=P["mid_ref"], annotation_font_size=9,
    )
    fig4.update_layout(**_layout(
        P, barmode="overlay", height=380,
        xaxis_title="Net Profit ($M)", yaxis_title="Probability Density",
    ))
    st.plotly_chart(fig4, use_container_width=True)

    rows = []
    for sk in list(SCENARIOS.keys()):
        rd, ra = results[sk]["DUNE"], results[sk]["AVENGERS"]
        rows.append({
            "Scenario": SCENARIOS[sk]["label"],
            "Dune P10": f"${rd['p10']:.0f}M", "Dune P50": f"${rd['p50']:.0f}M",
            "Dune P90": f"${rd['p90']:.0f}M", "Dune BE%": f"{rd['breakeven_pct']:.0f}%",
            "Av P10":   f"${ra['p10']:.0f}M", "Av P50":   f"${ra['p50']:.0f}M",
            "Av P90":   f"${ra['p90']:.0f}M", "Av BE%":   f"{ra['breakeven_pct']:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 5: DISNEY DECISION ────────────────────────────────────────────────────
with tab5:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>SHOULD DISNEY MOVE? — DECISION FRAMEWORK</p>",
                unsafe_allow_html=True)

    sc_a_av = results["A_Both_Hold"]["AVENGERS"]["p50"]
    sc_b_av = results["B_Disney_May"]["AVENGERS"]["p50"]
    sc_c_av = results["C_Disney_Jan"]["AVENGERS"]["p50"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Hold Dec 18 P50", f"${sc_a_av:.0f}M",
              delta=f"{results['A_Both_Hold']['AVENGERS']['breakeven_pct']:.0f}% BE")
    c2.metric("Move to May P50", f"${sc_b_av:.0f}M",
              delta=f"+${sc_b_av - sc_a_av:.0f}M vs holding")
    c3.metric("Move to Jan P50", f"${sc_c_av:.0f}M",
              delta=f"+${sc_c_av - sc_a_av:.0f}M vs holding",
              delta_color="normal" if sc_c_av > sc_a_av else "inverse")

    st.divider()
    prob_data = {
        "Window":      ["Now → Apr 12", "CinemaCon Apr 13–20", "Apr 21 – Jul 1", "Jul 1+", "Never"],
        "Probability": [8, 35, 15, 4, 38],
        "Trigger": [
            "Reshoot crisis",
            "Trailer underperforms + exhibitor pressure",
            "Dune trailer is a phenomenon",
            "Film emergency",
            "Hold Dec 18, absorb IMAX hit, own Dunesday narrative",
        ],
    }

    # gradient: av red → amber → dune gold, encoding urgency
    bar_colors = [P["av"], "#c05020", "#a07020", "#706040", P["dune"]]
    fig5 = go.Figure(go.Bar(
        x=prob_data["Window"],
        y=prob_data["Probability"],
        marker_color=bar_colors,
        text=[f"{p}%" for p in prob_data["Probability"]],
        textposition="outside",
        textfont=dict(size=11, color=P["chart_font"]),
    ))
    fig5.add_hline(y=0, line_width=0.5, line_color=P["axis"])
    fig5.update_layout(**_layout(P, height=280, yaxis_title="%", yaxis_range=[0, 50]))
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"""
    <div class='verdict-box'>
    <b style='color:{P['dune']}'>This is Walden & D'Amaro's first major theatrical decision together.</b><br><br>
    The model sets the floor — financial stakes are quantified. Everything above the floor is
    judgment, franchise strategy, competitive psychology, and institutional ego.<br><br>
    <b>What holding says:</b> We trust the franchise. Marvel doesn't blink.<br>
    <b>What moving says:</b> We're strategic operators, not sentimentalists.<br><br>
    The trailer is the permission structure. If it hits → hold. If it lands soft →
    the move conversation becomes real. <b>CinemaCon April 16 is the decision point.</b>
    <br><br>
    <span style='color:{P['dim']}; font-size:0.78rem'>
    Live signals: {' · '.join(cal['sources'])}
    </span>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<p style='color:{P['footer']}; font-size:0.6rem; letter-spacing:1.5px;
   margin-top:12px; text-align:right;'>
DUNESDAY v5 &nbsp;·&nbsp; {signals['last_updated']} &nbsp;·&nbsp; {' · '.join(cal['sources'])}
</p>
""", unsafe_allow_html=True)
