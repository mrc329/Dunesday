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

# ── PALETTE ───────────────────────────────────────────────────────────────────
DUNE_COLOR  = "#d4a030"    # Arrakis amber — spice, desert sun
AV_COLOR    = "#cc2030"    # Avengers crimson — deeper, more saturated
CYAN_COLOR  = "#00b4c2"    # teal reference line (muted)
BG_COLOR    = "#04060c"    # near-black
PANEL_COLOR = "#070a12"    # barely lighter (charts only)
TEXT_COLOR  = "#cdc8be"    # warm off-white — high contrast, eye-easy
DIM_COLOR   = "#3c4e62"    # muted labels / section heads
AXIS_COLOR  = "rgba(255,255,255,0.09)"

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* layout density */
  .block-container {{
    padding-top: 0.6rem !important;
    padding-bottom: 0.5rem !important;
  }}
  .main {{ background-color: {BG_COLOR}; }}

  /* metric cards — Tufte: top rule, no box */
  .stMetric {{
    background: transparent !important;
    border: none !important;
    border-top: 1px solid #141e2c !important;
    padding: 10px 0 6px !important;
    border-radius: 0 !important;
  }}
  [data-testid="stMetricValue"] {{
    font-size: 1.65rem !important;
    font-variant-numeric: tabular-nums !important;
    font-weight: 700 !important;
    color: {TEXT_COLOR} !important;
    line-height: 1.15 !important;
    letter-spacing: -0.5px !important;
  }}
  [data-testid="stMetricLabel"] {{
    font-size: 0.58rem !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: {DIM_COLOR} !important;
    font-weight: 400 !important;
  }}
  [data-testid="stMetricDelta"] svg {{ display: none !important; }}
  [data-testid="stMetricDelta"] > div {{
    font-size: 0.68rem !important;
    font-variant-numeric: tabular-nums !important;
  }}

  /* verdict box */
  .verdict-box {{
    border-left: 2px solid {DUNE_COLOR};
    padding: 14px 20px;
    margin: 8px 0;
    line-height: 1.8;
  }}

  /* sidebar */
  div[data-testid="stSidebarContent"] {{ background: #040710; }}

  /* tab labels */
  .stTabs [data-baseweb="tab"] {{
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    padding: 6px 14px !important;
  }}

  /* dataframe */
  [data-testid="stDataFrame"] {{ font-size: 0.78rem !important; }}
</style>
""", unsafe_allow_html=True)


def _layout(**kw):
    """Tufte-style Plotly base layout — transparent bg, no gridlines."""
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor=BG_COLOR,
        font=dict(color="#8a9daf", size=11),
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
            linecolor=AXIS_COLOR,
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


# ── HEADER — compact, inline, no wasted vertical space ───────────────────────
st.markdown(f"""
<div style='padding: 4px 0 12px; border-bottom: 1px solid #141e2c; margin-bottom: 14px;'>
  <span style='font-size:1.4rem; font-weight:700; letter-spacing:5px; color:{TEXT_COLOR};'>
    <span style='color:{DUNE_COLOR}'>DUNE</span>SDAY
  </span>
  <span style='color:{DIM_COLOR}; font-size:0.62rem; letter-spacing:3px; margin-left:20px;
               vertical-align: middle;'>
    BOX OFFICE MODEL &nbsp;·&nbsp; DEC 18 2026 &nbsp;·&nbsp; LIVE SIGNALS
  </span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Controls")

    st.markdown("**Base Audience Scores**")
    st.caption("Live signals auto-calibrate from these baselines")
    base_dune_aud = st.slider("Dune base score", 60, 100, 87)
    base_av_aud   = st.slider("Avengers base score", 60, 100, 88)

    st.divider()

    with st.spinner("Fetching live signals..."):
        signals = load_signals(base_dune_aud, base_av_aud)

    cal = signals["calibration"]
    confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    conf_icon = confidence_colors.get(cal["signal_confidence"], "⚪")

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
    <div style='font-size:0.75rem; color:{DIM_COLOR}; line-height:1.9;'>
    Dune exclusive: <b style='color:{DUNE_COLOR}'>Days 1–21</b><br>
    Dune screens: <b style='color:{DUNE_COLOR}'>400</b><br>
    Avengers day 1: <b style='color:{AV_COLOR}'>0 screens</b><br>
    Avengers first IMAX: <b style='color:{AV_COLOR}'>Jan 8</b>
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


# ── KPI ROW ───────────────────────────────────────────────────────────────────
st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2.5px; color:{DIM_COLOR}; margin:4px 0 8px;'>VERDICT — BOTH HOLDING DEC 18</p>",
            unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
sc_a = results["A_Both_Hold"]

with k1:
    st.metric("Dune P50 Profit", f"${sc_a['DUNE']['p50']:.0f}M",
              delta="100% break-even")
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
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR}; margin-bottom:10px;'>NET PROFIT BY SCENARIO — P10 / P50 / P90</p>",
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
        marker_color=DUNE_COLOR, offsetgroup=0,
        text=[f"${v:.0f}M" for v in dune_p50s],
        textposition="outside",
        textfont=dict(size=10, color=DUNE_COLOR),
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50 for p90, p50 in zip(dune_p90s, dune_p50s)],
            arrayminus=[p50 - p10 for p50, p10 in zip(dune_p50s, dune_p10s)],
            color=DUNE_COLOR, thickness=1.5, width=5,
        ),
    ))
    fig.add_trace(go.Bar(
        name="Avengers", x=sc_labels, y=av_p50s,
        marker_color=AV_COLOR, offsetgroup=1,
        text=[f"${v:.0f}M" for v in av_p50s],
        textposition="outside",
        textfont=dict(size=10, color=AV_COLOR),
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50 for p90, p50 in zip(av_p90s, av_p50s)],
            arrayminus=[p50 - p10 for p50, p10 in zip(av_p50s, av_p10s)],
            color=AV_COLOR, thickness=1.5, width=5,
        ),
    ))
    fig.add_hline(y=0, line_width=0.5, line_color=AXIS_COLOR)
    fig.update_layout(**_layout(
        barmode="group", height=390,
        yaxis_title="Net Profit ($M)",
    ))
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
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR}; margin-bottom:10px;'>IMAX SCREEN ALLOCATION — DAYS 1–45</p>",
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
                          name="Dune", marker_color=DUNE_COLOR), row=1, col=1)
    fig2.add_trace(go.Bar(x=date_labels, y=av_screens,
                          name="Avengers", marker_color=AV_COLOR), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["dune_daily"],
        name="Dune IMAX rev",
        line=dict(color=DUNE_COLOR, width=2),
        fill="tozeroy", fillcolor=f"rgba(212,160,48,0.12)",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["avengers_daily"],
        name="Avengers IMAX rev",
        line=dict(color=AV_COLOR, width=2),
        fill="tozeroy", fillcolor=f"rgba(204,32,48,0.10)",
    ), row=2, col=1)

    for day, color, label in [
        (7,  "rgba(255,255,255,0.15)", "Wk 1"),
        (14, "rgba(255,255,255,0.15)", "Wk 2"),
        (21, CYAN_COLOR,               "Day 21"),
    ]:
        for row in [1, 2]:
            fig2.add_vline(
                x=date_labels[day], line_dash="dot",
                line_color=color, line_width=0.8,
                annotation_text=label if row == 1 else "",
                annotation_font_color=color,
                annotation_font_size=9,
                row=row, col=1,
            )

    fig2.update_layout(**_layout(
        height=480, barmode="stack",
        margin=dict(t=32, b=10, l=4, r=8),
    ))
    fig2.update_xaxes(tickangle=45, nticks=15, showgrid=False,
                      showline=True, linecolor=AXIS_COLOR, linewidth=0.5)
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
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR}; margin-bottom:10px;'>LIVE SIGNAL DASHBOARD</p>",
                unsafe_allow_html=True)

    conf       = cal["signal_confidence"]
    conf_label = {
        "high":   "HIGH — multiple live sources",
        "medium": "MEDIUM — partial live data",
        "low":    "LOW — fallback values only",
    }.get(conf, conf)
    conf_color = {"high": DUNE_COLOR, "medium": "#d48020", "low": AV_COLOR}.get(conf, "#888")

    st.markdown(f"""
    <div style='border-left:2px solid {conf_color}; padding:8px 14px; margin-bottom:14px;'>
      <span style='color:{conf_color}; font-size:0.6rem; letter-spacing:2px'>
        SIGNAL CONFIDENCE: {conf_label}
      </span><br>
      <span style='color:#7a8fa0; font-size:0.82rem'>{cal.get("notes", "")}</span>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<span style='color:{AV_COLOR}; font-size:0.82rem; font-weight:600; letter-spacing:2px;'>AVENGERS: DOOMSDAY</span>",
                    unsafe_allow_html=True)

        av_sig  = signals["avengers"]
        teasers = av_sig.get("teaser_views_x_M", [])

        if teasers:
            labels = [f"T{i+1}" for i in range(len(teasers))]
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=labels, y=teasers,
                marker_color=AV_COLOR,
                text=[f"{v:.0f}M" for v in teasers],
                textposition="outside",
                textfont=dict(size=10, color=AV_COLOR),
                showlegend=False,
            ))
            fig_d.add_trace(go.Scatter(
                x=labels, y=teasers,
                line=dict(color=AV_COLOR, dash="dot", width=1),
                mode="lines", showlegend=False,
            ))
            t1 = teasers[0]
            fig_d.add_hline(
                y=t1 * 0.77, line_dash="dash", line_width=0.8,
                line_color=DUNE_COLOR, opacity=0.7,
                annotation_text="D&W held 77%",
                annotation_font_color=DUNE_COLOR, annotation_font_size=9,
            )
            fig_d.add_hline(
                y=t1 * 0.47, line_dash="dash", line_width=0.8,
                line_color="#667788", opacity=0.6,
                annotation_text="L&T soft 47%",
                annotation_font_color="#667788", annotation_font_size=9,
            )
            fig_d.update_layout(**_layout(
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
            decay_color  = {"strong": DUNE_COLOR, "neutral": "#667788", "soft": AV_COLOR}.get(decay_signal, "#667788")
            st.markdown(f"""
            <div style='border-left:2px solid {decay_color}; padding:7px 12px; margin-top:8px;'>
              <span style='color:{decay_color}; font-size:0.6rem; letter-spacing:2px'>
                DECAY: {decay_signal.upper()}
              </span><br>
              <span style='color:#5a6e80; font-size:0.76rem'>
                T1→T2: {teasers[0]:.0f}M → {teasers[1]:.0f}M
                ({(teasers[1]/teasers[0]*100):.0f}% of T1)
                {'— matches Love&Thunder pattern' if (teasers[1]/teasers[0]) < 0.55 else '— tracking neutral'}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"<span style='color:{DUNE_COLOR}; font-size:0.82rem; font-weight:600; letter-spacing:2px;'>DUNE: PART THREE</span>",
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
            name="Avengers", marker_color=AV_COLOR,
            text=[f"Avengers {av_t / total_t * 100:.0f}%"],
            textposition="inside",
            textfont=dict(size=10, color="white"),
        ))
        fig_ratio.add_trace(go.Bar(
            x=["Search Interest Share"],
            y=[dune_t / total_t * 100],
            name="Dune", marker_color=DUNE_COLOR,
            text=[f"Dune {dune_t / total_t * 100:.0f}%"],
            textposition="inside",
            textfont=dict(size=10, color="#1a1000"),
        ))
        fig_ratio.add_hline(
            y=18, line_dash="dot", line_width=0.8,
            line_color=DUNE_COLOR, opacity=0.6,
            annotation_text="Expected Dune baseline (no trailer)",
            annotation_font_color=DUNE_COLOR, annotation_font_size=9,
        )
        fig_ratio.update_layout(**_layout(
            barmode="stack", height=180,
            yaxis_title="%",
        ))
        st.plotly_chart(fig_ratio, use_container_width=True)

        st.caption("Dune's 13/100 vs Avengers 72/100 is marketing stage, not demand. "
                   "Dune has released zero promotional materials.")

    st.divider()
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR};'>HOW SIGNALS FEED THE MODEL</p>",
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
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR}; margin-bottom:10px;'>NET PROFIT DISTRIBUTIONS — SCENARIO A (BOTH HOLD)</p>",
                unsafe_allow_html=True)

    dune_profits = results["A_Both_Hold"]["DUNE"]["profits"]
    av_profits   = results["A_Both_Hold"]["AVENGERS"]["profits"]

    fig4 = go.Figure()
    for profits, name, color in [
        (dune_profits, "Dune",     DUNE_COLOR),
        (av_profits,   "Avengers", AV_COLOR),
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
        x=0, line_width=0.5, line_color=AXIS_COLOR,
        annotation_text="Break-even",
        annotation_font_color="#667788", annotation_font_size=9,
    )
    fig4.update_layout(**_layout(
        barmode="overlay", height=380,
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
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{DIM_COLOR}; margin-bottom:10px;'>SHOULD DISNEY MOVE? — DECISION FRAMEWORK</p>",
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
        "Trigger":     [
            "Reshoot crisis",
            "Trailer underperforms + exhibitor pressure",
            "Dune trailer is a phenomenon",
            "Film emergency",
            "Hold Dec 18, absorb IMAX hit, own Dunesday narrative",
        ],
    }
    fig5 = go.Figure(go.Bar(
        x=prob_data["Window"],
        y=prob_data["Probability"],
        marker_color=[AV_COLOR, "#d44020", "#c07010", "#804020", DUNE_COLOR],
        text=[f"{p}%" for p in prob_data["Probability"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig5.add_hline(y=0, line_width=0.5, line_color=AXIS_COLOR)
    fig5.update_layout(**_layout(
        height=280, yaxis_title="%", yaxis_range=[0, 50],
    ))
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(pd.DataFrame(prob_data), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(f"""
    <div class='verdict-box'>
    <b style='color:{DUNE_COLOR}'>This is Walden & D'Amaro's first major theatrical decision together.</b><br><br>
    The model sets the floor — financial stakes are quantified. Everything above the floor is
    judgment, franchise strategy, competitive psychology, and institutional ego.<br><br>
    <b>What holding says:</b> We trust the franchise. Marvel doesn't blink.<br>
    <b>What moving says:</b> We're strategic operators, not sentimentalists.<br><br>
    The trailer is the permission structure. If it hits → hold. If it lands soft →
    the move conversation becomes real. <b>CinemaCon April 16 is the decision point.</b>
    <br><br>
    <span style='color:{DIM_COLOR}; font-size:0.78rem'>
    Live signals: {' · '.join(cal['sources'])}
    </span>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<p style='color:#1e3048; font-size:0.6rem; letter-spacing:1.5px; margin-top:12px; text-align:right;'>
DUNESDAY v5 &nbsp;·&nbsp; {signals['last_updated']} &nbsp;·&nbsp; {' · '.join(cal['sources'])}
</p>
""", unsafe_allow_html=True)
