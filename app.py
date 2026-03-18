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

from model.config import (FILM_PARAMS, IMAX_CONFIG,
                          AUDIENCE_SCORE_BENCHMARKS,
                          SPIDEY_IMPACT_ADJ, WEEKLY_DECAY_BENCHMARKS)
from model.core import run_all_scenarios, imax_gap_summary, SCENARIOS
from model.signals import (fetch_and_calibrate, YOUTUBE_TRAILER_URLS,
                           YOUTUBE_VIDEO_IDS, fetch_youtube_views)

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
# We use JS-based OS detection: JS reads prefers-color-scheme and sets ?sys_theme=
# in the URL on first load so Python can read the real system preference.
_qs_theme = st.query_params.get("sys_theme", "")

if "theme" not in st.session_state:
    if _qs_theme in ("light", "dark"):
        st.session_state.theme = _qs_theme
    else:
        st.session_state.theme = "light"   # Tufte light as safe default

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

  /* ── plotly — prevent bar labels from clipping at chart boundary ── */
  .js-plotly-plot .plotly .main-svg {{
    overflow: visible !important;
  }}
  .js-plotly-plot .plotly {{
    overflow: visible !important;
  }}

  /* ── video embeds — full-width, consistent aspect ratio ── */
  [data-testid="stVideo"] {{
    width: 100% !important;
  }}
  [data-testid="stVideo"] iframe,
  [data-testid="stVideo"] video {{
    width: 100% !important;
    aspect-ratio: 16/9 !important;
    height: auto !important;
    border: none !important;
  }}

  /* ── tab content — ensure no section collides with the tab bar ── */
  .stTabs [data-baseweb="tab-panel"] {{
    padding-top: 16px !important;
  }}

  /* ── toggle / slider labels ── */
  [data-testid="stToggleLabel"] p,
  [data-testid="stSliderLabel"] p,
  [data-testid="stSelectSliderLabel"] p {{
    color: {P['text']} !important;
  }}
</style>
"""

st.markdown(build_css(P), unsafe_allow_html=True)

# ── OS THEME DETECTION (runs once on first load, before sys_theme param is set) ─
# Reads the browser's prefers-color-scheme and redirects with ?sys_theme=<value>
# so the Python side can set session state to match the user's OS preference.
if not _qs_theme:
    import streamlit.components.v1 as _cv1
    _cv1.html("""<script>
    (function() {
        var dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        var t = dark ? 'dark' : 'light';
        var u = new URL(window.parent.location.href);
        if (u.searchParams.get('sys_theme') !== t) {
            u.searchParams.set('sys_theme', t);
            window.parent.location.replace(u.toString());
        }
    })();
    </script>""", height=0, scrolling=False)


# ── CHART LAYOUT HELPER ────────────────────────────────────────────────────────
def _layout(P: dict, outside_text: bool = False, **kw) -> dict:
    """Tufte-style Plotly base layout — transparent bg, no gridlines.

    outside_text=True adds extra top margin so textposition='outside' bars
    are never clipped by the chart frame.
    """
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
        margin=dict(t=44 if outside_text else 20, b=30, l=48, r=16),
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_yt_stats() -> dict:
    """Fetch per-video YouTube stats for all known trailer IDs."""
    ids = [v for v in YOUTUBE_VIDEO_IDS.values() if v]
    return fetch_youtube_views(video_ids=ids)


# Reverse map: video_id → slot name (e.g. "399Ez7WHK5s" → "avengers_t1")
_YT_ID_TO_SLOT = {v: k for k, v in YOUTUBE_VIDEO_IDS.items() if v}

# Human-readable labels for each slot
_YT_SLOT_LABELS = {
    "avengers_t1":        "Avengers T1",
    "avengers_t2":        "Avengers T2",
    "avengers_t3":        "Avengers T3",
    "avengers_t4":        "Avengers T4",
    "avengers_countdown": "Avengers Countdown",
    "dune_t1":            "Dune T1",
}

# Ordered teaser slots for decay chart
_AV_TEASER_SLOTS = ["avengers_t1", "avengers_t2", "avengers_t3", "avengers_t4"]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Theme toggle — top of sidebar, before everything else
    theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(theme_label, use_container_width=False):
        new_theme = "light" if st.session_state.theme == "dark" else "dark"
        st.session_state.theme = new_theme
        st.query_params["sys_theme"] = new_theme
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Model Controls")

    st.markdown("**Base Audience Scores**")
    st.caption(
        "PostTrak/CinemaScore-style satisfaction index (0–100). "
        "Live signals auto-calibrate from these baselines."
    )
    base_dune_aud = st.slider("Dune base score", 60, 100, 87)
    base_av_aud   = st.slider("Avengers base score", 60, 100, 88)

    with st.expander("Score benchmarks"):
        bm_rows = [
            {"Film": f"{t} ({yr})", "Score": s,
             "Film group": g.title()}
            for t, yr, s, g in sorted(
                AUDIENCE_SCORE_BENCHMARKS, key=lambda x: -x[2]
            )
        ]
        st.dataframe(pd.DataFrame(bm_rows), hide_index=True,
                     use_container_width=True)

    st.divider()

    with st.spinner("Fetching live signals..."):
        signals  = load_signals(base_dune_aud, base_av_aud)
        yt_stats = load_yt_stats()

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
    st.markdown("**Competitor Signals**")
    spidey_tier = st.select_slider(
        "Spider-Man: Brand New Day (Jul 25 2026)",
        options=["Disappoints", "Soft", "Neutral", "Strong", "Blockbuster"],
        value="Neutral",
    )
    spidey_adj = SPIDEY_IMPACT_ADJ[spidey_tier]
    spidey_color = (P["av"] if spidey_adj < 0 else
                    P["dune"] if spidey_adj > 0 else P["dim"])
    st.caption(
        f"MCU brand signal → Avengers score "
        f"**{spidey_adj:+d} pts** · OW mult "
        f"{'↑' if spidey_adj > 0 else '↓' if spidey_adj < 0 else '—'}"
    )

    st.divider()
    override = st.toggle("Manual score override", value=False)
    if override:
        dune_aud = st.slider("Dune (manual)", 60, 100, int(cal["dune_calibrated"]))
        av_aud   = st.slider("Avengers (manual)", 60, 100, int(cal["avengers_calibrated"]))
        st.caption("⚠️ Overriding live calibration")
    else:
        dune_aud = int(cal["dune_calibrated"])
        # Apply Spider-Man MCU brand signal on top of live calibration
        av_aud   = int(np.clip(cal["avengers_calibrated"] + spidey_adj, 60, 100))

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
        spidey_tier=spidey_tier,
    )
    imax = imax_gap_summary()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:4px 0 12px; border-bottom:1px solid {P['card_rule']}; margin-bottom:14px;
            display:flex; align-items:baseline; gap:20px; flex-wrap:wrap;'>
  <span style='font-size:2rem; font-weight:800; letter-spacing:-0.5px; line-height:1;
               color:{P['text']}; font-family:Georgia, "Times New Roman", serif;'>
    <span style='color:{P['dune']}'>Dune</span>sday
  </span>
  <span style='color:{P['dim']}; font-size:0.65rem; letter-spacing:2px;
               text-transform:uppercase; line-height:1;'>
    Box Office Model &nbsp;·&nbsp; Dec 18 2026 &nbsp;·&nbsp; Live Signals
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "SCENARIOS", "IMAX TIMELINE", "LIVE SIGNALS", "DISTRIBUTIONS", "DISNEY DECISION", "TRAILERS",
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
        cliponaxis=False,
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
        cliponaxis=False,
        error_y=dict(
            type="data", symmetric=False,
            array=[p90 - p50 for p90, p50 in zip(av_p90s, av_p50s)],
            arrayminus=[p50 - p10 for p50, p10 in zip(av_p50s, av_p10s)],
            color=P["av"], thickness=1.5, width=5,
        ),
    ))
    fig.add_hline(y=0, line_width=0.5, line_color=P["axis"])
    fig.update_layout(**_layout(P, outside_text=True,
                                barmode="group", height=420,
                                yaxis_title="Net Profit ($M)"))
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

    fig2.update_layout(**_layout(P, height=520, barmode="stack",
                                 margin=dict(t=44, b=40, l=52, r=16)))
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

    # ── Parse YouTube API results ─────────────────────────────────────────────
    yt_videos   = yt_stats.get("videos", {}) if yt_stats.get("status") == "ok" else {}
    yt_live     = bool(yt_videos)
    yt_fetched  = yt_stats.get("fetched_at", "")

    def _yt_views_M(slot: str) -> float | None:
        """Return YouTube view count in millions for a slot, or None."""
        vid_id = YOUTUBE_VIDEO_IDS.get(slot)
        if not vid_id or vid_id not in yt_videos:
            return None
        return yt_videos[vid_id]["views"] / 1_000_000

    # Avengers teaser views — prefer YouTube live, fall back to X/Twitter
    av_sig   = signals["avengers"]
    dune_sig = signals["dune"]
    yt_av_teasers = [_yt_views_M(s) for s in _AV_TEASER_SLOTS]
    use_yt_teasers = any(v is not None for v in yt_av_teasers)

    if use_yt_teasers:
        teasers      = [v for v in yt_av_teasers if v is not None]
        teaser_src   = "YouTube"
    else:
        teasers    = av_sig.get("teaser_views_x_M", [])
        teaser_src = "X/Twitter"

    col_a, col_b = st.columns(2)

    with col_a:
        _av_color = P["av"]
        st.markdown(f"<div style='color:{_av_color}; font-size:0.9rem; font-weight:700; letter-spacing:2px; margin-bottom:10px; padding-bottom:4px; border-bottom:1px solid {_av_color};'>AVENGERS: DOOMSDAY</div>",
                    unsafe_allow_html=True)

        if teasers:
            # If YouTube has data, show all 4 slots (zero-fill any missing)
            if use_yt_teasers:
                labels_decay = [_YT_SLOT_LABELS[s].replace("Avengers ", "")
                                for s in _AV_TEASER_SLOTS]
                vals_decay   = [_yt_views_M(s) or 0.0 for s in _AV_TEASER_SLOTS]
            else:
                labels_decay = [f"T{i+1}" for i in range(len(teasers))]
                vals_decay   = teasers

            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=labels_decay, y=vals_decay,
                marker_color=P["av"],
                text=[f"{v:.0f}M" for v in vals_decay],
                textposition="outside",
                textfont=dict(size=10, color=P["av"]),
                cliponaxis=False,
                showlegend=False,
            ))
            fig_d.add_trace(go.Scatter(
                x=labels_decay, y=vals_decay,
                line=dict(color=P["av"], dash="dot", width=1),
                mode="lines", showlegend=False,
            ))
            t1 = vals_decay[0] if vals_decay[0] > 0 else 1
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
                P, outside_text=True,
                title=dict(text=f"Teaser Decay — {teaser_src} Views",
                           font=dict(size=11), x=0),
                height=300, yaxis_title="Views (M)",
            ))
            st.plotly_chart(fig_d, use_container_width=True)

        _cal_sources = cal.get("sources", [])
        _trends_live = "Google Trends" in _cal_sources
        _reddit_live = "Reddit API" in _cal_sources

        # YouTube total views across all 4 teasers
        _av_metrics = st.columns(2) if _trends_live else None
        if yt_live:
            av_yt_total = sum(
                yt_videos[YOUTUBE_VIDEO_IDS[s]]["views"]
                for s in _AV_TEASER_SLOTS
                if YOUTUBE_VIDEO_IDS.get(s) and YOUTUBE_VIDEO_IDS[s] in yt_videos
            )
            _yt_col = _av_metrics[0] if _av_metrics else st
            _yt_col.metric("YT teaser views",
                           f"{av_yt_total / 1_000_000:.0f}M" if av_yt_total else "—",
                           delta=f"T1–T4 combined · {yt_fetched[:10]}")
        else:
            _yt_col = _av_metrics[0] if _av_metrics else st
            _yt_col.metric("YT trailer views",
                           f"{av_sig['yt_trailer_views']:,}" if av_sig.get("yt_trailer_views") else "—",
                           delta="Full trailer not released" if not av_sig.get("full_trailer_out") else "Live")

        if _trends_live:
            _av_metrics[1].metric("Trends interest",
                                  f"{av_sig.get('trends_interest', '—')}/100",
                                  delta="Google Trends US")

        if _reddit_live and (av_sig.get("reddit_hot_avg") is not None or av_sig.get("reddit_posts_24h") is not None):
            r1, r2 = st.columns(2)
            r1.metric("r/marvelstudios hot avg",
                      f"{av_sig['reddit_hot_avg']:,.0f}" if av_sig.get("reddit_hot_avg") is not None else "—",
                      delta="Upvote velocity")
            r2.metric("r/marvelstudios posts/24h",
                      str(av_sig["reddit_posts_24h"]) if av_sig.get("reddit_posts_24h") is not None else "—",
                      delta="Post volume")

        if len(teasers) >= 2 and teasers[0] > 0:
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
                &nbsp;·&nbsp; source: {teaser_src}
              </span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        _dune_color = P["dune"]
        st.markdown(f"<div style='color:{_dune_color}; font-size:0.9rem; font-weight:700; letter-spacing:2px; margin-bottom:10px; padding-bottom:4px; border-bottom:1px solid {_dune_color};'>DUNE: PART THREE</div>",
                    unsafe_allow_html=True)

        dune_yt_views = _yt_views_M("dune_t1")
        dune_color    = P["dune"]
        dune_dim      = P["dim"]
        if dune_yt_views is not None:
            st.markdown(
                f"<div style='border-left:2px solid {dune_color}; padding:8px 14px; "
                f"font-size:0.82rem; color:{dune_dim};'>"
                f"Trailer live · <b style='color:{dune_color}'>{dune_yt_views:.0f}M</b> YouTube views</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No trailer released. WB following Part Two marketing cadence — strategic delay.")

        _dune_m1, _dune_m2 = st.columns(2)
        _dune_m2.metric("Alamo poll", "#1 Most Anticipated", delta="14,000 respondents")
        if _trends_live:
            _dune_m1.metric("Trends interest",
                            f"{dune_sig.get('trends_interest', '—')}/100",
                            delta=f"vs Avengers {av_sig.get('trends_interest', '?')}/100")
        else:
            _dune_m1.empty()

        if _reddit_live and (dune_sig.get("reddit_hot_avg") is not None or dune_sig.get("reddit_posts_24h") is not None):
            r1, r2 = st.columns(2)
            r1.metric("r/dune hot avg",
                      f"{dune_sig['reddit_hot_avg']:,.0f}" if dune_sig.get("reddit_hot_avg") is not None else "—",
                      delta="Upvote velocity")
            r2.metric("r/dune posts/24h",
                      str(dune_sig["reddit_posts_24h"]) if dune_sig.get("reddit_posts_24h") is not None else "—",
                      delta="Post volume")

        if _trends_live:
            av_t   = av_sig.get("trends_interest", 72)
            dune_t = dune_sig.get("trends_interest", 13)
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
            fig_ratio.update_layout(**_layout(P, barmode="stack", height=260, yaxis_title="%"))
            st.plotly_chart(fig_ratio, use_container_width=True)
            st.caption("Dune's 13/100 vs Avengers 72/100 is marketing stage, not demand. "
                       "Dune has released zero promotional materials.")

    # ── YouTube per-video stats table ─────────────────────────────────────────
    if yt_live:
        st.divider()
        st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>YOUTUBE TRAILER VIEWS — LIVE DATA</p>",
                    unsafe_allow_html=True)

        yt_rows = []
        for slot in list(_AV_TEASER_SLOTS) + ["avengers_countdown", "dune_t1"]:
            vid_id = YOUTUBE_VIDEO_IDS.get(slot)
            if not vid_id:
                continue
            label = _YT_SLOT_LABELS.get(slot, slot)
            if vid_id in yt_videos:
                vd = yt_videos[vid_id]
                yt_rows.append({
                    "Video":  label,
                    "Title":  vd["title"][:60] + ("…" if len(vd["title"]) > 60 else ""),
                    "Views":  f"{vd['views'] / 1_000_000:.1f}M",
                    "Likes":  f"{vd['likes'] / 1_000:.0f}K",
                })
            else:
                yt_rows.append({"Video": label, "Title": "—", "Views": "—", "Likes": "—"})

        st.dataframe(pd.DataFrame(yt_rows), use_container_width=True, hide_index=True)

    # ── Avengers Projected Decay Rate ─────────────────────────────────────────
    st.divider()
    st.markdown(
        f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>"
        "AVENGERS: PROJECTED WEEKLY HOLD CURVE</p>",
        unsafe_allow_html=True,
    )

    wk_labels = ["OW", "Wk 2", "Wk 3", "Wk 4", "Wk 5", "Wk 6", "Wk 7"]

    # Build projected curve from calibrated audience score via WOM
    from model.core import wom_mult as _wom_mult
    wm_proj  = float(np.clip(_wom_mult(av_aud), 0.5, 1.5))
    base_holds = [1.00, 0.56, 0.43, 0.34, 0.27, 0.22, 0.18]
    proj_holds = [base_holds[0]] + [
        float(np.clip(h * wm_proj, 0.05, 1.05))
        for h in base_holds[1:]
    ]
    proj_pct = [v * 100 for v in proj_holds]

    fig_decay = go.Figure()

    # Benchmark bands (light, background first)
    bench_styles = {
        "Endgame (strong)":    dict(dash="dot",   color=P["dune"],    opacity=0.55),
        "D&W / held well":     dict(dash="dot",   color=P["dune"],    opacity=0.30),
        "Neutral MCU":         dict(dash="dash",  color=P["mid_ref"], opacity=0.70),
        "Love&Thunder (soft)": dict(dash="dot",   color=P["av"],      opacity=0.55),
    }
    for bname, bvals in WEEKLY_DECAY_BENCHMARKS.items():
        sty = bench_styles[bname]
        fig_decay.add_trace(go.Scatter(
            x=wk_labels, y=[v * 100 for v in bvals],
            name=bname,
            mode="lines",
            line=dict(dash=sty["dash"], color=sty["color"],
                      width=1.2),
            opacity=sty["opacity"],
        ))

    # Projected curve — prominent
    fig_decay.add_trace(go.Scatter(
        x=wk_labels, y=proj_pct,
        name=f"Projected (score {av_aud})",
        mode="lines+markers",
        line=dict(color=P["av"], width=2.5),
        marker=dict(size=6, color=P["av"]),
    ))

    # Annotate each projected point
    for i, (lbl, val) in enumerate(zip(wk_labels, proj_pct)):
        if i == 0:
            continue
        fig_decay.add_annotation(
            x=lbl, y=val,
            text=f"{val:.0f}%",
            font=dict(color=P["av"], size=9),
            showarrow=False,
            yanchor="bottom",
            yshift=6,
        )

    fig_decay.update_layout(**_layout(
        P, outside_text=True,
        height=300,
        yaxis_title="% of OW gross",
        yaxis_ticksuffix="%",
        yaxis_range=[0, 115],
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=9), orientation="h", y=1.06, x=0),
    ))
    st.plotly_chart(fig_decay, use_container_width=True)

    # Decay delta vs neutral
    delta_vs_neutral = proj_pct[1] - WEEKLY_DECAY_BENCHMARKS["Neutral MCU"][1] * 100
    decay_note_color = P["dune"] if delta_vs_neutral >= 0 else P["av"]
    st.caption(
        f"Wk 2 hold projected at **{proj_pct[1]:.0f}%** of OW "
        f"({delta_vs_neutral:+.0f}% vs neutral MCU) "
        f"· WOM multiplier {wm_proj:.2f}× · audience score {av_aud}"
        + (f" (incl. Spider-Man {spidey_adj:+d}pt)" if spidey_adj != 0 else "")
    )

    st.divider()
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>HOW SIGNALS FEED THE MODEL</p>",
                unsafe_allow_html=True)

    # Build YouTube API row dynamically
    if yt_live:
        av_yt_sum = sum(
            yt_videos[YOUTUBE_VIDEO_IDS[s]]["views"]
            for s in _AV_TEASER_SLOTS
            if YOUTUBE_VIDEO_IDS.get(s) and YOUTUBE_VIDEO_IDS[s] in yt_videos
        )
        yt_row_val    = f"Av {av_yt_sum/1_000_000:.0f}M (T1–T4)"
        yt_row_status = "✓ Live"
    else:
        yt_row_val    = "—"
        yt_row_status = ("⚠ Key needed" if yt_stats.get("status") == "no_key"
                         else f"⚠ {yt_stats.get('status','unavailable')}")

    # Build trends / reddit rows only when those sources are active
    _av_t   = av_sig.get("trends_interest", 0)
    _dune_t = dune_sig.get("trends_interest", 0)
    _trends_row = ["Google Trends", "Search interest ratio",
                   f"Av {_av_t} / Dune {_dune_t}" if _trends_live else "—",
                   f"Av {cal['avengers_adj']:+.1f}pt / Dune {cal['dune_adj']:+.1f}pt",
                   "✓ Live" if _trends_live else "⚠ Not configured"]
    _reddit_row = ["Reddit API", "Post volume + upvote velocity",
                   f"r/marvelstudios {av_sig.get('reddit_hot_avg') or '—'} avg / "
                   f"r/dune {dune_sig.get('reddit_hot_avg') or '—'} avg"
                   if _reddit_live else "—",
                   "±3 pts max (score avg + post volume)",
                   "✓ Live" if _reddit_live else "⚠ Not configured"]

    rows_sig = [
        _trends_row,
        ["Teaser decay", "T1→T2 view retention",
         f"{teasers[0]:.0f}M → {teasers[1]:.0f}M ({teasers[1]/teasers[0]*100:.0f}%)"
         if len(teasers) >= 2 and teasers[0] > 0 else "—",
         f"Av {cal['avengers_adj']:+.1f}pt (decay component)",
         f"✓ {teaser_src}"],
        ["YouTube API", "Official trailer views", yt_row_val,
         "Feeds teaser decay + audience score calibration", yt_row_status],
        _reddit_row,
        ["Fandango presales", "Purchase intent", "Not open yet",
         "Opens Sept 2026", "⏳ Pending"],
    ]
    st.dataframe(
        pd.DataFrame(rows_sig, columns=["Source", "Signal", "Current Value", "Model Impact", "Status"]),
        use_container_width=True, hide_index=True,
    )

    if yt_stats.get("status") == "no_key":
        with st.expander("🔑 Set up YouTube API key (5 min)"):
            st.markdown("""
            1. Go to [console.cloud.google.com](https://console.cloud.google.com)
            2. Create project → Enable **YouTube Data API v3**
            3. Credentials → Create API Key (free, 10k units/day)
            4. In Streamlit Cloud: App Settings → Secrets → add:
            ```toml
            YOUTUBE_API_KEY = "your-key-here"
            ```
            Once added, view counts for all trailers update automatically
            on every page load and feed the audience score calibration.
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
        P, barmode="overlay", height=400,
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
    fig5.update_layout(**_layout(P, outside_text=True,
                                 height=320, yaxis_title="%", yaxis_range=[0, 55]))
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


# ── TAB 6: TRAILERS ───────────────────────────────────────────────────────────
with tab6:
    st.markdown(
        f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:16px;'>"
        "OFFICIAL TRAILERS — EMBEDDED FROM YOUTUBE</p>",
        unsafe_allow_html=True,
    )

    # ── Avengers: Doomsday ────────────────────────────────────────────────────
    st.markdown(
        f"<span style='color:{P['av']}; font-size:0.82rem; font-weight:600; "
        f"letter-spacing:2px;'>AVENGERS: DOOMSDAY</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:6px 0 14px;'>",
                unsafe_allow_html=True)

    # Row 1: T1 + T2
    a1, a2 = st.columns(2)
    with a1:
        st.caption("Teaser 1")
        st.video(YOUTUBE_TRAILER_URLS["avengers_t1"])
    with a2:
        st.caption("Teaser 2")
        st.video(YOUTUBE_TRAILER_URLS["avengers_t2"])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Row 2: T3 + T4
    a3, a4 = st.columns(2)
    with a3:
        st.caption("Teaser 3")
        st.video(YOUTUBE_TRAILER_URLS["avengers_t3"])
    with a4:
        st.caption("Teaser 4")
        st.video(YOUTUBE_TRAILER_URLS["avengers_t4"])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Countdown clock — full width
    st.caption("Countdown Clock")
    st.video(YOUTUBE_TRAILER_URLS["avengers_countdown"])

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:20px 0 14px;'>",
                unsafe_allow_html=True)

    # ── Dune: Part Three ──────────────────────────────────────────────────────
    st.markdown(
        f"<span style='color:{P['dune']}; font-size:0.82rem; font-weight:600; "
        f"letter-spacing:2px;'>DUNE: PART THREE</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:6px 0 14px;'>",
                unsafe_allow_html=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        st.caption("Teaser 1")
        st.video(YOUTUBE_TRAILER_URLS["dune_t1"])
    with d2:
        st.markdown(
            f"""
            <div style='padding:24px 0; color:{P['dim']}; font-size:0.82rem; line-height:1.8;'>
            <b style='color:{P['dune']}'>Strategic silence.</b><br><br>
            WB following the Part Two marketing cadence — first trailer
            expected closer to the Dec 18 release window.<br><br>
            Dune's low Google Trends score (13/100 vs Avengers 72/100)
            reflects zero promotional material released, not audience demand.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<p style='color:{P['footer']}; font-size:0.6rem; letter-spacing:1.5px;
   margin-top:12px; text-align:right;'>
DUNESDAY v5 &nbsp;·&nbsp; {signals['last_updated']} &nbsp;·&nbsp; {' · '.join(cal['sources'])}
</p>
""", unsafe_allow_html=True)
