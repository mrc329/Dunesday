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
from model.core import (run_all_scenarios, imax_gap_summary, SCENARIOS,
                        polymarket_scenario_weights)
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
    padding-top: 1.5rem !important;
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
    "spiderman_full":     "Spider-Man BND T1",
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
    base_dune_aud = st.slider("Dune base score", 60, 100, 87,
        help="PostTrak/CinemaScore-style audience satisfaction (0–100). "
             "87 = Dune: Part Two benchmark. Live signals adjust this up or down.")
    base_av_aud   = st.slider("Avengers base score", 60, 100, 88,
        help="88 = Infinity War / Endgame baseline. Adjusted down if teaser decay "
             "matches Love & Thunder pattern, up if engagement matches Endgame.")

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
                  delta=f"{adj:+.1f}" if adj != 0 else "no adj",
                  help=f"Live-calibrated audience score. Base: {cal['dune_base']} + "
                       f"signal adjustment: {adj:+.1f}. Drives WOM multiplier and weekly hold rates.")
    with c2:
        adj = cal["avengers_adj"]
        st.metric("Avengers score", f"{cal['avengers_calibrated']:.0f}",
                  delta=f"{adj:+.1f}" if adj != 0 else "no adj",
                  help=f"Live-calibrated audience score. Base: {cal['avengers_base']} + "
                       f"signal adjustment: {adj:+.1f}. Spider-Man tier applied on top.")

    st.caption(f"Sources: {' · '.join(cal['sources'])}")
    if cal.get("notes"):
        st.caption(f"📌 {cal['notes']}")

    st.divider()
    st.markdown("**Competitor Signals**")
    _default_spidey = (
        cal.get("spidey_suggested_tier")
        or signals.get("spiderman", {}).get("suggested_tier")
        or "Neutral"
    )
    spidey_tier = st.select_slider(
        "Spider-Man: Brand New Day (Jul 25 2026)",
        options=["Disappoints", "Soft", "Neutral", "Strong", "Blockbuster"],
        value=_default_spidey,
        help="Spider-Man opens Jul 25 — 5 months before Avengers. Its performance "
             "is an MCU brand health signal. Blockbuster = +4pts to Avengers score + 1.10x OW. "
             "Disappoints = −5pts + 0.90x OW. Auto-calibrated from trailer data; override manually.",
    )
    spidey_adj = SPIDEY_IMPACT_ADJ[spidey_tier]
    spidey_color = (P["av"] if spidey_adj < 0 else
                    P["dune"] if spidey_adj > 0 else P["dim"])
    st.caption(
        f"MCU brand signal → Avengers score "
        f"**{spidey_adj:+d} pts** · OW mult "
        f"{'↑' if spidey_adj > 0 else '↓' if spidey_adj < 0 else '—'}"
    )
    _auto_tier = cal.get("spidey_suggested_tier")
    if _auto_tier and _auto_tier != spidey_tier:
        st.caption(f"Trailer signal suggests: **{_auto_tier}** ↑ update slider")

    st.divider()
    override = st.toggle("Manual score override", value=False,
        help="When off, audience scores come from live signal calibration (YouTube + Wikipedia + "
             "TMDB + Trakt + Polymarket). Turn on to set them manually for scenario exploration.")
    if override:
        dune_aud = st.slider("Dune (manual)", 60, 100, int(cal["dune_calibrated"]),
            help="Override Dune audience score. 87 = Part Two actual. "
                 "Below 80 = mixed reception; above 92 = exceptional (Part One/Two territory).")
        av_aud   = st.slider("Avengers (manual)", 60, 100, int(cal["avengers_calibrated"]),
            help="Override Avengers audience score. 88 = IW/Endgame baseline. "
                 "Below 80 = Love & Thunder territory (76). Above 93 = Deadpool & Wolverine territory (95).")
        st.caption("⚠️ Overriding live calibration")
    else:
        dune_aud = int(cal["dune_calibrated"])
        # Apply Spider-Man MCU brand signal on top of live calibration
        av_aud   = int(np.clip(cal["avengers_calibrated"] + spidey_adj, 60, 100))

    st.divider()
    st.markdown("**International Multipliers**")
    dune_intl = st.slider("Dune intl mult", 0.8, 2.5, 1.48, 0.05,
        help="Ratio of international gross to domestic gross. 1.48 = Dune: Part Two actual. "
             "Lower end (~0.9) = limited international appeal; upper end (~2.2) = global phenomenon.")
    av_intl   = st.slider("Avengers intl mult", 1.0, 3.5, 2.18, 0.05,
        help="2.18 = MCU historical average. Endgame was 2.1×, IW was 2.2×. "
             "China market uncertainty adds downside risk — slide down to 1.6× for China-out scenario.")

    st.markdown("**Simulation**")
    n_trials = st.select_slider("MC trials", [500, 1000, 2000, 5000], value=1000,
        help="Number of Monte Carlo simulation trials. 1,000 is fast and sufficient for P50. "
             "5,000 narrows the P10/P90 confidence interval by ~2× but takes ~5× longer to run.")

    st.divider()
    st.markdown("**IMAX Config (Locked)**")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:{P['dim']}; line-height:1.9;'>
    Dune exclusive: <b style='color:{P['dune']}'>Days 1–21</b><br>
    Dune screens: <b style='color:{P['dune']}'>400</b><br>
    Avengers day 1: <b style='color:{P['av']}'>0 screens</b><br>
    Avengers first IMAX: <b style='color:{P['av']}'>Jan 8</b>
    </div>
    """.strip(), unsafe_allow_html=True)


# ── RUN MODEL ─────────────────────────────────────────────────────────────────
_poly_sig       = signals.get("polymarket", {})
_poly_ow_odds   = _poly_sig.get("avengers_ow_odds")
_poly_fy_odds   = _poly_sig.get("avengers_full_year_odds")
_poly_ratio     = _poly_sig.get("ow_decay_ratio")
_poly_move_sig  = _poly_sig.get("move_signal")
_poly_source    = _poly_sig.get("source", "fallback")
_poly_weights   = polymarket_scenario_weights(_poly_ratio)

with st.spinner("Running Monte Carlo..."):
    results = run_all_scenarios(
        n=n_trials,
        dune_aud=dune_aud, av_aud=av_aud,
        dune_intl=dune_intl, av_intl=av_intl,
        spidey_tier=spidey_tier,
        polymarket_ow_odds=_poly_ow_odds,
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
    st.metric("Dune P50 Profit", f"${sc_a['DUNE']['p50']:.0f}M", delta="100% break-even",
              help="Median net profit for Dune: Part Three in Scenario A (both films hold Dec 18). "
                   "P50 = 50th percentile of 5,000 MC trials. Budget: $175M + ~$88M P&A = $263M all-in.")
with k2:
    st.metric("Avengers P50 Profit", f"${sc_a['AVENGERS']['p50']:.0f}M",
              delta=f"{sc_a['AVENGERS']['breakeven_pct']:.0f}% BE",
              help="Median net profit for Avengers: Doomsday with zero IMAX screens for 21 days. "
                   f"Break-even probability: {sc_a['AVENGERS']['breakeven_pct']:.0f}%. "
                   "Budget: $550M + ~$275M P&A = $825M all-in.")
with k3:
    st.metric("IMAX Gap", f"${imax['gap']:.1f}M", delta="Dune advantage",
              help="Total 45-day IMAX revenue difference: Dune minus Avengers. "
                   "Dune gets all 400 US IMAX screens for 21 days. "
                   "Avengers gets zero IMAX until Jan 8, then splits 200 screens.")
with k4:
    st.metric("Dune Xmas IMAX", f"${imax['xmas_day_dune']:.2f}M",
              help="Dune's estimated Christmas Day (Dec 25) IMAX revenue. "
                   "Day 7 of exclusive window — all 400 screens, peak holiday multiplier (1.65×).")
with k5:
    st.metric("Avengers Xmas IMAX", "$0.00M", delta="Zero screens", delta_color="inverse",
              help="Avengers earns zero IMAX revenue on Christmas Day. "
                   "Dune's 21-day exclusive window runs Dec 18 – Jan 7, covering the entire holiday premium.")
with k6:
    st.metric("Avengers Locked Out", "21 days", delta="Dec 18 – Jan 7", delta_color="off",
              help="Avengers is shut out of all 400 US IMAX screens for its first 21 days. "
                   "This is the confirmed exclusive window Dune negotiated. "
                   "Avengers' first IMAX show is Jan 8, splitting 200 screens with Dune.")

st.divider()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab7, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "MODEL", "SCENARIOS", "IMAX TIMELINE", "LIVE SIGNALS", "DISTRIBUTIONS", "DISNEY DECISION", "TRAILERS",
])


# ── TAB 1: SCENARIOS ──────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>NET PROFIT BY SCENARIO — P10 / P50 / P90</p>",
                unsafe_allow_html=True)
    st.caption("Bar = P50 median outcome. Whiskers show P10 (downside, 10th percentile) and P90 (upside, 90th percentile) from 5,000 Monte Carlo trials. Convention follows standard financial analysis: P10 is pessimistic, P90 is optimistic.")

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
        customdata=list(zip(dune_p10s, dune_p90s,
                            [results[sk]["DUNE"]["breakeven_pct"] for sk in sk_list])),
        hovertemplate=(
            "<b>Dune — %{x}</b><br>"
            "P50 (median): <b>$%{y:.0f}M</b><br>"
            "P10 (downside): $%{customdata[0]:.0f}M<br>"
            "P90 (upside): $%{customdata[1]:.0f}M<br>"
            "Break-even: %{customdata[2]:.0f}%"
            "<extra></extra>"
        ),
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
        customdata=list(zip(av_p10s, av_p90s,
                            [results[sk]["AVENGERS"]["breakeven_pct"] for sk in sk_list])),
        hovertemplate=(
            "<b>Avengers — %{x}</b><br>"
            "P50 (median): <b>$%{y:.0f}M</b><br>"
            "P10 (downside): $%{customdata[0]:.0f}M<br>"
            "P90 (upside): $%{customdata[1]:.0f}M<br>"
            "Break-even: %{customdata[2]:.0f}%"
            "<extra></extra>"
        ),
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
    fig2.add_trace(go.Bar(x=date_labels, y=dune_screens, name="Dune",
                          marker_color=P["dune"],
                          hovertemplate="%{x}<br>Dune IMAX screens: <b>%{y}</b><extra></extra>"),
                  row=1, col=1)
    fig2.add_trace(go.Bar(x=date_labels, y=av_screens, name="Avengers",
                          marker_color=P["av"],
                          hovertemplate="%{x}<br>Avengers IMAX screens: <b>%{y}</b><extra></extra>"),
                  row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["dune_daily"],
        name="Dune IMAX rev",
        line=dict(color=P["dune"], width=2),
        fill="tozeroy", fillcolor=P["fill_dune"],
        hovertemplate="%{x}<br>Dune IMAX revenue: <b>$%{y:.2f}M</b><extra></extra>",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=date_labels, y=imax["avengers_daily"],
        name="Avengers IMAX rev",
        line=dict(color=P["av"], width=2),
        fill="tozeroy", fillcolor=P["fill_av"],
        hovertemplate="%{x}<br>Avengers IMAX revenue: <b>$%{y:.2f}M</b><extra></extra>",
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
                                 margin=dict(t=44, b=40, l=52, r=16),
                                 hovermode="x unified"))
    fig2.update_xaxes(tickangle=45, nticks=15, showgrid=False,
                      showline=True, linecolor=P["axis"], linewidth=0.5,
                      showspikes=True, spikemode="across",
                      spikesnap="cursor", spikecolor=P["vline_ref"],
                      spikethickness=1, spikedash="dot")
    fig2.update_yaxes(showgrid=False, showline=False, zeroline=False)
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dune excl window", f"${imax['dune_excl_rev']:.1f}M", "Days 1–21",
              help="Dune's IMAX revenue during the 21-day exclusive window (Dec 18 – Jan 7). "
                   "All 400 US IMAX screens, peak holiday calendar, no competition.")
    c2.metric("Avengers excl window", "$0.0M", "Zero screens", delta_color="inverse",
              help="Avengers earns exactly $0 from IMAX during Dec 18 – Jan 7. "
                   "This is the direct financial cost of the date conflict.")
    c3.metric("Dune 45-day IMAX", f"${imax['dune_total']:.1f}M",
              help="Dune's total IMAX revenue over 45 days: 21-day exclusive (400 screens) "
                   "+ 24-day split (200 screens). Includes daily calendar and decay multipliers.")
    c4.metric("Avengers 45-day IMAX", f"${imax['avengers_total']:.1f}M",
              delta=f"-${imax['gap']:.1f}M vs Dune", delta_color="inverse",
              help="Avengers' total IMAX revenue over 45 days, starting Jan 8 with only 200 split screens. "
                   f"The ${imax['gap']:.1f}M gap is pure lost revenue from the exclusive window.")


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
    """.strip(), unsafe_allow_html=True)

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

    # Avengers teaser views — prefer YouTube live, fall back to X/Twitter per slot
    av_sig   = signals["avengers"]
    dune_sig = signals["dune"]
    yt_av_teasers = [_yt_views_M(s) for s in _AV_TEASER_SLOTS]
    use_yt_teasers = any(v is not None for v in yt_av_teasers)
    x_fallbacks = av_sig.get("teaser_views_x_M", [])

    if use_yt_teasers:
        # For each slot, use YouTube if available, otherwise fall back to X/Twitter estimate
        merged = [
            yt if yt is not None else (x_fallbacks[i] if i < len(x_fallbacks) else None)
            for i, yt in enumerate(yt_av_teasers)
        ]
        teasers    = [v for v in merged if v is not None]
        teaser_src = "YouTube"
    else:
        merged     = [None] * len(_AV_TEASER_SLOTS)
        teasers    = x_fallbacks
        teaser_src = "X/Twitter"

    col_a, col_b = st.columns(2)

    with col_a:
        _av_color = P["av"]
        st.markdown(f"<div style='color:{_av_color}; font-size:0.9rem; font-weight:700; letter-spacing:2px; margin-bottom:10px; padding-bottom:4px; border-bottom:1px solid {_av_color};'>AVENGERS: DOOMSDAY</div>",
                    unsafe_allow_html=True)

        if teasers:
            if use_yt_teasers:
                # Show all 4 slots; YouTube data where available, X/Twitter fallback for the rest
                labels_decay = [_YT_SLOT_LABELS[s].replace("Avengers ", "")
                                for s in _AV_TEASER_SLOTS]
                vals_decay   = [v if v is not None else 0.0 for v in merged]
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
                hovertemplate="%{x}: <b>%{y:.0f}M views</b><br>"
                              "T1→this ratio: %{customdata:.0%}"
                              "<extra></extra>",
                customdata=[v / vals_decay[0] if vals_decay[0] > 0 else 0 for v in vals_decay],
            ))
            fig_d.add_trace(go.Scatter(
                x=labels_decay, y=vals_decay,
                line=dict(color=P["av"], dash="dot", width=1),
                mode="lines", showlegend=False,
                hoverinfo="skip",
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

        # YouTube total views across all 4 teasers
        if yt_live:
            av_yt_total = sum(
                yt_videos[YOUTUBE_VIDEO_IDS[s]]["views"]
                for s in _AV_TEASER_SLOTS
                if YOUTUBE_VIDEO_IDS.get(s) and YOUTUBE_VIDEO_IDS[s] in yt_videos
            )
            av_yt_likes = sum(
                yt_videos[YOUTUBE_VIDEO_IDS[s]].get("likes", 0)
                for s in _AV_TEASER_SLOTS
                if YOUTUBE_VIDEO_IDS.get(s) and YOUTUBE_VIDEO_IDS[s] in yt_videos
            )
            _av_eng_ratio = av_yt_likes / av_yt_total if av_yt_total else None
            st.metric("YT teaser views",
                           f"{av_yt_total / 1_000_000:.0f}M" if av_yt_total else "—",
                           delta=f"T1–T4 combined · {yt_fetched[:10]}",
                           help="Combined YouTube view count across all 4 Avengers teasers. "
                                "Benchmarks: Endgame 289M T1, IW 230M T1, D&W 365M T1. "
                                "≥300M combined → +4pts score. <100M → −5pts.")
            if _av_eng_ratio is not None:
                _av_color = P["av"]
                _av_dim   = P["dim"]
                st.markdown(
                    f"<div style='border-left:2px solid {_av_color}; padding:8px 14px; "
                    f"font-size:0.82rem; color:{_av_dim};'>"
                    f"T1–T4 combined · <b style='color:{_av_color}'>{av_yt_likes:,}</b> likes · "
                    f"like/view ratio <b style='color:{_av_color}'>{_av_eng_ratio * 100:.2f}%</b>"
                    f" — YouTube audience (primary platform: X)</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.metric("YT trailer views",
                      f"{av_sig['yt_trailer_views']:,}" if av_sig.get("yt_trailer_views") else "—",
                      delta="Full trailer not released" if not av_sig.get("full_trailer_out") else "Live",
                      help="Full trailer 24h YouTube view count. Add YOUTUBE_API_KEY to Streamlit secrets "
                           "to enable live data. Benchmarks: Endgame 289M, IW 230M, D&W 365M.")

        # Wikipedia pageviews
        _av_wiki_7d  = av_sig.get("wiki_views_7d")
        _av_wiki_wow = av_sig.get("wiki_wow_pct")
        w1, w2 = st.columns(2)
        w1.metric("Wikipedia views (7d)",
                  f"{_av_wiki_7d:,}" if _av_wiki_7d else "—",
                  delta="Research interest · no API key",
                  help="Wikipedia pageviews for 'Avengers: Doomsday' over the last 7 days. "
                       "Measures active research interest. No API key needed — free Wikimedia REST API. "
                       "≥14,000/week = strong (+2pts). <3,000/week = soft (−1pt).")
        w2.metric("Week-over-week",
                  f"{_av_wiki_wow:+.0f}%" if _av_wiki_wow is not None else "—",
                  delta="↑ growing" if (_av_wiki_wow or 0) > 10 else "↓ cooling" if (_av_wiki_wow or 0) < -10 else "stable",
                  help="Change in Wikipedia pageviews vs the prior 7-day period. "
                       "Momentum signal: +50%+ = surging (+1pt). −30%+ drop = fading (−1pt).")

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
            """.strip(), unsafe_allow_html=True)

    with col_b:
        _dune_color = P["dune"]
        st.markdown(f"<div style='color:{_dune_color}; font-size:0.9rem; font-weight:700; letter-spacing:2px; margin-bottom:10px; padding-bottom:4px; border-bottom:1px solid {_dune_color};'>DUNE: PART THREE</div>",
                    unsafe_allow_html=True)

        dune_yt_views = _yt_views_M("dune_t1")
        dune_color    = P["dune"]
        dune_dim      = P["dim"]
        _dune_t1_fresh   = cal.get("dune_t1_fresh", False)
        _dune_eng_ratio  = dune_sig.get("yt_engagement_ratio")
        _dune_likes      = dune_sig.get("yt_trailer_likes")
        if dune_yt_views is not None and _dune_t1_fresh:
            _dune_ratio_pct = f"{_dune_eng_ratio * 100:.1f}%" if _dune_eng_ratio else "—"
            _dune_likes_str = f" · {_dune_likes:,} likes" if _dune_likes else ""
            st.markdown(
                f"<div style='border-left:2px solid {dune_color}; padding:8px 14px; "
                f"font-size:0.82rem; color:{dune_dim};'>"
                f"Trailer released today · <b style='color:{dune_color}'>{dune_yt_views:.0f}M</b> views"
                f"{_dune_likes_str} · like/view ratio "
                f"<b style='color:{dune_color}'>{_dune_ratio_pct}</b>"
                f" — calibrated from engagement (view-count benchmarks need 24h)</div>",
                unsafe_allow_html=True,
            )
        elif dune_yt_views is not None:
            _dune_ratio_pct = f"{_dune_eng_ratio * 100:.2f}%" if _dune_eng_ratio else "—"
            _dune_likes_str = f" · {_dune_likes:,} likes" if _dune_likes else ""
            _ratio_str = (
                f" · like/view ratio <b style='color:{dune_color}'>{_dune_ratio_pct}</b>"
                if _dune_eng_ratio else ""
            )
            st.markdown(
                f"<div style='border-left:2px solid {dune_color}; padding:8px 14px; "
                f"font-size:0.82rem; color:{dune_dim};'>"
                f"Trailer live · <b style='color:{dune_color}'>{dune_yt_views:.0f}M</b> views"
                f"{_dune_likes_str}{_ratio_str}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No trailer released. WB following Part Two marketing cadence — strategic delay.")

        _dune_m1, _dune_m2 = st.columns(2)
        _dune_m2.metric("Alamo poll", "#1 Most Anticipated", delta="14,000 respondents",
                        help="Alamo Drafthouse most-anticipated films survey. Dune ranked #1 among "
                             "14,000 respondents — a strong signal of art-house and cinephile demand, "
                             "the core audience that drives Dune's unusually high audience scores.")

        # Wikipedia pageviews
        _dune_wiki_7d  = dune_sig.get("wiki_views_7d")
        _dune_wiki_wow = dune_sig.get("wiki_wow_pct")
        _dune_m1.metric("Wikipedia views (7d)",
                        f"{_dune_wiki_7d:,}" if _dune_wiki_7d else "—",
                        delta="Research interest",
                        help="Wikipedia pageviews for 'Dune: Part Three' over the last 7 days. "
                             "Dune has a smaller but highly engaged fanbase — lower absolute thresholds apply. "
                             "≥7,000/week = strong (+2pts). <1,500/week = soft (−1pt).")
        w3, w4 = st.columns(2)
        w3.metric("Week-over-week",
                  f"{_dune_wiki_wow:+.0f}%" if _dune_wiki_wow is not None else "—",
                  delta="↑ growing" if (_dune_wiki_wow or 0) > 10 else "↓ cooling" if (_dune_wiki_wow or 0) < -10 else "stable",
                  help="Change in Dune: Part Three Wikipedia pageviews vs prior 7-day period. "
                       "Momentum signal: sustained growth = organic interest compounding. "
                       "Drop = marketing needs to re-engage the fanbase.")

        # Wikipedia comparison chart
        if _av_wiki_7d and _dune_wiki_7d:
            fig_wiki = go.Figure()
            fig_wiki.add_trace(go.Bar(
                x=["Wikipedia Research Interest (7d)"],
                y=[_av_wiki_7d],
                name="Avengers", marker_color=P["av"],
                text=[f"Av {_av_wiki_7d:,}"],
                textposition="inside", textfont=dict(size=10, color="white"),
                hovertemplate="Avengers Wikipedia (7d): <b>%{y:,} views</b>"
                              "<extra></extra>",
            ))
            fig_wiki.add_trace(go.Bar(
                x=["Wikipedia Research Interest (7d)"],
                y=[_dune_wiki_7d],
                name="Dune", marker_color=P["dune"],
                text=[f"Dune {_dune_wiki_7d:,}"],
                textposition="inside", textfont=dict(size=10, color=P["bg"]),
                hovertemplate="Dune Wikipedia (7d): <b>%{y:,} views</b>"
                              "<extra></extra>",
            ))
            fig_wiki.update_layout(**_layout(P, barmode="group", height=240, yaxis_title="Pageviews"))
            st.plotly_chart(fig_wiki, use_container_width=True)
            st.caption("Wikipedia pageviews = active research interest. No API key needed. Updates daily.")

    # ── YouTube per-video stats table ─────────────────────────────────────────
    if yt_live:
        st.divider()
        st.markdown(f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>YOUTUBE TRAILER VIEWS — LIVE DATA</p>",
                    unsafe_allow_html=True)

        yt_rows = []
        for slot in list(_AV_TEASER_SLOTS) + ["avengers_countdown", "dune_t1", "spiderman_full"]:
            vid_id = YOUTUBE_VIDEO_IDS.get(slot)
            if not vid_id:
                continue
            label = _YT_SLOT_LABELS.get(slot, slot)
            if vid_id in yt_videos:
                vd = yt_videos[vid_id]
                _ratio = vd["likes"] / vd["views"] if vd.get("views") else None
                yt_rows.append({
                    "Video":        label,
                    "Title":        vd["title"][:60] + ("…" if len(vd["title"]) > 60 else ""),
                    "Views":        f"{vd['views'] / 1_000_000:.1f}M",
                    "Likes":        f"{vd['likes'] / 1_000:.0f}K",
                    "Like/View %":  f"{_ratio * 100:.2f}%" if _ratio else "—",
                })
            else:
                yt_rows.append({"Video": label, "Title": "—", "Views": "—", "Likes": "—", "Like/View %": "—"})

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
            line=dict(dash=sty["dash"], color=sty["color"], width=1.2),
            opacity=sty["opacity"],
            hovertemplate=f"<b>{bname}</b><br>%{{x}}: %{{y:.0f}}% of OW<extra></extra>",
        ))

    # Projected curve — prominent
    fig_decay.add_trace(go.Scatter(
        x=wk_labels, y=proj_pct,
        name=f"Projected (score {av_aud})",
        mode="lines+markers",
        line=dict(color=P["av"], width=2.5),
        marker=dict(size=6, color=P["av"]),
        hovertemplate=(
            f"<b>Projected (audience {av_aud})</b><br>"
            "%{x}: <b>%{y:.0f}%</b> of OW gross<br>"
            f"WOM multiplier: {wm_proj:.2f}×"
            "<extra></extra>"
        ),
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

    # ── Spider-Man: Brand New Day trailer signal ──────────────────────────────
    st.divider()
    st.markdown(
        f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']};'>"
        "SPIDER-MAN: BRAND NEW DAY — MCU BRAND SIGNAL</p>",
        unsafe_allow_html=True,
    )
    _spidey_sig_tab     = signals.get("spiderman", {})
    _spidey_views_M_tab = _spidey_sig_tab.get("yt_trailer_views_M")
    _spidey_likes_tab   = _spidey_sig_tab.get("yt_trailer_likes")
    _spidey_ratio_tab   = _spidey_sig_tab.get("yt_engagement_ratio")
    _spidey_tier_tab    = cal.get("spidey_suggested_tier") or _spidey_sig_tab.get("suggested_tier")
    _spidey_fresh       = cal.get("spidey_trailer_fresh", False)
    _spidey_color = P["av"] if _spidey_tier_tab in ("Disappoints", "Soft") else \
                    P["dune"] if _spidey_tier_tab in ("Strong", "Blockbuster") else P["mid_ref"]

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _sc1.metric(
        "Trailer status",
        "Released" if _spidey_sig_tab.get("full_trailer_released") else "Not released",
        delta=_spidey_sig_tab.get("trailer_date", ""),
        help="Spider-Man: Brand New Day trailer release status. Once released, "
             "YouTube view count and like/view ratio are pulled automatically "
             "to suggest an impact tier for the sidebar slider.",
    )
    _sc2.metric(
        "YouTube views",
        f"{_spidey_views_M_tab:.0f}M" if _spidey_views_M_tab else "—",
        delta="Accumulating" if _spidey_fresh and _spidey_views_M_tab else
              "Set spiderman_full video ID" if not _spidey_views_M_tab else "Live",
        help="24-hour YouTube view count for the Spider-Man: BND trailer. "
             "Benchmarks: NWH T1 355M (Blockbuster), FFH T1 135M (Neutral), "
             "Homecoming T1 64M (Soft). Set YOUTUBE_VIDEO_IDS['spiderman_full'] to enable.",
    )
    _sc3.metric(
        "Like/view ratio",
        f"{_spidey_ratio_tab * 100:.1f}%" if _spidey_ratio_tab else "—",
        delta=f"{_spidey_likes_tab:,} likes" if _spidey_likes_tab else
              "Day-1 signal — time-independent",
        help="Like/view ratio is time-independent and works from minute one — "
             "unlike raw view counts which require 24h to compare. "
             "≥4.5% = Blockbuster, ≥3.3% = Strong, ≥2.2% = Neutral, ≥1.3% = Soft.",
    )
    _sc4.metric(
        "Suggested impact tier",
        _spidey_tier_tab or "—",
        delta="Via engagement ratio" if _spidey_fresh and _spidey_tier_tab else
              "Via 24h view count" if _spidey_tier_tab else "MCU brand signal → Avengers score",
        help="Auto-suggested tier for the sidebar Spider-Man slider based on trailer data. "
             "Each tier maps to an Avengers audience score adjustment and OW gross multiplier. "
             "Override manually in the sidebar if you disagree with the auto-suggestion.",
    )
    if _spidey_tier_tab:
        _ratio_pct = f"{_spidey_ratio_tab * 100:.1f}%" if _spidey_ratio_tab else "—"
        _method    = "DAY-1 ENGAGEMENT RATIO" if _spidey_fresh else "AUTO-CALIBRATION"
        _detail    = (
            f"Like/view ratio <b style='color:{_spidey_color}'>{_ratio_pct}</b> → "
            f"<b style='color:{_spidey_color}'>{_spidey_tier_tab}</b> tier. "
            "View-count benchmarks need 24 hours — ratio compares trailers released months apart."
            if _spidey_fresh else
            f"24h view count suggests <b style='color:{_spidey_color}'>{_spidey_tier_tab}</b>"
            " tier — adjust the sidebar slider to override."
        )
        st.markdown(f"""
        <div style='border-left:2px solid {_spidey_color}; padding:7px 12px; margin-top:6px;'>
          <span style='color:{_spidey_color}; font-size:0.6rem; letter-spacing:2px'>
            {_method}: {_spidey_tier_tab.upper()}
          </span><br>
          <span style='color:{P["dim"]}; font-size:0.76rem'>{_detail}</span>
        </div>
        """.strip(), unsafe_allow_html=True)
    else:
        st.info(
            "Spider-Man: Brand New Day trailer released 2026-03-18 across Sony Pictures channels. "
            "Set `YOUTUBE_VIDEO_IDS['spiderman_full']` to the video ID to enable auto-calibration.",
            icon="🕷",
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

    # Build Wikipedia row
    _wiki_live = "Wikipedia" in cal.get("sources", [])
    _wiki_av   = av_sig.get("wiki_views_7d")
    _wiki_dune = dune_sig.get("wiki_views_7d")
    _wiki_row  = [
        "Wikipedia pageviews", "Research interest (7d views)",
        f"Av {_wiki_av:,} / Dune {_wiki_dune:,}" if (_wiki_av and _wiki_dune) else "—",
        "±3 pts (volume + WoW momentum)",
        "✓ Live" if _wiki_live else "⚠ Unavailable",
    ]

    # Spider-Man trailer signal row
    _spidey_sig     = signals.get("spiderman", {})
    _spidey_views_M = _spidey_sig.get("yt_trailer_views_M")
    _spidey_tier    = cal.get("spidey_suggested_tier") or _spidey_sig.get("suggested_tier")
    _spidey_row = [
        "Spider-Man: BND Trailer",
        "MCU brand health signal",
        f"{_spidey_views_M:.0f}M YT views" if _spidey_views_M else "Released 2026-03-18 — set video ID",
        f"Suggests '{_spidey_tier}' tier → Av score adj"
        if _spidey_tier else "Pending view count data",
        "✓ Live" if _spidey_views_M else "⚠ Video ID needed",
    ]

    rows_sig = [
        ["Teaser decay", "T1→T2 view retention",
         f"{teasers[0]:.0f}M → {teasers[1]:.0f}M ({teasers[1]/teasers[0]*100:.0f}%)"
         if len(teasers) >= 2 and teasers[0] > 0 else "—",
         f"Av {cal['avengers_adj']:+.1f}pt (decay component)",
         f"✓ {teaser_src}"],
        ["YouTube API", "Official trailer views", yt_row_val,
         "Feeds teaser decay + audience score calibration", yt_row_status],
        _wiki_row,
        _spidey_row,
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
            hovertemplate=f"<b>{name}</b><br>Net profit range: $%{{x:.0f}}M<br>Density: %{{y:.4f}}<extra></extra>",
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
              delta=f"{results['A_Both_Hold']['AVENGERS']['breakeven_pct']:.0f}% BE",
              help="Avengers median net profit if Disney holds Dec 18 and absorbs the 21-day IMAX lockout. "
                   "P50 of 5,000 MC trials. Avengers gets zero IMAX revenue for its entire opening weekend.")
    c2.metric("Move to May P50", f"${sc_b_av:.0f}M",
              delta=f"+${sc_b_av - sc_a_av:.0f}M vs holding",
              help="Avengers median net profit if Disney moves to May 1 (uncontested summer). "
                   "Gets full 400-screen IMAX exclusive. Loses Christmas premium but gains legs. "
                   f"Uplift vs holding: +${sc_b_av - sc_a_av:.0f}M at P50.")
    c3.metric("Move to Jan P50", f"${sc_c_av:.0f}M",
              delta=f"+${sc_c_av - sc_a_av:.0f}M vs holding",
              delta_color="normal" if sc_c_av > sc_a_av else "inverse",
              help="Avengers median net profit if Disney moves to Jan 16 (after Dune's exclusive expires). "
                   "Gets ~200 IMAX screens from day 1 but misses Christmas. Partial compromise scenario.")

    # ── Polymarket integration explainer ──────────────────────────────────────
    st.divider()
    st.markdown(
        f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:10px;'>"
        "POLYMARKET SIGNAL</p>",
        unsafe_allow_html=True,
    )

    _pm_src_label = "LIVE" if _poly_source == "live" else "FALLBACK (2026-03-18)"
    _pm_src_color = P["dune"] if _poly_source == "live" else P["dim"]
    _rec_color = {
        "move":      P["av"],
        "lean_move": "#d48020",
        "neutral":   P["mid_ref"],
        "hold":      P["dune"],
    }.get(_poly_weights["recommendation"], P["mid_ref"])

    pm_c1, pm_c2, pm_c3, pm_c4 = st.columns(4)
    pm_c1.metric(
        "Avengers Best OW",
        f"{_poly_ow_odds:.0%}" if _poly_ow_odds else "—",
        delta=(f"OW scalar {(1.05 if _poly_ow_odds >= 0.70 else 1.00 if _poly_ow_odds >= 0.50 else 0.90 if _poly_ow_odds >= 0.30 else 0.80):.2f}x in MC" if _poly_ow_odds else "OW scalar N/A"),
        help="Polymarket: probability Avengers has the best domestic opening weekend of 2026. "
             "This is a direct crowd signal on opening-weekend demand. "
             "Maps to an OW gross multiplier in the Monte Carlo: ≥70% → 1.05×, <30% → 0.80×.",
    )
    pm_c2.metric(
        "Avengers Best Full Year",
        f"{_poly_fy_odds:.0%}" if _poly_fy_odds else "—",
        delta="Calendar-year domestic",
        help="Polymarket: probability Avengers is the highest-grossing film of 2026 (calendar year). "
             "Note: measures Jan 1 – Dec 31 domestic only. Avengers opens Dec 18 — "
             "Spider-Man opens Jul 25 and has 5× more calendar-year accumulation time. "
             "The low odds partly reflect timing, not just legs quality.",
    )
    pm_c3.metric(
        "OW / Full-Year Ratio",
        f"{_poly_ratio:.1f}x" if _poly_ratio else "—",
        delta="Legs collapse signal",
        delta_color="inverse" if (_poly_ratio or 0) >= 3.0 else "off",
        help="OW odds ÷ full-year odds. A high ratio means the crowd thinks Avengers opens "
             "huge but underperforms over its full run — the IMAX legs collapse in one number. "
             "≥3.0× → STRONG move signal. 2.0–3.0× → moderate concern. <1.3× → market comfortable.",
    )
    pm_c4.metric(
        "Move Signal",
        (_poly_weights["recommendation"].replace("_", " ").upper()),
        delta=_pm_src_label,
        help="Disney move recommendation derived from the OW/FY ratio. "
             "'MOVE' = ratio ≥3.0×, market strongly prices in a legs collapse. "
             "'LEAN MOVE' = 2.0–3.0×. 'NEUTRAL' = 1.3–2.0×. 'HOLD' = <1.3×. "
             f"Current: {_poly_weights['label']}",
    )

    # Weighted expected P50
    _weighted_p50 = sum(
        _poly_weights["weights"].get(sk, 0) * results[sk]["AVENGERS"]["p50"]
        for sk in results
    )
    _sc_a_p50 = results["A_Both_Hold"]["AVENGERS"]["p50"]
    _sc_b_p50 = results["B_Disney_May"]["AVENGERS"]["p50"]

    with st.expander("How Polymarket feeds into this model", expanded=False):
        st.markdown(f"""
<div style='font-size:0.82rem; line-height:1.85; color:{P["text"]};'>

**What Polymarket is**

Polymarket is a prediction market where traders put real money on outcomes.
Prices are probabilities set by the crowd — not polls, not analysts.
With $1.1M+ traded on these two markets, the signal carries meaningful weight.

**The two markets**

| Market | Avengers odds | What it measures |
|---|---|---|
| Best opening weekend in 2026 | **{f"{_poly_ow_odds:.0%}" if _poly_ow_odds is not None else "—"}** | Pure opening-weekend demand |
| Highest full-year gross in 2026 | **{f"{_poly_fy_odds:.0%}" if _poly_fy_odds is not None else "—"}** | Full domestic run (calendar year) |

**Why the gap matters**

The OW/FY ratio is **{f"{_poly_ratio:.1f}x" if _poly_ratio is not None else "—"}**. Avengers is the heavy favorite to open
biggest, but the crowd gives it only {f"{_poly_fy_odds:.0%}" if _poly_fy_odds is not None else "—"} to dominate the full year.
That gap is the market pricing in the IMAX conflict: Avengers loses 400 IMAX
screens for 21 days, while Dune banks Christmas on all of them.

**The calendar caveat**

The full-year market measures *2026 calendar-year domestic gross*. Avengers
opens Dec 18 — it only has ~2 weeks of 2026 to accumulate. Spider-Man opens
July 25 and has ~5 months. So part of the ratio is just the opening-date math,
not a pure legs signal. The ratio overstates the legs problem slightly, but the
*direction* is correct.

**How it enters the Monte Carlo**

*Option A — OW scalar on opening weekend gross:*
The OW odds ({f"{_poly_ow_odds:.0%}" if _poly_ow_odds is not None else "—"}) map to a **{f"{(1.05 if _poly_ow_odds >= 0.70 else 1.00 if _poly_ow_odds >= 0.50 else 0.90 if _poly_ow_odds >= 0.30 else 0.80):.2f}x" if _poly_ow_odds is not None else "—"}** applied to Avengers' mean
opening-weekend gross in every trial. At 75% the crowd confirms a blockbuster
opening — the model gets a +5% OW nudge. If odds fell to 40%, the model would
apply a −10% OW penalty. This is the crowd acting as a real-money sentiment
check on the $240M base assumption.

*Option B — Scenario weights:*
The OW/FY ratio ({f"{_poly_ratio:.1f}x" if _poly_ratio is not None else "—"}) maps to how much financial merit each scenario
has, weighted by what the market is implicitly pricing. At {f"{_poly_ratio:.1f}x" if _poly_ratio is not None else "—"}:

| Scenario | Weight | Avengers P50 |
|---|---|---|
| A: Both Hold | {_poly_weights["weights"]["A_Both_Hold"]:.0%} | ${results["A_Both_Hold"]["AVENGERS"]["p50"]:.0f}M |
| B: Disney → May | {_poly_weights["weights"]["B_Disney_May"]:.0%} | ${results["B_Disney_May"]["AVENGERS"]["p50"]:.0f}M |
| C: Disney → Jan | {_poly_weights["weights"]["C_Disney_Jan"]:.0%} | ${results["C_Disney_Jan"]["AVENGERS"]["p50"]:.0f}M |

**Polymarket-weighted expected P50: ${_weighted_p50:.0f}M** vs ${_sc_a_p50:.0f}M if Disney holds.
That's a **${_weighted_p50 - _sc_a_p50:+.0f}M** signal in favor of moving.

</div>
""", unsafe_allow_html=True)

    # Scenario weights bar
    _weight_df = pd.DataFrame({
        "Scenario": [SCENARIOS[sk]["label"] for sk in ["A_Both_Hold", "B_Disney_May", "C_Disney_Jan"]],
        "Weight":   [round(_poly_weights["weights"][sk] * 100) for sk in ["A_Both_Hold", "B_Disney_May", "C_Disney_Jan"]],
        "Avengers P50 ($M)": [round(results[sk]["AVENGERS"]["p50"]) for sk in ["A_Both_Hold", "B_Disney_May", "C_Disney_Jan"]],
    })
    _wt_colors = [P["av"], P["dune"], P["cyan"]]
    fig_pm = go.Figure(go.Bar(
        x=_weight_df["Scenario"],
        y=_weight_df["Weight"],
        marker_color=_wt_colors,
        text=[f'{w}%' for w in _weight_df["Weight"]],
        textposition="outside",
        textfont=dict(size=11, color=P["chart_font"]),
        customdata=_weight_df["Avengers P50 ($M)"],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Polymarket weight: <b>%{y}%</b><br>"
            "Avengers P50: <b>$%{customdata}M</b><br>"
            "Contribution to weighted P50: $%{customdata:.0f}M × %{y}%"
            "<extra></extra>"
        ),
    ))
    fig_pm.add_annotation(
        text=f"Polymarket-weighted P50: <b>${_weighted_p50:.0f}M</b>",
        xref="paper", yref="paper", x=0.5, y=1.08,
        showarrow=False, font=dict(size=11, color=P["text"]),
        align="center",
    )
    fig_pm.update_layout(**_layout(P, outside_text=True, height=280,
                                    yaxis_title="Scenario weight (%)",
                                    yaxis_range=[0, 80]))
    st.plotly_chart(fig_pm, use_container_width=True)

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
    """.strip(), unsafe_allow_html=True)


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
            Dune's lower Wikipedia search interest reflects zero promotional
            material released to date, not audience demand.
            </div>
            """.strip(),
            unsafe_allow_html=True,
        )


# ── TAB 7: MODEL — ASSUMPTIONS, VARIABLES, FORMULAS ──────────────────────────
with tab7:
    st.markdown(
        f"<p style='font-size:0.58rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:6px;'>"
        "MODEL ASSUMPTIONS &amp; FORMULA SHEET</p>",
        unsafe_allow_html=True,
    )

    # ── AT-A-GLANCE SUMMARY BLOCK ──────────────────────────────────────────────
    fp_dune = FILM_PARAMS["DUNE"]
    fp_av   = FILM_PARAMS["AVENGERS"]
    _glance_html = (
f"<div style='background:{P['info_bg']};border:1px solid {P['card_rule']};border-radius:6px;padding:18px 24px 14px;margin-bottom:20px;'>"
f"<p style='font-size:0.58rem;letter-spacing:2px;color:{P['dim']};margin:0 0 12px;'>KEY ASSUMPTIONS AT A GLANCE</p>"
f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px 24px;'>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>SIMULATIONS</div><div style='font-size:1.25rem;font-weight:600;color:{P['text']};'>5,000</div><div style='font-size:0.7rem;color:{P['dim']};'>Monte Carlo trials</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>OUTPUT</div><div style='font-size:1.25rem;font-weight:600;color:{P['text']};'>P10 \u00b7 P50 \u00b7 P90</div><div style='font-size:0.7rem;color:{P['dim']};'>net-profit distribution</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>WINDOW</div><div style='font-size:1.25rem;font-weight:600;color:{P['text']};'>45 days</div><div style='font-size:0.7rem;color:{P['dim']};'>from opening date</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>IMAX EXCLUSIVE</div><div style='font-size:1.25rem;font-weight:600;color:{P['dune']};'>21 days</div><div style='font-size:0.7rem;color:{P['dim']};'>Dune \u00b7 Dec 18 \u2013 Jan 7</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>DUNE OW MEAN</div><div style='font-size:1.25rem;font-weight:600;color:{P['dune']};'>${fp_dune['ow_gross_mean_M']:.0f}M</div><div style='font-size:0.7rem;color:{P['dim']};'>\u00b1${fp_dune['ow_gross_std_M']:.0f}M \u03c3</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>AVENGERS OW MEAN</div><div style='font-size:1.25rem;font-weight:600;color:{P['av']};'>${fp_av['ow_gross_mean_M']:.0f}M</div><div style='font-size:0.7rem;color:{P['dim']};'>\u00b1${fp_av['ow_gross_std_M']:.0f}M \u03c3</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>DUNE BUDGET</div><div style='font-size:1.25rem;font-weight:600;color:{P['dune']};'>${fp_dune['budget_M']:.0f}M</div><div style='font-size:0.7rem;color:{P['dim']};'>+{fp_dune['mktg_phi']:.0%} mktg \u2192 ${fp_dune['budget_M']*(1+fp_dune['mktg_phi']):.0f}M all-in</div></div>"
f"<div><div style='font-size:0.62rem;color:{P['dim']};letter-spacing:1px;margin-bottom:2px;'>AVENGERS BUDGET</div><div style='font-size:1.25rem;font-weight:600;color:{P['av']};'>${fp_av['budget_M']:.0f}M</div><div style='font-size:0.7rem;color:{P['dim']};'>+{fp_av['mktg_phi']:.0%} mktg \u2192 ${fp_av['budget_M']*(1+fp_av['mktg_phi']):.0f}M all-in</div></div>"
"</div></div>"
    )
    st.markdown(_glance_html, unsafe_allow_html=True)

    # ── OBJECTIVE ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='border-left:2px solid {P['dune']}; padding:10px 18px; margin-bottom:18px;
                    color:{P['text']}; line-height:1.8; font-size:0.88rem;'>
        <b style='color:{P['dune']}; letter-spacing:1px;'>OBJECTIVE</b><br>
        Quantify the net-profit impact on Disney/Marvel of opening
        <i>Avengers: Doomsday</i> on <b>Dec 18, 2026</b> — the same date WB's
        <i>Dune: Part Three</i> holds a confirmed 21-day IMAX exclusive —
        versus moving to <b>May 1, 2027</b> or <b>Jan 16, 2027</b>.<br><br>
        The model produces a <b>P10 / P50 / P90</b> net-profit distribution for
        each film under four mutually exclusive scenarios, using a 5,000-trial
        Monte Carlo simulation with calibrated WOM (word-of-mouth) multipliers,
        calendar demand weights, and IMAX scarcity constraints.
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:4px 0 18px;'>", unsafe_allow_html=True)

    # ── SECTION A: CORE INPUTS ─────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "A · CORE INPUTS (config.py)</p>",
        unsafe_allow_html=True,
    )

    core_rows = [
        ("Release date",                  "Dec 18, 2026",      "Dec 18, 2026",       "OPEN_DATE in config.py"),
        ("Simulation window",             "45 days",           "45 days",            "DAYS = 45"),
        ("Opening weekend gross — mean",  f"${fp_dune['ow_gross_mean_M']:.0f}M",
                                          f"${fp_av['ow_gross_mean_M']:.0f}M",       "FILM_PARAMS[film]['ow_gross_mean_M']"),
        ("Opening weekend gross — σ",     f"${fp_dune['ow_gross_std_M']:.0f}M",
                                          f"${fp_av['ow_gross_std_M']:.0f}M",        "ow_gross_std_M  (±1σ band)"),
        ("Production budget",             f"${fp_dune['budget_M']:.0f}M",
                                          f"${fp_av['budget_M']:.0f}M",              "budget_M"),
        ("Marketing multiplier (φ)",      f"{fp_dune['mktg_phi']:.0%}",
                                          f"{fp_av['mktg_phi']:.0%}",                "All-in cost = budget × (1 + φ)"),
        ("All-in break-even cost",        f"${fp_dune['budget_M']*(1+fp_dune['mktg_phi']):.0f}M",
                                          f"${fp_av['budget_M']*(1+fp_av['mktg_phi']):.0f}M",
                                                                                      "budget_M × 1.50  [cell: =B5*(1+B6)]"),
        ("Audience score — mean",         f"{fp_dune['audience_mean']}",
                                          f"{fp_av['audience_mean']}",               "audience_mean  (0–100 RT/CinemaScore proxy)"),
        ("Audience score — σ",            f"{fp_dune['audience_std']}",
                                          f"{fp_av['audience_std']}",                "audience_std"),
        ("International revenue mult — mean", f"{fp_dune['intl_mult_mean']:.2f}×",
                                          f"{fp_av['intl_mult_mean']:.2f}×",         "intl_mult_mean  (of domestic gross)"),
        ("International revenue mult — σ",    f"{fp_dune['intl_mult_std']:.2f}",
                                          f"{fp_av['intl_mult_std']:.2f}",           "intl_mult_std"),
        ("Studio revenue split",          "60%",               "60%",                "STUDIO_SPLIT = 0.60"),
    ]

    df_core = pd.DataFrame(core_rows, columns=["Input", "Dune: Pt Three", "Avengers: Doomsday", "Variable / Formula"])
    st.dataframe(df_core, use_container_width=True, hide_index=True)

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION B: IMAX CONFIGURATION ─────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "B · IMAX CONFIGURATION (confirmed)</p>",
        unsafe_allow_html=True,
    )

    imax_rows = [
        ("Total US IMAX screens",       "400",    "400",   "SCREEN_INVENTORY['IMAX'] = 400"),
        ("Dune exclusive window",        "21 days","0 days","Dec 18 – Jan 7  (confirmed WB deal)"),
        ("Dune screens during exclusive","400",    "0",     "dune_screens_excl / avengers_screens_excl"),
        ("Split screens after Jan 8",    "200",    "200",   "split_screens = 200 each"),
        ("IMAX ticket price (national avg)","$23.50","$23.50","BASE_PRICE['IMAX']"),
        ("IMAX seat capacity",           "285 seats","285 seats","SEAT_CAPACITY['IMAX']"),
        ("IMAX daily base revenue",      "$4.9M/day","$4.9M/day",
         "IMAX_DAILY_BASE_M = 4.9  [400 screens × mean occ × $23.50]"),
        ("Daily IMAX formula",           "screens/400 × cal_mult × decay_hold × wom_mult × $4.9M",
                                         "same",
         "core.py: compute_imax_revenue()  [cell: =D4/400*CAL*DECAY*WOM*$4.9]"),
    ]

    df_imax = pd.DataFrame(imax_rows, columns=["Input", "Dune: Pt Three", "Avengers: Doomsday", "Variable / Formula"])
    st.dataframe(df_imax, use_container_width=True, hide_index=True)

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION C: WOM MODEL ──────────────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "C · WORD-OF-MOUTH MODEL</p>",
        unsafe_allow_html=True,
    )

    from model.config import WOM_SLOPE, WOM_INTERCEPT
    wom_rows = [
        ("WOM formula",   f"wom_mult = {WOM_SLOPE} × audience_score + ({WOM_INTERCEPT})",
                          "[cell: =WOM_SLOPE*B9+WOM_INTERCEPT]",
                          "Linear regression on 7 comparable films"),
        ("Slope",         f"{WOM_SLOPE}", "WOM_SLOPE = 0.0199",
                          "Calibrated: Endgame, IW, Top Gun, Dune P2, The Flash, Black Adam, DS MoM"),
        ("Intercept",     f"{WOM_INTERCEPT}", "WOM_INTERCEPT = −0.7748", "Same calibration set"),
        ("Clamp range",   "0.5 – 1.5×", "max(0.5, formula)", "Prevents extreme outliers"),
        ("Example @ score=88", f"{max(0.5, WOM_SLOPE*88 + WOM_INTERCEPT):.3f}×",
                          "[cell: =MAX(0.5, 0.0199*88-0.7748)]",
                          "Avengers base: ~1.00×"),
        ("Example @ score=87", f"{max(0.5, WOM_SLOPE*87 + WOM_INTERCEPT):.3f}×",
                          "[cell: =MAX(0.5, 0.0199*87-0.7748)]",
                          "Dune base: ~0.98×"),
        ("WOM effect on holds", "week 2+: hold × clip(wom, 0.6, 1.4)",
                          "core.py line ~262", "Only activates from week 2 — OW holds are fixed"),
    ]

    df_wom = pd.DataFrame(wom_rows, columns=["Parameter", "Value", "Excel Cell Equivalent", "Notes"])
    st.dataframe(df_wom, use_container_width=True, hide_index=True)

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION D: WEEKLY DECAY CURVE ─────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "D · WEEKLY DECAY — BASE HOLDS (fraction of OW gross retained)</p>",
        unsafe_allow_html=True,
    )

    from model.config import WEEKLY_DECAY_BENCHMARKS
    decay_base = [1.00, 0.56, 0.43, 0.34, 0.27, 0.22, 0.18]   # Neutral MCU — used in Monte Carlo
    decay_rows = []
    for wk_idx, hold in enumerate(decay_base):
        label = f"Week {wk_idx}" if wk_idx == 0 else f"Week {wk_idx}"
        ow_label = "Opening Weekend" if wk_idx == 0 else ""
        decay_rows.append((label, f"{hold:.0%}", ow_label,
                           f"wk_holds_base[{wk_idx}]  [cell: =B{wk_idx+2}/B2]"))
    df_decay = pd.DataFrame(decay_rows, columns=["Week", "Hold vs OW", "Notes", "Formula Ref"])
    st.dataframe(df_decay, use_container_width=True, hide_index=True)

    st.caption(
        "Base curve = Neutral MCU (Avengers default). "
        "WOM multiplier scales holds from week 2+ so a high audience score "
        "flattens the curve — modeled as: hold_adj = hold × clip(wom_mult, 0.6, 1.4)."
    )

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION E: CALENDAR DEMAND WEIGHTS ────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "E · CALENDAR DEMAND WEIGHTS</p>",
        unsafe_allow_html=True,
    )

    from model.config import DOW_MULTIPLIERS, HOLIDAY_OVERRIDES
    dow_names = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
    dow_rows = [(dow_names[d], f"{m:.2f}×", "DOW_MULTIPLIERS") for d, m in DOW_MULTIPLIERS.items()]
    holiday_rows = [(f"{m_}/{d_}", f"{mult:.2f}×", key) for (m_, d_), mult in HOLIDAY_OVERRIDES.items()
                    for key in ["HOLIDAY_OVERRIDES"]]
    # de-dup holiday rows
    holiday_rows = [(f"{m_}/{d_}", f"{mult:.2f}×", "HOLIDAY_OVERRIDES") for (m_, d_), mult in HOLIDAY_OVERRIDES.items()]

    cal_rows = dow_rows + holiday_rows
    df_cal = pd.DataFrame(cal_rows, columns=["Date / Day", "Demand Multiplier", "Source"])
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.dataframe(df_cal, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown(
            f"""
            <div style='color:{P['text']}; font-size:0.82rem; line-height:1.9; padding-top:6px;'>
            <b>How the calendar multiplier works:</b><br>
            1. Look up day-of-week baseline (Mon–Sun).<br>
            2. If the date matches a <code>HOLIDAY_OVERRIDES</code> entry, replace baseline.<br>
            3. Dec 28–30 are floored at 0.75× (post-Christmas lingering demand).<br>
            4. Each day's gross = OW_gross × hold_wk × <b>cal_mult</b> / 7<br>
            &nbsp;&nbsp;&nbsp;<span style='font-family:monospace; font-size:0.78rem;'>
            [cell: =OW * HOLD_WK * CAL / 7]</span>
            </div>
            """.strip(),
            unsafe_allow_html=True,
        )

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION F: NET PROFIT FORMULA (FULL WALKTHROUGH) ─────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "F · NET PROFIT FORMULA — FULL WALKTHROUGH (as if Excel)</p>",
        unsafe_allow_html=True,
    )

    _formula_html = (
        f"<div style='color:{P['text']}; font-size:0.82rem; line-height:2.1;"
        f" font-family:monospace; background:transparent; padding:8px 0;'>"
        f"<b style='color:{P['dune']}; font-family:sans-serif; letter-spacing:1px;'>"
        f"PER TRIAL (each of 5,000 Monte Carlo draws)</b><br><br>"
        f"B2  =  NORM.INV(RAND(), audience_mean, audience_std)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ sampled audience score</span><br>"
        f"B3  =  MAX(0.5,&nbsp; WOM_SLOPE × B2 + WOM_INTERCEPT)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ WOM multiplier&nbsp; (e.g. 1.02×)</span><br><br>"
        f"B4  =  NORM.INV(RAND(),&nbsp; ow_gross_mean × scenario_adj × spidey_adj × poly_adj,<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ow_gross_std&nbsp; × scenario_adj)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ opening weekend gross ($M)</span><br><br>"
        f"<b style='color:{P['dim']}; font-family:sans-serif;'>Domestic gross (days 1–45)</b><br>"
        f"B5  =  SUMPRODUCT( B4 × hold[wk] × wom_adj[wk] × cal_mult[day] / 7 )<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ for each day:&nbsp; OW × hold × WOM-adj × CAL / 7</span><br>"
        f"B6  =  B5 × STUDIO_SPLIT&nbsp; <span style='color:{P['dim']};'>→ studio domestic ($M)</span><br><br>"
        f"<b style='color:{P['dim']}; font-family:sans-serif;'>IMAX revenue</b><br>"
        f"B7  =  SUM( IMAX_DAILY_BASE_M × (screens[day]/400) × cal_mult[day] × decay[wk] × B3 )<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ total IMAX gross ($M)</span><br>"
        f"B8  =  B7 × STUDIO_SPLIT&nbsp; <span style='color:{P['dim']};'>→ studio IMAX ($M)</span><br><br>"
        f"<b style='color:{P['dim']}; font-family:sans-serif;'>International</b><br>"
        f"B9  =  B5 × MAX(0.5, NORM.INV(RAND(), intl_mult_mean, intl_mult_std)) × STUDIO_SPLIT<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ studio international ($M)</span><br><br>"
        f"<b style='color:{P['dim']}; font-family:sans-serif;'>Cost &amp; net profit</b><br>"
        f"B10 =  budget_M × (1 + mktg_phi)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dim']};'>→ all-in cost ($M)</span><br>"
        f"<b>B11 =  (B6 + B8 + B9) − B10<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{P['dune']};'>→ NET STUDIO PROFIT ($M)&nbsp; ← output cell</span></b><br><br>"
        f"<b style='color:{P['dim']}; font-family:sans-serif;'>Output statistics (over 5,000 trials)</b><br>"
        f"P10  =  PERCENTILE(B11:range, 0.10) &nbsp;&nbsp;"
        f"P50  =  PERCENTILE(B11:range, 0.50) &nbsp;&nbsp;"
        f"P90  =  PERCENTILE(B11:range, 0.90)<br>"
        f"Break-even%  =  COUNTIF(B11:range, \"&gt;0\") / 5000 × 100"
        f"</div>"
    )
    st.markdown(_formula_html, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION G: SCENARIO OW ADJUSTMENTS ────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "G · SCENARIO OW MULTIPLIERS</p>",
        unsafe_allow_html=True,
    )

    from model.core import SCENARIO_OW_ADJ
    scen_rows = []
    for (film, sk), mult in SCENARIO_OW_ADJ.items():
        scen_rows.append((SCENARIOS[sk]["label"], film.title(), f"{mult:.2f}×",
                          SCENARIOS[sk]["description"]))
    df_scen = pd.DataFrame(scen_rows, columns=["Scenario", "Film", "OW Gross Multiplier", "Description"])
    st.dataframe(df_scen, use_container_width=True, hide_index=True)

    st.caption(
        "Scenario OW multiplier is applied to ow_gross_mean_M before sampling. "
        "E.g. Avengers in Scenario B gets 1.22× — a $240M mean becomes $293M mean — "
        "reflecting uncontested May holiday positioning."
    )

    st.markdown(f"<hr style='border-color:{P['card_rule']}; margin:14px 0;'>", unsafe_allow_html=True)

    # ── SECTION H: KEY ASSUMPTIONS & CAVEATS ─────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.62rem; letter-spacing:2px; color:{P['dim']}; margin-bottom:8px;'>"
        "H · KEY ASSUMPTIONS &amp; WHAT THE MODEL DOES NOT CAPTURE</p>",
        unsafe_allow_html=True,
    )

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown(
            f"""
            <div style='color:{P['text']}; font-size:0.82rem; line-height:1.9;'>
            <b style='color:{P['dune']}'>Assumptions baked in</b><br>
            • WB's 21-day IMAX exclusive is <b>confirmed</b> and non-negotiable<br>
            • Studio keeps 60% of gross (all formats)<br>
            • Marketing spend = 50% of production budget<br>
            • International = domestic × intl_mult (correlated in mean, independent σ)<br>
            • Calendar multipliers calibrated from Avatar (2022) and TFA (2015) Christmas data<br>
            • Audience score drives WOM via linear regression on 7 comparable films<br>
            • Polymarket odds are treated as crowd-wisdom priors, not direct inputs
            </div>
            """.strip(),
            unsafe_allow_html=True,
        )
    with h_col2:
        st.markdown(
            f"""
            <div style='color:{P['text']}; font-size:0.82rem; line-height:1.9;'>
            <b style='color:{P['av']}'>What the model does NOT capture</b><br>
            • PLF / 3D / standard-screen revenue (IMAX + domestic gross proxy only)<br>
            • Streaming / home-video revenue (upside not modeled)<br>
            • China box office (excluded — China often reported separately)<br>
            • Competitive spillover: if both hold, audiences may self-sort by preference<br>
            • Marketing effectiveness variance (φ held fixed at 50%)<br>
            • The Polymarket full-year market includes a calendar-year bias vs. Avengers'
              2-week Dec run — ratio signal is directional, not exact
            </div>
            """.strip(),
            unsafe_allow_html=True,
        )


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<p style='color:{P['footer']}; font-size:0.6rem; letter-spacing:1.5px;
   margin-top:12px; text-align:right;'>
DUNESDAY v5 &nbsp;·&nbsp; {signals['last_updated']} &nbsp;·&nbsp; {' · '.join(cal['sources'])}
</p>
""", unsafe_allow_html=True)
