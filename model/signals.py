"""
model/signals.py

Live signal fetching — Google Trends + YouTube Data API.
Called on every Streamlit page load. Fails gracefully to last
known values if network is unavailable.

GOOGLE TRENDS: No API key needed. Runs immediately.
YOUTUBE API:   Set YOUTUBE_API_KEY in Streamlit secrets or .env.
               Returns None gracefully until key is configured.
"""

import os
import json
import datetime
import numpy as np

# ── FALLBACK VALUES (update manually after each major event) ──────────────────
FALLBACK_SIGNALS = {
    "last_updated":  "2026-02-26",
    "source":        "fallback",
    "avengers": {
        "trends_interest":   72,    # Google Trends index 0-100 (US, 90d)
        "trends_dune_ratio": 0.18,  # Dune / Avengers search ratio
        "yt_trailer_views":  None,  # YouTube view count (full trailer not out yet)
        "teaser_views_x_M":  [53.0, 25.0, 15.0, 9.0],  # T1-T4 on X
        "combined_views_B":  1.02,
        "full_trailer_out":  False,
    },
    "dune": {
        "trends_interest":   13,
        "yt_trailer_views":  None,
        "full_trailer_out":  False,
        "alamo_rank":        1,
    },
    "calibration": {
        "avengers_score_adj": 0.0,   # adjustment to base audience score
        "dune_score_adj":     0.0,
        "signal_confidence":  "low",  # low / medium / high
        "notes":              "No full trailer yet. Calibration based on teaser decay only.",
    }
}

# ── BENCHMARK DECAY CURVE (Deadpool & Wolverine — nostalgia spike held) ───────
# T1→T2 view ratio benchmarks:
# D&W:           365M → 280M = −23%   (held well)
# Love&Thunder:  200M → 95M  = −53%   (didn't hold)  ← Avengers matches this
# Endgame:       289M → 210M = −27%   (held)
DECAY_BENCHMARKS = {
    "held":      0.77,   # D&W, Endgame — good sign
    "neutral":   0.60,
    "soft":      0.47,   # Love & Thunder territory
    "collapsed": 0.30,
}


# ── AUDIENCE SCORE CALIBRATION ────────────────────────────────────────────────

def calibrate_from_teaser_decay(views_list: list) -> float:
    """
    Convert teaser view decay curve into audience score adjustment.
    Uses T1→T2 ratio as primary signal (most predictive pair).

    Returns: float adjustment to add to base audience score (-8 to +4)
    """
    if not views_list or len(views_list) < 2:
        return 0.0

    t1, t2 = views_list[0], views_list[1]
    if t1 <= 0:
        return 0.0

    ratio = t2 / t1  # 0-1, higher = better retention

    if ratio >= DECAY_BENCHMARKS["held"]:
        return +3.0    # Endgame/D&W territory — upgrade
    elif ratio >= DECAY_BENCHMARKS["neutral"]:
        return +1.0
    elif ratio >= DECAY_BENCHMARKS["soft"]:
        return -2.0    # Love&Thunder territory
    else:
        return -5.0    # collapsed — significant downgrade


def calibrate_from_trends(av_interest: int, dune_interest: int) -> dict:
    """
    Convert Google Trends data into audience score adjustments.
    Trends index is relative (0-100), so only the ratio matters.

    Returns adjustments for both films.
    """
    total = av_interest + dune_interest
    if total == 0:
        return {"avengers": 0.0, "dune": 0.0}

    av_share   = av_interest / total
    dune_share = dune_interest / total

    # Avengers baseline expectation at this stage: ~80% of search share
    # (has 4 teasers, Dune has zero trailers)
    av_expected  = 0.82
    dune_expected = 0.18

    av_adj   = (av_share   - av_expected)  * 20   # ±4 max
    dune_adj = (dune_share - dune_expected) * 20

    return {
        "avengers": float(np.clip(av_adj,   -4, 4)),
        "dune":     float(np.clip(dune_adj, -4, 4)),
    }


def calibrate_from_yt_views(views: int, film: str) -> float:
    """
    Convert YouTube full-trailer view count into audience score adjustment.
    Benchmarks based on 24hr view counts at similar release stages.
    """
    if views is None:
        return 0.0

    views_M = views / 1_000_000

    if film == "AVENGERS":
        # Benchmarks (24hr views at comparable stage, ~8 months out):
        # Endgame:   289M → opened $357M OW, 91 score
        # IW:        230M → opened $258M OW, 91 score
        # D&W:       365M → opened $211M OW, 95 score
        # L&T:       ~200M→ opened $144M OW, 78 score
        if views_M >= 300:   return +4.0   # Endgame tier
        elif views_M >= 200: return +2.0   # IW tier
        elif views_M >= 150: return  0.0   # neutral
        elif views_M >= 100: return -2.0   # soft
        else:                return -5.0   # MCU fatigue signal
    else:  # DUNE
        # Dune Part Two: ~40M first trailer → $82M OW, 96 score
        if views_M >= 80:    return +4.0   # exceeded Part Two
        elif views_M >= 40:  return +2.0   # matches Part Two
        elif views_M >= 25:  return  0.0
        else:                return -2.0


# ── GOOGLE TRENDS FETCH ───────────────────────────────────────────────────────

def fetch_google_trends() -> dict:
    """
    Fetch 90-day search interest for both films.
    Returns dict with interest scores and week-over-week change.
    Falls back to None on any error.
    """
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload(
            ["Avengers Doomsday", "Dune Part Three"],
            timeframe="today 3-m",
            geo="US",
        )
        df = pytrends.interest_over_time()
        if df.empty:
            return None

        latest = df.iloc[-1]
        prev   = df.iloc[-2] if len(df) > 1 else latest

        return {
            "avengers_now":  int(latest.get("Avengers Doomsday", 0)),
            "dune_now":      int(latest.get("Dune Part Three", 0)),
            "avengers_prev": int(prev.get("Avengers Doomsday", 0)),
            "dune_prev":     int(prev.get("Dune Part Three", 0)),
            "fetched_at":    datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return None


# ── YOUTUBE API FETCH ─────────────────────────────────────────────────────────

# Official video IDs — update when full trailers drop
YOUTUBE_VIDEO_IDS = {
    "avengers_t1": "cz9JFwwgm3k",   # Teaser 1 (Steve Rogers)
    "avengers_t2": "placeholder",    # Teaser 2 (Thor) — update with real ID
    "avengers_full": None,           # Full trailer — not released yet
    "dune_t1": None,                 # Not released yet
}

def fetch_youtube_views(video_ids: list = None) -> dict:
    """
    Fetch view counts for official trailer videos.
    Requires YOUTUBE_API_KEY in environment or Streamlit secrets.

    To set up:
    1. Go to console.cloud.google.com
    2. Enable YouTube Data API v3
    3. Create API key (free, 10k units/day quota)
    4. Add to Streamlit secrets: YOUTUBE_API_KEY = "your-key"
    """
    api_key = None

    # Try Streamlit secrets first (deployed)
    try:
        import streamlit as st
        api_key = st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        pass

    # Fallback to environment variable (local dev)
    if not api_key:
        api_key = os.environ.get("YOUTUBE_API_KEY")

    if not api_key:
        return {"status": "no_key", "message": "Add YOUTUBE_API_KEY to Streamlit secrets"}

    try:
        import requests
        ids_to_fetch = [v for v in (video_ids or list(YOUTUBE_VIDEO_IDS.values())) if v]
        if not ids_to_fetch:
            return {"status": "no_ids"}

        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part":  "statistics,snippet",
            "id":    ",".join(ids_to_fetch),
            "key":   api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = {}
        for item in data.get("items", []):
            vid_id = item["id"]
            stats  = item.get("statistics", {})
            title  = item.get("snippet", {}).get("title", "")
            results[vid_id] = {
                "views":    int(stats.get("viewCount", 0)),
                "likes":    int(stats.get("likeCount", 0)),
                "title":    title,
            }

        return {"status": "ok", "videos": results, "fetched_at": datetime.datetime.utcnow().isoformat()}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── MASTER FETCH + CALIBRATE ──────────────────────────────────────────────────

def fetch_and_calibrate(base_dune_score: int = 87, base_av_score: int = 88) -> dict:
    """
    Main entry point called by Streamlit on page load.
    Returns calibrated audience scores + all signal data.
    Gracefully falls back to hardcoded values on any failure.
    """
    signals = dict(FALLBACK_SIGNALS)  # start with fallback
    av_adj  = 0.0
    dune_adj = 0.0
    sources_used = []

    # ── 1. Google Trends ──────────────────────────────────────────────────────
    trends = fetch_google_trends()
    if trends:
        signals["avengers"]["trends_interest"] = trends["avengers_now"]
        signals["dune"]["trends_interest"]     = trends["dune_now"]
        signals["last_updated"] = trends["fetched_at"][:10]
        signals["source"] = "live"

        trend_adj = calibrate_from_trends(trends["avengers_now"], trends["dune_now"])
        av_adj   += trend_adj["avengers"]
        dune_adj += trend_adj["dune"]
        sources_used.append("Google Trends")
    else:
        sources_used.append("Google Trends (fallback)")

    # ── 2. Teaser decay calibration ───────────────────────────────────────────
    av_decay_adj = calibrate_from_teaser_decay(
        signals["avengers"]["teaser_views_x_M"]
    )
    av_adj += av_decay_adj
    sources_used.append("Teaser decay curve")

    # ── 3. YouTube API ────────────────────────────────────────────────────────
    yt = fetch_youtube_views()
    if yt.get("status") == "ok" and yt.get("videos"):
        # Sum avengers trailer views
        av_yt_total = sum(
            v["views"] for vid_id, v in yt["videos"].items()
            if "avengers" in vid_id.lower() or vid_id in [
                YOUTUBE_VIDEO_IDS.get("avengers_t1"),
                YOUTUBE_VIDEO_IDS.get("avengers_full"),
            ]
        )
        dune_yt_total = sum(
            v["views"] for vid_id, v in yt["videos"].items()
            if "dune" in vid_id.lower() or vid_id == YOUTUBE_VIDEO_IDS.get("dune_t1")
        )
        if av_yt_total > 0:
            av_adj   += calibrate_from_yt_views(av_yt_total, "AVENGERS")
            sources_used.append("YouTube API")
        if dune_yt_total > 0:
            dune_adj += calibrate_from_yt_views(dune_yt_total, "DUNE")

        signals["avengers"]["yt_trailer_views"] = av_yt_total or None
        signals["dune"]["yt_trailer_views"]     = dune_yt_total or None
    else:
        signals["_yt_status"] = yt.get("status", "unavailable")
        signals["_yt_message"] = yt.get("message", "")

    # ── 4. Final calibrated scores ────────────────────────────────────────────
    av_calibrated   = float(np.clip(base_av_score   + av_adj,   60, 100))
    dune_calibrated = float(np.clip(base_dune_score + dune_adj, 60, 100))

    # Signal confidence
    if "YouTube API" in sources_used and "Google Trends" in sources_used \
            and not any("fallback" in s for s in sources_used):
        confidence = "high"
    elif any(s for s in sources_used if "fallback" not in s):
        confidence = "medium"
    else:
        confidence = "low"

    signals["calibration"] = {
        "avengers_base":      base_av_score,
        "dune_base":          base_dune_score,
        "avengers_adj":       round(av_adj, 1),
        "dune_adj":           round(dune_adj, 1),
        "avengers_calibrated": av_calibrated,
        "dune_calibrated":     dune_calibrated,
        "sources":             sources_used,
        "signal_confidence":   confidence,
        "teaser_decay_signal": "soft" if av_decay_adj < -1 else "neutral" if av_decay_adj == 0 else "strong",
        "notes": _build_notes(av_adj, dune_adj, trends, yt),
    }

    return signals


def _build_notes(av_adj, dune_adj, trends, yt) -> str:
    notes = []
    if av_adj < -2:
        notes.append(f"Avengers downgraded {av_adj:+.0f}pts — teaser decay matches Love&Thunder pattern.")
    elif av_adj > 2:
        notes.append(f"Avengers upgraded {av_adj:+.0f}pts — strong engagement signal.")
    else:
        notes.append("Avengers signal neutral — insufficient data for strong calibration.")

    if yt and yt.get("status") != "ok":
        notes.append("Add YOUTUBE_API_KEY to Streamlit secrets for live trailer view data.")

    if not trends:
        notes.append("Google Trends unavailable — using fallback search interest values.")

    return " ".join(notes)
