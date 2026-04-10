"""
model/signals.py

Live signal fetching — YouTube Data API + Wikipedia + TMDB + Trakt.
Called on every Streamlit page load. Fails gracefully to last
known values if network is unavailable.

YOUTUBE API:   Set YOUTUBE_API_KEY in Streamlit secrets or .env.
               Returns None gracefully until key is configured.
TMDB API:      Set TMDB_API_KEY in Streamlit secrets or .env.
               Free key at themoviedb.org/settings/api (~2 min).
               Pulls popularity score + vote count for both films.
               Set TMDB_MOVIE_IDS below after looking up IDs at
               themoviedb.org (search film title, copy ID from URL).
TRAKT API:     Set TRAKT_CLIENT_ID in Streamlit secrets or .env.
               Free app at trakt.tv/oauth/applications (script type,
               no redirect URI needed, ~2 min). Pulls watchlist
               collectors + watcher count — organic demand signal.
"""

import os
import json
import datetime
import numpy as np

# ── FALLBACK VALUES (update manually after each major event) ──────────────────
def _is_fresh_trailer(date_str: str) -> bool:
    """Return True if a trailer was released today (< 24 hours old by date).
    Suppresses calibration when view counts are too early to compare
    against 24-hour benchmarks."""
    if not date_str:
        return False
    try:
        release_date = datetime.date.fromisoformat(date_str)
        return (datetime.date.today() - release_date).days < 1
    except ValueError:
        return False


FALLBACK_SIGNALS = {
    "last_updated":  "2026-02-26",
    "source":        "fallback",
    "avengers": {
        "yt_trailer_views":  None,  # YouTube view count (full trailer not out yet)
        "teaser_views_x_M":  [53.0, 25.0, 15.0, 9.0],  # T1-T4 on X
        "combined_views_B":  1.02,
        "full_trailer_out":  False,
        "wiki_views_7d":     None,  # Wikipedia pageviews last 7 days
        "wiki_wow_pct":      None,  # week-over-week % change
    },
    "dune": {
        "yt_trailer_views":    None,
        "full_trailer_out":    False,
        "trailer_date":        "2026-03-18",
        "alamo_rank":          1,
        "wiki_views_7d":       None,
        "wiki_wow_pct":        None,
        "imax_70mm_sold_out":  True,    # confirmed Apr 2026 — all US 70mm IMAX dates gone
    },
    "spiderman": {
        "full_trailer_released": True,
        "trailer_date":          "2026-03-18",
        "yt_trailer_views_M":    None,   # populated from YouTube API when ID is configured
        "suggested_tier":        None,   # auto-calibrated from view count
    },
    "calibration": {
        "avengers_score_adj":  0.0,   # adjustment to base audience score
        "dune_score_adj":      0.0,
        "signal_confidence":   "low",  # low / medium / high
        "spidey_suggested_tier": None, # auto-suggested from Spider-Man trailer views
        "notes":               "Spider-Man: Brand New Day trailer released 2026-03-18. Calibration pending view count data.",
    }
}

# ── BENCHMARK DECAY CURVE (Deadpool & Wolverine — nostalgia spike held) ──────
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


def calibrate_from_spidey_trailer(views_M: float) -> str:
    """
    Map Spider-Man: Brand New Day trailer view count to an impact tier.
    Used to auto-suggest the SPIDEY_IMPACT_ADJ tier in the sidebar.

    Benchmarks (24hr combined views):
      BND T1:         718.6M → all-time record, MCU demand at historic peak
      D&W T1:         365M   → previous film record (Super Bowl drop)
      NWH T1:         355M   → previous MCU record
      FFH T1:         135M   → strong, healthy MCU
      Homecoming T1:  ~64M   → acceptable for reboot
    Returns: tier string matching SPIDEY_IMPACT_ADJ keys
    """
    if views_M is None:
        return None

    if views_M >= 350:    return "Blockbuster"   # D&W/NWH territory and above
    elif views_M >= 200:  return "Strong"         # healthy MCU demand signal
    elif views_M >= 110:  return "Neutral"        # matches FFH baseline
    elif views_M >= 60:   return "Soft"           # below Homecoming level
    else:                 return "Disappoints"    # MCU fatigue confirmed


def calibrate_from_imax_70mm_sellout(sold_out: bool, film: str) -> float:
    """
    Audience score adjustment from 70mm IMAX advance ticket sellout.

    A full 70mm IMAX sellout before the full trailer drop is unusually strong —
    it means the film's hardcore cinephile base has already committed real money
    with almost no marketing. That audience is the same one that drives Dune's
    high audience scores and repeat-viewing holds.

    Dune Part Two sold out 70mm in <48hrs after its first trailer. Part Three
    selling out with only a teaser is a meaningfully stronger signal at the
    same stage of the release cycle.

    Only applies to Dune — Avengers doesn't get 70mm allocation (IMAX Digital).
    Returns: float adjustment to add to base audience score
    """
    if not sold_out or film != "DUNE":
        return 0.0
    return +2.0   # cinephile core locked in — stronger than Alamo #1 signal


def calibrate_from_trailer_engagement(views: int, likes: int) -> str | None:
    """
    Map like/view ratio to a Spider-Man impact tier.
    Time-independent — valid from minute one, so trailers released months
    apart can be compared on equal footing.

    Benchmarks (early-period like rates for comparable MCU/blockbuster releases):
      NWH T1:         ~4–5%  → record-breaking enthusiasm
      FFH / D&W:      ~3–4%  → healthy MCU engagement
      Homecoming:     ~2–3%  → acceptable reboot interest
      Thor L&T:       ~1–2%  → underwhelming, MCU fatigue showing
    """
    if not views or not likes:
        return None
    ratio = likes / views   # e.g. 0.035 = 3.5%

    if ratio >= 0.045:   return "Blockbuster"
    elif ratio >= 0.033: return "Strong"
    elif ratio >= 0.022: return "Neutral"
    elif ratio >= 0.013: return "Soft"
    else:                return "Disappoints"



# ── YOUTUBE API FETCH ─────────────────────────────────────────────────────────

# Official video IDs — update when full trailers drop
YOUTUBE_VIDEO_IDS = {
    "avengers_t1":       "399Ez7WHK5s",   # Teaser 1
    "avengers_t2":       "kH1XlwHQv9o",   # Teaser 2
    "avengers_t3":       "1clWprLC5Ak",   # Teaser 3
    "avengers_t4":       "UiMg566PREA",   # Teaser 4
    "avengers_full":     None,            # Full trailer — not released yet
    "avengers_countdown": "f17J3AXVK5w",  # Countdown clock
    "dune_t1":           "3_9vCamtuPY",   # Dune: Part Three Teaser
    "spiderman_full":    "8TZMtslA3UY",   # Spider-Man: BND trailer (2026-03-18)
}

# YouTube URLs for embedded playback (trailer display)
YOUTUBE_TRAILER_URLS = {
    "dune_t1":            "https://www.youtube.com/watch?v=3_9vCamtuPY",
    "avengers_t1":        "https://www.youtube.com/watch?v=399Ez7WHK5s",
    "avengers_t2":        "https://www.youtube.com/watch?v=kH1XlwHQv9o",
    "avengers_t3":        "https://www.youtube.com/watch?v=1clWprLC5Ak",
    "avengers_t4":        "https://www.youtube.com/watch?v=UiMg566PREA",
    "avengers_countdown": "https://www.youtube.com/watch?v=f17J3AXVK5w",
    "spiderman_full":     "https://www.youtube.com/watch?v=8TZMtslA3UY",
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



# ── WIKIPEDIA PAGEVIEWS FETCH ─────────────────────────────────────────────────

# Wikipedia article titles for each film
WIKIPEDIA_ARTICLES = {
    "avengers": "Avengers:_Doomsday",
    "dune":     "Dune:_Part_Three",
}

def fetch_wikipedia_pageviews(days: int = 30) -> dict:
    """
    Fetch daily Wikipedia pageview counts for both films.
    Uses the free Wikimedia REST API — no key, no registration, works from cloud.

    Returns total + last-7-day views and week-over-week % change per film.
    """
    try:
        import requests

        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)

        results = {}
        for film, article in WIKIPEDIA_ARTICLES.items():
            url = (
                f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
                f"/en.wikipedia/all-access/all-agents/{article}/daily"
                f"/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
            )
            resp = requests.get(
                url, timeout=10,
                headers={"User-Agent": "dunesday/1.0 (box office research)"}
            )
            if resp.status_code != 200:
                results[film] = None
                continue

            items = resp.json().get("items", [])
            if not items:
                results[film] = None
                continue

            views_by_day = [item["views"] for item in items]
            total        = sum(views_by_day)
            last_7       = sum(views_by_day[-7:])  if len(views_by_day) >= 7  else sum(views_by_day)
            prev_7       = sum(views_by_day[-14:-7]) if len(views_by_day) >= 14 else None
            wow_pct      = round((last_7 - prev_7) / prev_7 * 100, 1) if prev_7 else None

            results[film] = {
                "total_views":   total,
                "last_7d_views": last_7,
                "prev_7d_views": prev_7,
                "wow_pct":       wow_pct,
                "daily_avg":     round(total / len(views_by_day), 1),
            }

        return {
            "status":      "ok",
            "films":       results,
            "period_days": days,
            "fetched_at":  datetime.datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def calibrate_from_wikipedia(last_7d_views: int, wow_pct: float, film: str) -> float:

    """
    Convert Wikipedia pageview data into audience score adjustment.

    last_7d_views : total pageviews over the last 7 days
    wow_pct       : week-over-week % change (positive = growing interest)
    film          : "AVENGERS" or "DUNE"

    Thresholds scaled to each film's expected baseline at this stage (9 months out).
    Returns: float adjustment (-3 to +3)
    """
    if last_7d_views is None:
        return 0.0

    # Base adjustment from absolute view level
    if film == "AVENGERS":
        # ~2k/day = strong, ~1k/day = normal, <500/day = soft
        if last_7d_views >= 14000:  base_adj = +2.0
        elif last_7d_views >= 7000: base_adj = +1.0
        elif last_7d_views >= 3000: base_adj =  0.0
        else:                       base_adj = -1.0
    else:  # DUNE — smaller fanbase, lower baseline expected
        if last_7d_views >= 7000:   base_adj = +2.0
        elif last_7d_views >= 3500: base_adj = +1.0
        elif last_7d_views >= 1500: base_adj =  0.0
        else:                       base_adj = -1.0

    # Momentum adjustment from week-over-week trend
    if wow_pct is not None:
        if wow_pct >= 50:    mom_adj = +1.0   # surging
        elif wow_pct >= 10:  mom_adj = +0.5   # growing
        elif wow_pct >= -10: mom_adj =  0.0   # stable
        elif wow_pct >= -30: mom_adj = -0.5   # cooling
        else:                mom_adj = -1.0   # fading fast
    else:
        mom_adj = 0.0

    return float(np.clip(base_adj + mom_adj, -3, 3))


# ── TMDB ──────────────────────────────────────────────────────────────────────

# TMDB movie IDs — look up at themoviedb.org (search title, copy integer from URL)
# e.g. themoviedb.org/movie/693134  →  ID is 693134
TMDB_MOVIE_IDS = {
    "avengers": 1003596,   # Avengers: Doomsday — themoviedb.org/movie/1003596
    "dune":     1170608,   # Dune: Part Three  — themoviedb.org/movie/1170608
}

def fetch_tmdb_signals() -> dict:
    """
    Fetch TMDB popularity score + vote count for both films.
    Requires TMDB_API_KEY in environment or Streamlit secrets.

    TMDB popularity is a rolling composite of:
      - Page views on TMDB
      - Watchlist/favourite adds
      - Vote activity
    At 9 months out, a score > 200 indicates strong tentpole tracking.
    """
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("TMDB_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return {"status": "no_key", "message": "Add TMDB_API_KEY to Streamlit secrets"}

    results = {}
    try:
        import requests
        for film, movie_id in TMDB_MOVIE_IDS.items():
            if movie_id is None:
                results[film] = None
                continue
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            resp = requests.get(url, params={"api_key": api_key}, timeout=10)
            if resp.status_code != 200:
                results[film] = None
                continue
            data = resp.json()
            results[film] = {
                "popularity":   data.get("popularity"),
                "vote_count":   data.get("vote_count"),
                "vote_average": data.get("vote_average"),
                "title":        data.get("title"),
            }
        return {"status": "ok", "films": results,
                "fetched_at": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def calibrate_from_tmdb(popularity: float, film: str) -> float:
    """
    Map TMDB popularity score to an audience score adjustment.

    TMDB popularity benchmarks at ~9 months pre-release:
      Endgame-level tentpole:  500+
      Strong MCU/blockbuster:  200–500
      Neutral:                 75–200
      Soft tracking:           25–75
      Low / MCU fatigue:       <25

    Dune has a smaller but highly engaged fanbase — lower absolute
    threshold still represents strong relative demand.
    """
    if popularity is None:
        return 0.0

    if film == "AVENGERS":
        if popularity >= 500:   return +3.0
        elif popularity >= 200: return +1.5
        elif popularity >= 75:  return  0.0
        elif popularity >= 25:  return -1.5
        else:                   return -3.0
    else:  # DUNE — smaller fandom, lower absolute baseline
        if popularity >= 200:   return +3.0
        elif popularity >= 75:  return +1.5
        elif popularity >= 30:  return  0.0
        elif popularity >= 10:  return -1.5
        else:                   return -3.0


# ── TRAKT ──────────────────────────────────────────────────────────────────────

# Trakt slugs — usually lowercase-hyphenated title, verify at trakt.tv/movies/<slug>
TRAKT_MOVIE_SLUGS = {
    "avengers": "avengers-doomsday",
    "dune":     "dune-part-three",
}

def fetch_trakt_signals() -> dict:
    """
    Fetch Trakt watchlist collectors + watcher count for both films.
    Requires TRAKT_CLIENT_ID in environment or Streamlit secrets.

    'collectors' = users who added to collection (high-intent, organic demand)
    'watchers'   = users who have watched (pre-release: near-zero, ignore)
    'lists'      = times added to curated lists (taste-graph signal)

    At 9 months pre-release, collectors is the most meaningful metric.
    A collector count growing week-over-week signals building organic demand.
    """
    client_id = None
    try:
        import streamlit as st
        client_id = st.secrets.get("TRAKT_CLIENT_ID")
    except Exception:
        pass
    if not client_id:
        client_id = os.environ.get("TRAKT_CLIENT_ID")
    if not client_id:
        return {"status": "no_key", "message": "Add TRAKT_CLIENT_ID to Streamlit secrets"}

    results = {}
    try:
        import requests
        headers = {
            "Content-Type":      "application/json",
            "trakt-api-version": "2",
            "trakt-api-key":     client_id,
        }
        for film, slug in TRAKT_MOVIE_SLUGS.items():
            url = f"https://api.trakt.tv/movies/{slug}/stats"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                results[film] = None
                continue
            data = resp.json()
            results[film] = {
                "watchers":   data.get("watchers"),
                "collectors": data.get("collectors"),
                "lists":      data.get("lists"),
                "comments":   data.get("comments"),
            }
        return {"status": "ok", "films": results,
                "fetched_at": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def calibrate_from_trakt(collectors: int, film: str) -> float:
    """
    Map Trakt collector count to an audience score adjustment.

    'collectors' measures organic anticipation — users proactively
    adding the film to track it, independent of marketing spend.
    High collector counts at 9 months out predict strong opening weekends.

    Benchmarks (comparable blockbusters at 9 months pre-release):
      Endgame-level:  100k+ collectors
      Strong:          40k–100k
      Neutral:         15k–40k
      Soft:             5k–15k
      Low:             <5k
    """
    if collectors is None:
        return 0.0

    if film == "AVENGERS":
        if collectors >= 100_000:  return +3.0
        elif collectors >= 40_000: return +1.5
        elif collectors >= 15_000: return  0.0
        elif collectors >=  5_000: return -1.5
        else:                      return -3.0
    else:  # DUNE — smaller Trakt userbase overlap
        if collectors >= 40_000:   return +3.0
        elif collectors >= 15_000: return +1.5
        elif collectors >=  5_000: return  0.0
        elif collectors >=  1_500: return -1.5
        else:                      return -3.0


# ── POLYMARKET ────────────────────────────────────────────────────────────────
# No API key needed — public prediction market data.
# Two markets are tracked:
#   "full_year"  — "Highest grossing movie in 2026?" (calendar year domestic)
#   "ow"         — "Which movie has the biggest opening weekend in 2026?"
#
# Each event contains one Yes/No market per film.
# The Yes price (0–1) is the crowd's implied probability.
#
# KEY INSIGHT: The gap between opening weekend odds and full-year odds directly
# prices the IMAX / legs damage from the Dec 18 conflict. Currently:
#   Avengers opening weekend: ~75%  Full-year: ~21%  → market expects a legs collapse
# If Disney moved, the full-year odds would converge toward the opening weekend odds.

POLYMARKET_MARKET_SLUGS = {
    # Individual Yes/No market slugs from polymarket.com event URLs
    "avengers_opening_weekend": "will-avengers-doomsday-have-the-best-domestic-opening-weekend-in-2026",
    "avengers_full_year": "will-avengers-doomsday-be-the-top-grossing-movie-of-2026",
    "dune_full_year":     "will-dune-part-three-be-the-top-grossing-movie-of-2026",
}

POLYMARKET_EVENT_SLUGS = {
    "full_year":       "highest-grossing-movie-in-2026",
    "opening_weekend": "which-movie-has-biggest-opening-weekend-in-2026",
}

# Hardcoded fallback odds (update manually when odds shift significantly)
POLYMARKET_FALLBACK = {
    "avengers_opening_weekend_odds": 0.75,   # 75% — best opening weekend in 2026
    "avengers_full_year_odds": 0.21,   # 21% — highest full-year gross in 2026
    "dune_full_year_odds":     None,   # not in top markets yet
    "last_updated":            "2026-03-18",
}


def fetch_polymarket_signals() -> dict:
    """
    Fetch live prediction market odds from Polymarket's public Gamma API.
    No API key required.

    Tries the event-level endpoint first (one call, all films), then falls
    back to individual market slugs. Falls back to POLYMARKET_FALLBACK on
    any network failure.

    Returns dict with:
      avengers_opening_weekend_odds        — P(Avengers has best opening weekend in 2026)
      avengers_full_year_odds — P(Avengers is top grossing film in 2026)
      dune_full_year_odds     — P(Dune Part Three is top grossing film in 2026)
      opening_weekend_decay_ratio          — avengers_opening_weekend_odds / avengers_full_year_odds
                                (> 2.0 signals market expects a legs problem)
    """
    try:
        import requests
        headers = {
            "User-Agent": "dunesday/1.0 (box office research; contact via github)",
            "Accept":     "application/json",
        }

        result = {}

        # ── Strategy 1: event-level fetch (gets all films in one call) ────────
        for market_type, event_slug in POLYMARKET_EVENT_SLUGS.items():
            url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            events = resp.json()
            if not events:
                continue
            event = events[0] if isinstance(events, list) else events
            for mkt in event.get("markets", []):
                question   = (mkt.get("question") or mkt.get("groupItemTitle") or "").lower()
                prices_raw = mkt.get("outcomePrices") or mkt.get("bestBid") or []
                # outcomePrices is ["yes_price", "no_price"] as strings
                if not prices_raw:
                    continue
                try:
                    yes_price = float(prices_raw[0]) if isinstance(prices_raw, list) else None
                except (ValueError, TypeError):
                    yes_price = None
                if yes_price is None:
                    continue

                if "avengers" in question or "doomsday" in question:
                    if market_type == "opening_weekend":
                        result["avengers_opening_weekend_odds"] = yes_price
                    else:
                        result["avengers_full_year_odds"] = yes_price
                elif "dune" in question:
                    if market_type == "full_year":
                        result["dune_full_year_odds"] = yes_price

        # ── Strategy 2: individual market slugs for anything still missing ────
        needed = {
            "avengers_opening_weekend_odds":        POLYMARKET_MARKET_SLUGS["avengers_opening_weekend"],
            "avengers_full_year_odds": POLYMARKET_MARKET_SLUGS["avengers_full_year"],
            "dune_full_year_odds":     POLYMARKET_MARKET_SLUGS["dune_full_year"],
        }
        for key, slug in needed.items():
            if key in result:
                continue
            url  = f"https://gamma-api.polymarket.com/markets?slug={slug}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            mkts = resp.json()
            mkt  = mkts[0] if isinstance(mkts, list) and mkts else mkts
            prices_raw = mkt.get("outcomePrices") or []
            try:
                result[key] = float(prices_raw[0])
            except (IndexError, ValueError, TypeError):
                pass

        if not result:
            fallback_data = dict(POLYMARKET_FALLBACK)
            av_opening_weekend = fallback_data.get("avengers_opening_weekend_odds")
            av_fy = fallback_data.get("avengers_full_year_odds")
            if av_opening_weekend and av_fy and av_fy > 0:
                fallback_data["opening_weekend_decay_ratio"] = round(av_opening_weekend / av_fy, 2)
            return {"status": "error", "message": "no data returned from Gamma API",
                    **fallback_data, "source": "fallback"}

        # Compute the legs-damage ratio
        av_opening_weekend  = result.get("avengers_opening_weekend_odds")
        av_fy  = result.get("avengers_full_year_odds")
        if av_opening_weekend and av_fy and av_fy > 0:
            result["opening_weekend_decay_ratio"] = round(av_opening_weekend / av_fy, 2)

        result["status"]     = "ok"
        result["source"]     = "live"
        result["fetched_at"] = datetime.datetime.utcnow().isoformat()
        return result

    except Exception as e:
        fallback_data = dict(POLYMARKET_FALLBACK)
        av_opening_weekend = fallback_data.get("avengers_opening_weekend_odds")
        av_fy = fallback_data.get("avengers_full_year_odds")
        if av_opening_weekend and av_fy and av_fy > 0:
            fallback_data["opening_weekend_decay_ratio"] = round(av_opening_weekend / av_fy, 2)
        return {"status": "error", "message": str(e), **fallback_data, "source": "fallback"}


def calibrate_from_polymarket(av_opening_weekend_odds: float, av_fy_odds: float) -> dict:
    """
    Derive audience score adjustments and a move-signal from Polymarket odds.

    OW odds  → small Avengers audience score adjustment (crowd's OW conviction)
    FY odds  → surfaces the legs problem; NOT fed into audience score directly
               (IMAX damage is already modeled separately in core.py)
    Ratio    → opening_weekend_decay_ratio > 2.5 is a strong "market expects legs collapse" signal

    Returns:
      av_score_adj  — float, added to Avengers audience score
      move_signal   — "hold" | "neutral" | "move" | None
      notes         — human-readable interpretation
    """
    av_score_adj = 0.0
    move_signal  = None
    notes        = []

    if av_opening_weekend_odds is not None:
        if av_opening_weekend_odds >= 0.65:
            av_score_adj = +1.5
            notes.append(f"Polymarket OW: {av_opening_weekend_odds:.0%} — market confirms Avengers as dominant opener.")
        elif av_opening_weekend_odds >= 0.45:
            av_score_adj = 0.0
            notes.append(f"Polymarket OW: {av_opening_weekend_odds:.0%} — neutral OW signal.")
        else:
            av_score_adj = -2.0
            notes.append(f"Polymarket OW: {av_opening_weekend_odds:.0%} — soft OW signal, MCU fatigue priced in.")

    if av_opening_weekend_odds and av_fy_odds and av_fy_odds > 0:
        ratio = av_opening_weekend_odds / av_fy_odds
        if ratio >= 3.0:
            move_signal = "move"
            notes.append(
                f"OW/FY ratio {ratio:.1f}x — market prices a major legs collapse. "
                "Consistent with Dec 18 IMAX penalty. Move signal: STRONG."
            )
        elif ratio >= 2.0:
            move_signal = "neutral"
            notes.append(
                f"OW/FY ratio {ratio:.1f}x — market expects underperformance relative to opening. "
                "Moderate legs concern."
            )
        else:
            move_signal = "hold"
            notes.append(
                f"OW/FY ratio {ratio:.1f}x — market expects legs to hold. Hold signal."
            )

    return {
        "av_score_adj": av_score_adj,
        "move_signal":  move_signal,
        "notes":        " ".join(notes),
    }


# ── MASTER FETCH + CALIBRATE ──────────────────────────────────────────────────

def fetch_and_calibrate(base_dune_score: int = 87, base_av_score: int = 88) -> dict:
    """
    Main entry point called by Streamlit on page load.
    Returns calibrated audience scores + all signal data.
    Gracefully falls back to hardcoded values on any failure.
    """
    signals = dict(FALLBACK_SIGNALS)  # start with fallback
    av_score_adj   = 0.0
    dune_score_adj = 0.0
    sources_used = []

    # ── 1. Teaser decay calibration ───────────────────────────────────────────
    av_decay_adj = calibrate_from_teaser_decay(
        signals["avengers"]["teaser_views_x_M"]
    )
    av_score_adj += av_decay_adj
    sources_used.append("Teaser decay curve")

    # ── 2. YouTube API ────────────────────────────────────────────────────────
    dune_t1_fresh = _is_fresh_trailer(signals.get("dune", {}).get("trailer_date"))
    yt = fetch_youtube_views()
    if yt.get("status") == "ok" and yt.get("videos"):
        av_ids = {
            YOUTUBE_VIDEO_IDS.get(k)
            for k in ("avengers_t1", "avengers_t2", "avengers_t3",
                      "avengers_t4", "avengers_full")
            if YOUTUBE_VIDEO_IDS.get(k)
        }
        dune_ids = {
            YOUTUBE_VIDEO_IDS.get(k)
            for k in ("dune_t1",)
            if YOUTUBE_VIDEO_IDS.get(k)
        }
        av_yt_total = sum(
            v["views"] for vid_id, v in yt["videos"].items()
            if vid_id in av_ids
        )
        av_yt_likes = sum(
            v.get("likes", 0) for vid_id, v in yt["videos"].items()
            if vid_id in av_ids
        )
        dune_yt_total = sum(
            v["views"] for vid_id, v in yt["videos"].items()
            if vid_id in dune_ids
        )
        dune_yt_likes = sum(
            v.get("likes", 0) for vid_id, v in yt["videos"].items()
            if vid_id in dune_ids
        )
        if av_yt_total > 0:
            av_score_adj   += calibrate_from_yt_views(av_yt_total, "AVENGERS")
            sources_used.append("YouTube API")
        if dune_yt_total > 0:
            if dune_t1_fresh:
                # Day 1: use engagement ratio — can't compare hours-old views to 24h benchmarks
                _dune_eng_adj = calibrate_from_trailer_engagement(dune_yt_total, dune_yt_likes)
                # Map engagement tier → score adjustment for Dune (reuse existing thresholds)
                _dune_eng_map = {"Blockbuster": +4.0, "Strong": +2.0,
                                 "Neutral": 0.0, "Soft": -2.0, "Disappoints": -4.0}
                dune_score_adj += _dune_eng_map.get(_dune_eng_adj or "Neutral", 0.0)
            else:
                dune_score_adj += calibrate_from_yt_views(dune_yt_total, "DUNE")

        signals["avengers"]["yt_trailer_views"]    = av_yt_total or None
        signals["avengers"]["yt_trailer_likes"]    = av_yt_likes or None
        signals["avengers"]["yt_engagement_ratio"] = round(av_yt_likes / av_yt_total, 4) \
                                                     if av_yt_total else None
        signals["dune"]["yt_trailer_views"]     = dune_yt_total or None
        signals["dune"]["yt_trailer_likes"]     = dune_yt_likes or None
        signals["dune"]["yt_engagement_ratio"]  = round(dune_yt_likes / dune_yt_total, 4) \
                                                  if dune_yt_total else None
    else:
        signals["_yt_status"] = yt.get("status", "unavailable")
        signals["_yt_message"] = yt.get("message", "")

    # ── 3. Wikipedia pageviews ────────────────────────────────────────────────
    wiki = fetch_wikipedia_pageviews()
    if wiki.get("status") == "ok" and wiki.get("films"):
        av_wiki   = wiki["films"].get("avengers") or {}
        dune_wiki = wiki["films"].get("dune") or {}

        av_wiki_adj   = calibrate_from_wikipedia(
            av_wiki.get("last_7d_views"), av_wiki.get("wow_pct"), "AVENGERS"
        )
        dune_wiki_adj = calibrate_from_wikipedia(
            dune_wiki.get("last_7d_views"), dune_wiki.get("wow_pct"), "DUNE"
        )

        av_score_adj   += av_wiki_adj
        dune_score_adj += dune_wiki_adj
        sources_used.append("Wikipedia")

        signals["avengers"]["wiki_views_7d"] = av_wiki.get("last_7d_views")
        signals["avengers"]["wiki_wow_pct"]  = av_wiki.get("wow_pct")
        signals["dune"]["wiki_views_7d"]     = dune_wiki.get("last_7d_views")
        signals["dune"]["wiki_wow_pct"]      = dune_wiki.get("wow_pct")
    else:
        sources_used.append("Wikipedia (unavailable)")

    # ── 4. TMDB popularity ────────────────────────────────────────────────────
    tmdb = fetch_tmdb_signals()
    if tmdb.get("status") == "ok" and tmdb.get("films"):
        av_tmdb   = tmdb["films"].get("avengers") or {}
        dune_tmdb = tmdb["films"].get("dune") or {}

        av_tmdb_adj   = calibrate_from_tmdb(av_tmdb.get("popularity"),   "AVENGERS")
        dune_tmdb_adj = calibrate_from_tmdb(dune_tmdb.get("popularity"), "DUNE")

        av_score_adj   += av_tmdb_adj
        dune_score_adj += dune_tmdb_adj
        sources_used.append("TMDB")

        signals["avengers"]["tmdb_popularity"]   = av_tmdb.get("popularity")
        signals["avengers"]["tmdb_vote_count"]   = av_tmdb.get("vote_count")
        signals["dune"]["tmdb_popularity"]       = dune_tmdb.get("popularity")
        signals["dune"]["tmdb_vote_count"]       = dune_tmdb.get("vote_count")
    else:
        signals["_tmdb_status"]  = tmdb.get("status", "unavailable")
        signals["_tmdb_message"] = tmdb.get("message", "")

    # ── 5. Trakt collectors ───────────────────────────────────────────────────
    trakt = fetch_trakt_signals()
    if trakt.get("status") == "ok" and trakt.get("films"):
        av_trakt   = trakt["films"].get("avengers") or {}
        dune_trakt = trakt["films"].get("dune") or {}

        av_trakt_adj   = calibrate_from_trakt(av_trakt.get("collectors"),   "AVENGERS")
        dune_trakt_adj = calibrate_from_trakt(dune_trakt.get("collectors"), "DUNE")

        av_score_adj   += av_trakt_adj
        dune_score_adj += dune_trakt_adj
        sources_used.append("Trakt")

        signals["avengers"]["trakt_collectors"] = av_trakt.get("collectors")
        signals["avengers"]["trakt_watchers"]   = av_trakt.get("watchers")
        signals["avengers"]["trakt_lists"]      = av_trakt.get("lists")
        signals["dune"]["trakt_collectors"]     = dune_trakt.get("collectors")
        signals["dune"]["trakt_watchers"]       = dune_trakt.get("watchers")
        signals["dune"]["trakt_lists"]          = dune_trakt.get("lists")
    else:
        signals["_trakt_status"]  = trakt.get("status", "unavailable")
        signals["_trakt_message"] = trakt.get("message", "")

    # ── 6. Polymarket prediction market odds ──────────────────────────────────
    poly = fetch_polymarket_signals()
    poly_calibration = calibrate_from_polymarket(
        poly.get("avengers_opening_weekend_odds"),
        poly.get("avengers_full_year_odds"),
    )
    av_score_adj += poly_calibration["av_score_adj"]
    sources_used.append("Polymarket" if poly.get("source") == "live" else "Polymarket (fallback)")

    signals["polymarket"] = {
        "avengers_opening_weekend_odds":        poly.get("avengers_opening_weekend_odds"),
        "avengers_full_year_odds": poly.get("avengers_full_year_odds"),
        "dune_full_year_odds":     poly.get("dune_full_year_odds"),
        "opening_weekend_decay_ratio":          poly.get("opening_weekend_decay_ratio"),
        "move_signal":             poly_calibration["move_signal"],
        "notes":                   poly_calibration["notes"],
        "source":                  poly.get("source", "fallback"),
        "fetched_at":              poly.get("fetched_at"),
    }

    # ── 7. Spider-Man: BND trailer calibration ───────────────────────────────
    spidey_yt_id = YOUTUBE_VIDEO_IDS.get("spiderman_full")
    spidey_suggested_tier  = None
    spidey_trailer_fresh   = _is_fresh_trailer(signals.get("spiderman", {}).get("trailer_date"))
    if yt.get("status") == "ok" and spidey_yt_id and spidey_yt_id in yt.get("videos", {}):
        _spidey_vid    = yt["videos"][spidey_yt_id]
        spidey_views   = _spidey_vid["views"]
        spidey_views_M = spidey_views / 1_000_000
        spidey_likes   = _spidey_vid.get("likes", 0)
        signals["spiderman"]["yt_trailer_views_M"]  = round(spidey_views_M, 1)
        signals["spiderman"]["yt_trailer_likes"]    = spidey_likes
        signals["spiderman"]["yt_engagement_ratio"] = round(spidey_likes / spidey_views, 4) \
                                                      if spidey_views else None
        if spidey_trailer_fresh:
            # Day 1: prefer like/view ratio (time-independent), but fall back to
            # view count if likeCount is unavailable — YouTube's public API has
            # hidden likeCount since Nov 2021, so likes is often 0 via API key.
            spidey_suggested_tier = calibrate_from_trailer_engagement(spidey_views, spidey_likes)
            if spidey_suggested_tier is None:
                spidey_suggested_tier = calibrate_from_spidey_trailer(spidey_views_M)
        else:
            # Day 2+: use 24h view count benchmarks
            spidey_suggested_tier = calibrate_from_spidey_trailer(spidey_views_M)
        signals["spiderman"]["suggested_tier"] = spidey_suggested_tier
        sources_used.append("Spider-Man trailer (YouTube)")

    # ── 8. 70mm IMAX sellout signal ───────────────────────────────────────────
    imax_70mm_sold_out = signals.get("dune", {}).get("imax_70mm_sold_out", False)
    imax_sellout_adj   = calibrate_from_imax_70mm_sellout(imax_70mm_sold_out, "DUNE")
    if imax_sellout_adj != 0:
        dune_score_adj += imax_sellout_adj
        sources_used.append("70mm IMAX sellout")

    # ── 9. Final calibrated scores ────────────────────────────────────────────
    av_calibrated   = float(np.clip(base_av_score   + av_score_adj,   60, 100))
    dune_calibrated = float(np.clip(base_dune_score + dune_score_adj, 60, 100))

    # Signal confidence:
    #   high   — YouTube + Wikipedia + at least one of TMDB / Trakt
    #   medium — YouTube + Wikipedia only (or partial)
    #   low    — no live sources
    sources_set = set(sources_used)
    has_core      = {"YouTube API", "Wikipedia"}.issubset(sources_set)
    has_secondary = bool(sources_set & {"TMDB", "Trakt", "Polymarket"})
    no_fallback   = not any("fallback" in s for s in sources_used)

    if has_core and has_secondary and no_fallback:
        confidence = "high"
    elif has_core and no_fallback:
        confidence = "medium"
    elif any(s for s in sources_used if "fallback" not in s):
        confidence = "medium"
    else:
        confidence = "low"

    signals["calibration"] = {
        "avengers_base":        base_av_score,
        "dune_base":            base_dune_score,
        "avengers_adj":         round(av_score_adj, 1),
        "dune_adj":             round(dune_score_adj, 1),
        "avengers_calibrated":  av_calibrated,
        "dune_calibrated":      dune_calibrated,
        "sources":              sources_used,
        "signal_confidence":    confidence,
        "teaser_decay_signal":  "soft" if av_decay_adj < -1 else "neutral" if av_decay_adj == 0 else "strong",
        "spidey_suggested_tier": spidey_suggested_tier,
        "spidey_trailer_fresh":  spidey_trailer_fresh,
        "dune_t1_fresh":         dune_t1_fresh,
        "imax_70mm_sold_out":    imax_70mm_sold_out,
        "imax_70mm_sellout_adj": imax_sellout_adj,
        "notes": _build_notes(av_score_adj, dune_score_adj, yt, spidey_suggested_tier, imax_70mm_sold_out),
    }

    return signals


def _build_notes(av_score_adj, dune_score_adj, yt, spidey_tier=None, imax_70mm_sold_out=False) -> str:
    notes = []
    if av_score_adj < -2:
        notes.append(f"Avengers downgraded {av_score_adj:+.0f}pts — teaser decay matches Love&Thunder pattern.")
    elif av_score_adj > 2:
        notes.append(f"Avengers upgraded {av_score_adj:+.0f}pts — strong engagement signal.")
    else:
        notes.append("Avengers signal neutral — not enough signal to move the needle yet.")

    if spidey_tier:
        notes.append(f"Spider-Man: BND trailer auto-suggests '{spidey_tier}' tier for MCU brand signal.")
    else:
        notes.append("Spider-Man: BND trailer released 2026-03-18 — set YOUTUBE_VIDEO_IDS['spiderman_full'] to enable auto-calibration.")

    if imax_70mm_sold_out:
        notes.append("Dune 70mm IMAX sold out — cinephile core committed before full trailer drop (+2pts).")

    if yt and yt.get("status") != "ok":
        notes.append("Add YOUTUBE_API_KEY to Streamlit secrets for live trailer view data.")

    return " ".join(notes)
