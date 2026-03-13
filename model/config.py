"""
dunesday/model/config.py
All model assumptions in one place.
Change here, everything downstream updates automatically.
"""
import datetime

# ── RELEASE ──────────────────────────────────────────────────────────────────
OPEN_DATE = datetime.date(2026, 12, 18)
DAYS = 45

# ── THEATER OPERATIONS ───────────────────────────────────────────────────────
RUNTIME = {"DUNE": 165, "AVENGERS": 150}   # minutes

TURNAROUND = {
    "STD":  20,
    "3D":   25,
    "IMAX": 30,
    "PLF":  22,
}

OPERATING_HOURS = 14 * 60  # 10am–midnight = 840 min

# ── SCREEN INVENTORY ─────────────────────────────────────────────────────────
SCREEN_INVENTORY = {"STD": 29600, "3D": 8000, "IMAX": 400, "PLF": 2000}
SEAT_CAPACITY    = {"STD": 180,   "3D": 200,  "IMAX": 285, "PLF": 220}

# ── PRICING (CPI-adjusted by region) ─────────────────────────────────────────
BASE_PRICE = {"STD": 14.10, "3D": 17.20, "IMAX": 23.50, "PLF": 19.80}

FED_REGIONS = [
    dict(region="Northeast", cpi=1.14, screen_share=0.22, occ_boost=1.08),
    dict(region="South",     cpi=1.02, screen_share=0.34, occ_boost=0.97),
    dict(region="Midwest",   cpi=0.93, screen_share=0.24, occ_boost=0.94),
    dict(region="West",      cpi=1.11, screen_share=0.20, occ_boost=1.05),
]

STUDIO_SPLIT = 0.60

# ── IMAX CONFIGURATION (v5: confirmed real-world) ─────────────────────────────
IMAX_CONFIG = {
    "dune_exclusive_days": 21,       # Dec 18 – Jan 7 (confirmed)
    "dune_screens_excl":   400,      # all US IMAX screens
    "avengers_screens_excl": 0,      # zero during exclusive window
    "split_screens":       200,      # each film after Jan 8
}

# ── FILM PARAMETERS ───────────────────────────────────────────────────────────
FILM_PARAMS = {
    "DUNE": dict(
        ow_gross_mean_M=90,   ow_gross_std_M=18,
        wk2_drop_mean=0.44,   wk2_drop_std=0.06,
        wk3_hold_mean=0.74,   wk3_hold_std=0.05,
        late_hold_mean=0.78,  late_hold_std=0.04,
        intl_mult_mean=1.48,  intl_mult_std=0.18,
        budget_M=175,         mktg_phi=0.50,
        audience_mean=87,     audience_std=6,
    ),
    "AVENGERS": dict(
        ow_gross_mean_M=240,  ow_gross_std_M=40,
        wk2_drop_mean=0.40,   wk2_drop_std=0.05,
        wk3_hold_mean=0.77,   wk3_hold_std=0.04,
        late_hold_mean=0.80,  late_hold_std=0.03,
        intl_mult_mean=2.18,  intl_mult_std=0.25,
        budget_M=550,         mktg_phi=0.50,
        audience_mean=88,     audience_std=7,
    ),
}

# ── WOM CALIBRATION ───────────────────────────────────────────────────────────
# Linear fit: wom_mult = slope * audience_score + intercept
# Calibrated from: Endgame, IW, Top Gun, Dune P2, The Flash, Black Adam, DS MoM
WOM_SLOPE     = 0.0199
WOM_INTERCEPT = -0.7748

# ── CALENDAR HOLIDAY MULTIPLIERS ─────────────────────────────────────────────
# Applied on top of day-of-week baseline
HOLIDAY_OVERRIDES = {
    (12, 24): 1.25,   # Christmas Eve
    (12, 25): 1.65,   # Christmas Day
    (12, 26): 1.40,   # Day after Christmas
    (12, 27): 1.20,
    (12, 31): 0.95,   # NYE (people go out, not theaters)
    (1,  1):  1.35,   # New Year's Day
    (1,  2):  1.05,
}

DOW_MULTIPLIERS = {
    0: 0.55,   # Monday
    1: 0.55,   # Tuesday
    2: 0.55,   # Wednesday
    3: 0.65,   # Thursday
    4: 1.35,   # Friday
    5: 1.10,   # Saturday
    6: 1.10,   # Sunday
}

# ── HYPE SIGNALS (update manually until scraper is built) ────────────────────
HYPE_SIGNALS = {
    "last_updated": "2026-02-26",
    "avengers": {
        "teaser_views": [
            {"label": "T1 Steve Rogers", "platform": "X", "views_M": 53.0, "estimated": False},
            {"label": "T2 Thor",         "platform": "X", "views_M": 25.0, "estimated": False},
            {"label": "T3 X-Men",        "platform": "X", "views_M": 15.0, "estimated": True},
            {"label": "T4 Wakanda",      "platform": "X", "views_M":  9.0, "estimated": True},
        ],
        "combined_views_B": 1.02,
        "instagram_M": 505,
        "tiktok_M": 103,
        "social_vol_vs_avg_pct": 188,
        "full_trailer_released": False,
    },
    "dune": {
        "teaser_views": [],
        "combined_views_B": 0,
        "full_trailer_released": False,
        "alamo_poll_rank": 1,
        "alamo_poll_respondents": 14000,
    },
    "cinemacon_date": "2026-04-13",
    "disney_presentation_date": "2026-04-16",
    "wb_presentation_date": "2026-04-14",
}
