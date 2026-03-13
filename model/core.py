"""
dunesday/model/core.py
Monte Carlo engine, calendar, IMAX, WOM functions.
Pure Python — no Streamlit dependencies.
"""
import datetime
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from model.config import (
    OPEN_DATE, DAYS, RUNTIME, TURNAROUND, OPERATING_HOURS,
    SCREEN_INVENTORY, SEAT_CAPACITY, BASE_PRICE, FED_REGIONS,
    STUDIO_SPLIT, IMAX_CONFIG, FILM_PARAMS,
    WOM_SLOPE, WOM_INTERCEPT,
    HOLIDAY_OVERRIDES, DOW_MULTIPLIERS,
)


# ── THEATER OPERATIONS ────────────────────────────────────────────────────────

def shows_per_day(film: str, fmt: str) -> int:
    slot = RUNTIME[film] + TURNAROUND[fmt]
    return int(OPERATING_HOURS / slot)


# ── CALENDAR ─────────────────────────────────────────────────────────────────

def build_calendar_multipliers(n_days: int = DAYS) -> np.ndarray:
    """Day-by-day demand multiplier for Dec 18 opener."""
    multipliers = []
    for d in range(n_days):
        date = OPEN_DATE + datetime.timedelta(days=d)
        dow  = date.weekday()
        mo, dt = date.month, date.day
        m = DOW_MULTIPLIERS.get(dow, 0.55)
        key = (mo, dt)
        if key in HOLIDAY_OVERRIDES:
            m = HOLIDAY_OVERRIDES[key]
        elif mo == 12 and 28 <= dt <= 30:
            m = max(m, 0.75)
        multipliers.append(m)
    return np.array(multipliers)


CAL_MULT = build_calendar_multipliers()


# ── WORD OF MOUTH ─────────────────────────────────────────────────────────────

def wom_mult(audience_score: float) -> float:
    return max(0.5, WOM_SLOPE * audience_score + WOM_INTERCEPT)


# ── IMAX REVENUE ─────────────────────────────────────────────────────────────

IMAX_DAILY_BASE_M = 4.9   # $M/day at 400 screens, mean occupancy, no cal mult

def compute_imax_revenue(
    film: str,
    wom_mult_val: float = 1.0,
    rng: np.random.Generator = None,
    cfg: dict = None,
) -> dict:
    if cfg is None:
        cfg = IMAX_CONFIG
    if rng is None:
        rng = np.random.default_rng(42)

    excl_days = cfg["dune_exclusive_days"]
    decay_holds = [1.0, 0.56, 0.43, 0.34, 0.27, 0.22, 0.18]
    daily = []

    for day in range(DAYS):
        if film == "DUNE":
            screens = cfg["dune_screens_excl"] if day < excl_days else cfg["split_screens"]
        else:
            screens = cfg["avengers_screens_excl"] if day < excl_days else cfg["split_screens"]

        cal = CAL_MULT[day]
        dk  = decay_holds[min(day // 7, 6)]
        rev = IMAX_DAILY_BASE_M * (screens / 400) * cal * dk * wom_mult_val
        daily.append(rev)

    daily_arr = np.array(daily)
    return {
        "daily":    daily_arr,
        "total":    daily_arr.sum(),
        "excl_rev": daily_arr[:excl_days].sum(),
        "post_rev": daily_arr[excl_days:].sum(),
    }


# ── IMAX GAP SUMMARY ─────────────────────────────────────────────────────────

def imax_gap_summary() -> dict:
    dune_imax = compute_imax_revenue("DUNE")
    av_imax   = compute_imax_revenue("AVENGERS")
    return {
        "dune_total":        dune_imax["total"],
        "avengers_total":    av_imax["total"],
        "gap":               dune_imax["total"] - av_imax["total"],
        "dune_excl_rev":     dune_imax["excl_rev"],
        "avengers_excl_rev": av_imax["excl_rev"],
        "dune_daily":        dune_imax["daily"],
        "avengers_daily":    av_imax["daily"],
        "xmas_day_dune":     float(dune_imax["daily"][7]),
        "xmas_day_avengers": float(av_imax["daily"][7]),
        "ny_day_dune":       float(dune_imax["daily"][14]),
        "ny_day_avengers":   float(av_imax["daily"][14]),
    }


# ── MONTE CARLO ──────────────────────────────────────────────────────────────
# Scenario-level OW adjustments (vs contested Dec 18 baseline)
SCENARIO_OW_ADJ = {
    # (film, scenario_key): multiplier on ow_gross_mean_M
    ("DUNE",     "A_Both_Hold"):  1.00,
    ("DUNE",     "B_Disney_May"): 1.15,   # uncontested Dec 18, full IMAX
    ("DUNE",     "C_Disney_Jan"): 1.12,
    ("DUNE",     "D_WB_Moves"):   0.78,   # Dune moves to Jan — loses Christmas
    ("AVENGERS", "A_Both_Hold"):  1.00,
    ("AVENGERS", "B_Disney_May"): 1.22,   # uncontested May, full IMAX, no cannibalization
    ("AVENGERS", "C_Disney_Jan"): 1.05,   # Jan — post-holiday, some IMAX
    ("AVENGERS", "D_WB_Moves"):   1.18,   # Avengers uncontested Dec 18
}


def run_monte_carlo(
    film: str,
    scenario_key: str = "A_Both_Hold",
    n: int = 5000,
    seed: int = 42,
    audience_override: float = None,
    intl_override: float = None,
    imax_cfg: dict = None,
) -> dict:
    """
    Monte Carlo simulation for a single film in a single scenario.
    Returns P10/P50/P90 net profit, break-even %, IMAX revenue mean.
    """
    rng = np.random.default_rng(seed)
    p   = FILM_PARAMS[film]
    if imax_cfg is None:
        imax_cfg = IMAX_CONFIG

    aud_mean  = audience_override if audience_override is not None else p["audience_mean"]
    intl_mean = intl_override     if intl_override     is not None else p["intl_mult_mean"]
    ow_adj    = SCENARIO_OW_ADJ.get((film, scenario_key), 1.0)

    revenues  = []
    imax_revs = []

    # Weekly decay holds
    wk_holds_base = [1.0, 0.56, 0.43, 0.34, 0.27, 0.22, 0.18]

    for trial in range(n):
        # Sample audience score → WOM multiplier
        aud_score = rng.normal(aud_mean, p["audience_std"])
        wm = np.clip(wom_mult(aud_score), 0.5, 1.5)

        # Opening weekend draw
        ow_gross = rng.normal(p["ow_gross_mean_M"] * ow_adj, p["ow_gross_std_M"] * ow_adj)
        ow_gross = max(ow_gross, p["ow_gross_mean_M"] * ow_adj * 0.3)

        # 45-day domestic gross with calendar + decay
        dom_gross_M = 0.0
        for day in range(DAYS):
            wk  = min(day // 7, len(wk_holds_base) - 1)
            cal = CAL_MULT[day]
            # WoM adjusts hold rates from week 2 onward
            hold = wk_holds_base[wk]
            if wk >= 1:
                hold = hold * np.clip(wm, 0.6, 1.4)
                hold = np.clip(hold, 0.05, 1.05)
            dom_gross_M += ow_gross * hold * cal / 7.0

        dom_studio_M = dom_gross_M * STUDIO_SPLIT

        # IMAX revenue
        imax = compute_imax_revenue(film, wm, rng, cfg=imax_cfg)
        imax_studio_M = imax["total"] * STUDIO_SPLIT
        imax_revs.append(imax["total"])

        # International
        intl_mult  = rng.normal(intl_mean, p["intl_mult_std"])
        intl_mult  = max(0.5, intl_mult)
        intl_rev_M = dom_gross_M * intl_mult * STUDIO_SPLIT

        # Total & net profit
        total_studio_M = dom_studio_M + imax_studio_M + intl_rev_M
        cost_M = p["budget_M"] * (1 + p["mktg_phi"])
        revenues.append(total_studio_M - cost_M)

    arr = np.array(revenues)
    return {
        "profits":       arr,
        "p10":           float(np.percentile(arr, 10)),
        "p50":           float(np.percentile(arr, 50)),
        "p90":           float(np.percentile(arr, 90)),
        "mean":          float(arr.mean()),
        "breakeven_pct": float((arr > 0).mean() * 100),
        "imax_rev_mean": float(np.mean(imax_revs)),
    }


# ── SCENARIOS ─────────────────────────────────────────────────────────────────

SCENARIOS = {
    "A_Both_Hold": {
        "label":       "A: Both Hold Dec 18 (Current)",
        "description": "Dune gets 3-week IMAX exclusive. Avengers gets zero IMAX opening weekend.",
        "imax_cfg":    IMAX_CONFIG,
    },
    "B_Disney_May": {
        "label":       "B: Disney Moves to May 1",
        "description": "Avengers gets full IMAX uncontested. Loses Christmas premium.",
        "imax_cfg":    {**IMAX_CONFIG,
                        "dune_exclusive_days": 0,
                        "dune_screens_excl": 400,
                        "avengers_screens_excl": 400},
    },
    "C_Disney_Jan": {
        "label":       "C: Disney Moves to Jan 16",
        "description": "Avengers opens after Dune's exclusive expires. Partial IMAX, no Christmas.",
        "imax_cfg":    {**IMAX_CONFIG,
                        "dune_exclusive_days": 0,
                        "avengers_screens_excl": 200},
    },
    "D_WB_Moves": {
        "label":       "D: WB Moves Dune (Jan 2027)",
        "description": "Avengers uncontested Dec 18, full IMAX. Dune loses Christmas window.",
        "imax_cfg":    {**IMAX_CONFIG,
                        "dune_exclusive_days": 0,
                        "dune_screens_excl": 0,
                        "avengers_screens_excl": 400},
    },
}


def run_all_scenarios(
    n: int = 2000,
    seed: int = 42,
    dune_aud: float = None,
    av_aud: float = None,
    dune_intl: float = None,
    av_intl: float = None,
) -> dict:
    results = {}
    for sk, sc in SCENARIOS.items():
        results[sk] = {}
        for film in ["DUNE", "AVENGERS"]:
            aud  = dune_aud  if film == "DUNE" else av_aud
            intl = dune_intl if film == "DUNE" else av_intl
            results[sk][film] = run_monte_carlo(
                film,
                scenario_key=sk,
                n=n,
                seed=seed,
                audience_override=aud,
                intl_override=intl,
                imax_cfg=sc["imax_cfg"],
            )
    return results
