# DUNESDAY — Box Office Model v5

> Dec 18, 2026: Dune Part Three vs Avengers: Doomsday  
> Monte Carlo box office simulation with real IMAX data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dunesday.streamlit.app)

## What This Is

A management science model quantifying the financial stakes of the Dec 18, 2026 box office collision between Dune: Part Three and Avengers: Doomsday — with particular focus on the IMAX negotiation that WB won before Disney arrived.

**The core finding:** WB locked all 400 US IMAX screens for 21 days (Dec 18 – Jan 7), including Christmas Day (1.65× multiplier) and New Year's Day (1.35× multiplier). Avengers opens with zero IMAX screens domestically.

## Quick Start

```bash
git clone https://github.com/yourusername/dunesday.git
cd dunesday
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
dunesday/
├── app.py                 # Streamlit dashboard (5 tabs)
├── requirements.txt
├── model/
│   ├── __init__.py
│   ├── config.py          # All assumptions — change here
│   └── core.py            # MC engine, IMAX, calendar, WOM
└── README.md
```

## Updating the Model

### Change assumptions
Edit `model/config.py`. Key sections:
- `FILM_PARAMS` — audience scores, OW estimates, budgets
- `IMAX_CONFIG` — screen allocation (confirmed, don't change unless new data)
- `HYPE_SIGNALS` — update manually as trailer data comes in

### Add a scenario
In `model/core.py`, add to the `SCENARIOS` dict:
```python
"E_New_Scenario": {
    "label": "E: Description",
    "description": "What this models",
    "dune_screens":     {"STD": 3800, "3D": 800, "IMAX": 400, "PLF": 1100},
    "avengers_screens": {"STD": 4300, "3D": 900, "IMAX": 400, "PLF": 1200},
    "imax_cfg": {...},
}
```

### Update hype signals (Feb 17 trailer drops, CinemaCon, etc.)
In `model/config.py` → `HYPE_SIGNALS`:
```python
"avengers": {
    "teaser_views": [
        {"label": "Full Trailer", "platform": "YouTube", "views_M": ???, "estimated": False},
        ...
    ],
    "full_trailer_released": True,
}
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy
4. Done — live URL in ~2 minutes

## Key Model Decisions

| Input | Value | Source |
|-------|-------|--------|
| Dune IMAX exclusive | 21 days | Empire City Box Office, Jan 2026 |
| Avengers IMAX day 1 | 0 screens | DiscussingFilm, Feb 2026 |
| Christmas Day multiplier | 1.65× | Calibrated from Avatar WoW 2022, TFA 2015 |
| Dune audience score | 87 ± 6 | Franchise curve + Alamo Drafthouse poll |
| Avengers audience score | 88 ± 7 | Teaser decay signal (T1→T2: −53% on X) |
| Avengers budget | $550M | Reported estimates |
| Dune budget | $175M | Reported estimates |

## What This Model Gets Right vs Wrong

**Right:** IMAX scarcity as binding constraint, Christmas premium, WOM mechanics, break-even analysis  
**Wrong (v4):** Assumed IMAX was negotiable — it wasn't. WB locked it first and shot with IMAX cameras.

## Next Updates

- [ ] CinemaCon trailer performance (Apr 16)
- [ ] Dune first trailer 72hr view count (Q2 2026)  
- [ ] Fandango presale ratio (Sept 2026)
- [ ] Autonomous Google Trends / YouTube scraper

## Sources

Empire City Box Office · Dark Horizons · World of Reel · The Hollywood Reporter · Collider · Alamo Drafthouse/Collider Poll · Screen Rant · Wikipedia

*Armchair analysis — no studios were consulted or harmed in the making of this model.*
