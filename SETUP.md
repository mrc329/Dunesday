# SETUP.md — Getting This on GitHub from iPad

## Option 1: GitHub Web UI (easiest on iPad)

1. Go to github.com → New repository → name it `dunesday`
2. Set to Public (required for free Streamlit deployment)
3. Upload files one at a time via the web interface:
   - Upload `app.py`
   - Upload `requirements.txt`  
   - Upload `README.md`
   - Create folder `model/` → upload `__init__.py`, `config.py`, `core.py`

## Option 2: Working Copy app (iPad Git client)

1. Install Working Copy from App Store (~$20 one-time)
2. Download the zip of this repo from Claude
3. Import into Working Copy → push to GitHub

## Option 3: iSH Shell (free terminal on iPad)

1. Install iSH from App Store
2. `apk add git python3`
3. `git clone` or create repo and push

---

## Streamlit Deployment (2 minutes after GitHub)

1. Go to share.streamlit.io
2. Sign in with GitHub
3. "New app" → select `dunesday` repo → main branch → `app.py`
4. Click Deploy
5. You get a live URL: `https://dunesday.streamlit.app`

## Working with Claude Code

Once on GitHub, in Claude Code:
```bash
git clone https://github.com/yourusername/dunesday
cd dunesday
pip install -r requirements.txt
streamlit run app.py
```

To update model assumptions:
```
"Update FILM_PARAMS in model/config.py — set Avengers audience_mean to 85"
```

To add a new scenario after CinemaCon:
```
"Add scenario E to SCENARIOS in model/core.py — Disney moves to May 1 with 
full IMAX, model the calendar multiplier difference vs December"
```
