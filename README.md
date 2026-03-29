# HackPSU_SP26

## Run locally
- Requires Python 3.10+ and Git.
- Create and activate a virtual environment:
	- `python -m venv .venv`
	- Windows: `.venv\Scripts\activate`
	- macOS/Linux: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Start the app: `streamlit run app.py`
- Optional: refresh historical data with `python build_data.py` (set `FRED_API_KEY` if you want live rent CPI; otherwise uses the baked-in fallback).
- Optional: run tests with `pytest`.

## Environment variables
- `GEMINI_API_KEY` (optional) to enable Google Generative AI features in the UI. Leave unset to disable.
- `FRED_API_KEY` (optional) only for `build_data.py` when pulling rent CPI from FRED; not required to run the app.
