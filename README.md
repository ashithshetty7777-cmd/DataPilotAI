# DataPilot AI

AI-powered analytics platform built with Streamlit, Pandas and Plotly. Free tools only; AI uses the Google Gemini cloud API
(no local LLMs). Runs comfortably on an 8 GB laptop.

## Run (Windows / VS Code)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your Gemini key into .env (optional)
streamlit run app.py
```
Try `sample_sales.csv` (30 rows) and `sample_sales_2years.csv` (400 rows - long enough for forecasting).

## Sections
Dashboard · Datasets · AI Analyst · Data Cleaning · AI Chat · SQL Assistant · Python Assistant · Forecasting · Export Center

## How "no invented numbers" is enforced
* All figures come from pandas (`core/facts.py`). The UI labels them **CALCULATED**.
* Gemini only receives schema + those computed facts (raw rows only if you tick the sample-rows box) and is instructed to use
  nothing else. Its output is labelled **AI-GENERATED**.
* Questions that name a specific city/product/date are not answered from unfiltered totals - use SQL/Python for those.

## Gemini notes
Pick your model from the live list the app fetches with your key (model names and free-tier quotas change; per-model limits are
shown in Google AI Studio). Free-tier prompts may be used by Google to improve its products - don't send sensitive data.
Never commit `.env` (it is git-ignored).

## Tests
`pip install pytest` then `python -m pytest -q tests` (covers loading, KPIs, facts, cleaning, SQL safety, Python sandbox,
forecast, AI parsing with a mocked network, exports).

## Known limits
* Python Assistant uses an AST allow-list on a data copy - fine for a local single-user app, not a hardened sandbox.
* Chart export is interactive HTML (PNG needs the heavy `kaleido` package). PDF has tables + text, no chart images.
* Forecast = linear trend (+ monthly seasonality with 24+ months); it refuses < 12 periods of history.
