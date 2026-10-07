# Next steps: extending the year coverage (handoff for a new session)

Written 2026-10-07. Paste or point a new Claude session at this file to continue.

## Why this matters
Firm accounts cover only fiscal 2022–2025 (Yahoo). That is too short for shock designs (2008, euro crisis, COVID, the 2022 energy shock all lack a pre-period), for within-company change analysis (H9, H10 came out as "not supported" partly for this reason) and for the planned notebook **3.2** (energy-shock comparison of companies inside the same country: sector energy intensity × country energy-price shock).

## What the data covers today
| Source | Years | Notes |
|---|---|---|
| Yahoo (`source_id = 'YF'`) | fiscal 2022–2025 (2021 has only 89 rows) | yfinance returns about four annual statements; hard limit |
| ESEF (`'ESEF'`) | 2019–2025 | 70 companies in 2019, 626 in 2020, about 930–1,080 in 2021–2024; filings only became mandatory around 2020; 20 EU countries, no sector, no DEU/IRL |
| Country panel | 2012 onward | start year is a fetch setting, not a source limit |
| Prices (`data/flat/prices_monthly.parquet`) | long | market returns only, not accounting ROA |

## Options, in recommended order

### 1. Country-and-sector panels from free official sources (recommended first)
- BACH (Banque de France: aggregated company accounts by country and sector), Eurostat structural business statistics and national accounts, EU KLEMS (sector productivity, back to the 1990s), possibly OECD STAN.
- Not company-level, but gives ROA-like ratios by country × sector over 20+ years.
- Enough to run the shift-share design (sector exposure × country shock) across 2008, the euro crisis, COVID and 2022, and it avoids survivorship bias.
- First step: a feasibility check. Which of the 34 countries and which sectors are covered, from which year, and can the sectors be mapped to the Yahoo sector list (11 sectors)?
- Keep it outside the Postgres schema (the user owns the schema), like `data/flat/*.parquet` and `src/fetch/extra_indicators.py`.

### 2. Extend the country panel back (cheap)
- World Bank, IMF and WGI go back to the 1990s; change the start year in the fetch scripts (`src/fetch/`), keep raw data in `data/raw`.
- Eurostat series have a 2020/2021 SBS break: keep series separate, as agreed.

### 3. SEC EDGAR for US companies
- Free XBRL "company facts" API, about 2009 onward. US only (1 of 34 countries), so it supports company-level patterns over time, not cross-country ones.

### 4. Paid or university sources (check first whether the user has access)
- Compustat Global via WRDS, or Orbis: 20+ years of company accounts for all countries. Best option if the user has library access.
- EOD Historical Data: about €20 a month, about 30 years of fundamentals; verify coverage of the 34 countries before paying.
- Free APIs (Alpha Vantage, SimFin, Financial Modeling Prep) mostly limit history or non-US coverage.

### 5. Market-based outcomes over longer periods
- yfinance prices go back decades: returns and volatility could cover long windows (extension of H8). Stock performance, not ROA.

## Caveat for any company-level extension
The company list is today's listings, so firms that failed or were delisted earlier are missing and older years look healthier than they were (survivorship bias). Say this in any notebook that uses older firm data. The country-and-sector panels (option 1) do not have this problem.

## Project state to build on
- Notebooks: 0.x (data), 1.0–1.16 (hypotheses H0–H16), 2.1 (EU vs non-EU gap, AB1), 2.2 (compare-two-groups tool prototype), 3.1 (companies inside countries, W1 and W2). Naming rule from the user: keep the dotted numbers.
- Shared code: `src/analysis/prep.py` (data prep), `utils.py` (cluster regression, wild bootstrap, BH, verdict), `compare.py` (group comparison), `within.py` (within-country slopes, random-effects pooling, sector profile).
- Results registry: `results/H*.json` (H0–H16, the 15-test BH family), `results/AB1.json`, `results/W1.json`, `results/W2.json` (kept outside the correction family on purpose).
- Dashboard: `app/dash_app.py`, rebuild its data with `uv run --group dash python app/build_extract.py`. It has a "Company level" tab built from W1, W2, the sector ranking and AB1.
- Main results so far: development → ROA about −1.1 pp per SD (H0, small firms carry it, H13); equity ratio → ROA +1.2 pp per +0.1 inside 27 of 28 countries (W1); company size has no consistent pattern (W2); EU companies earn about 2.3 pp less ROA than non-EU (descriptive only).

## Working rules (from the user)
- `uv` for Python, user runs install commands; Postgres → DuckDB (`data/gbc.duckdb`); the user designs the DB schema, so new data goes in `data/flat/` or `data/raw/`.
- Do not re-add the dropped countries (BGR, HRV, CYP, SVN, SVK, MLT).
- Results are associations, not causation; label exploratory findings; keep verdicts rule-based and saved in `results/`.
- Never add a Claude/Anthropic co-author trailer to commit messages.
- Notebook generators live in the session scratchpad, not in the repo.

## Suggested first message in the new session
"Read next_steps_extend_year_coverage.md. Start with option 1: check which of our 34 countries and sectors BACH, Eurostat and EU KLEMS cover, from which year, and whether the sectors map onto our 11 Yahoo sectors. Report before downloading anything large."

## Status update (2026-10-07, option 1 built)
- `src/fetch/sector_panels.py` -> `data/flat/sector_country_year.parquet` (Eurostat, 21 EU countries, NACE A64, 1975/1995-2025: value added, net operating surplus, compensation, net fixed assets at current replacement cost).
- `src/fetch/stan_panels.py` -> `data/flat/sector_country_year_stan.parquet` (OECD STAN 2025, non-EU, current prices, no capital stock). Net operating surplus only for AUS, JPN, KOR, USA (KOR and JPN thin); CAN, CHL, MEX have value added and compensation only. BRA, CHN, IND, SGP, IDN, ZAF are not in STAN; GGDC 10-sector has them but stops in 2011/2012 and has no profits, so it was not used.
- `src/analysis/sector_panel.py`: `build()` maps to the 11 Yahoo sectors (margin = net operating surplus / value added) -> `sector_yahoo_country_year.parquet`; `build_sections()` letter-level panel (includes `ror` = net operating surplus / net fixed assets, only 21 EU countries) -> `sector_section_country_year.parquet`. Financials give odd ror (EST K above 100%): exclude sector K from return-on-capital work.
- Sanity check: Germany manufacturing ror 8% (2003) -> 17% (2007) -> 3.5% (2009) matches the known profit boom and crash.
- Option 4: no WRDS/Orbis/EOD credentials on the machine. EOD demo token returned 41 years of statements for AAPL (back to 1985), but country coverage for our 34 countries cannot be checked without a key. Waiting on the user's answer about university access (WRDS Compustat Global / Orbis).
