# Global Business Comparison: Full Build Plan

**Question:** How do the social and economic conditions of a country relate to the performance of the companies that operate there?

**Scope:** 40 countries (27 EU members and 13 countries outside Europe), several free APIs, a database you design yourself, two dashboards (raw data and analysis), and an optional ML "what-if" model.

---

## 0. The big picture

```
APIs (World Bank, WGI, IMF, Eurostat, OECD, company data)
        │
        ▼
data/raw/          ← untouched API responses (JSON/CSV), saved once
        │
        ▼
staging tables     ← one clean, long table per source
        │
        ▼
core tables        ← your schema: countries, indicators, values, companies
        │
        ▼
analysis views     ← wide tables, joined and lagged features
        │
        ├──► Dashboard 1: Raw data explorer
        ├──► Dashboard 2: Insights (filtered, correlated)
        └──► (stretch) ML what-if model page
```

There are two kinds of data, and the whole project depends on joining them:

1. **Country data (the "causes")**: economic, social, institutional and infrastructure indicators per country per year.
2. **Company data (the "outcome")**: how firms perform, measured with ratios such as return on assets, operating margin and revenue growth. Ratios are used because they don't depend on currency.

**Honest framing to use in the report:** this project measures *association*, not causation. Company performance depends mostly on the firm and its sector. Country effects exist but are smaller, and large listed companies often earn most of their money abroad. Mentioning this in your defense shows maturity, not weakness.

---

## 1. Country list (40)

You need three code columns because each API uses different codes. Store all three in your country table.

### 1.1 EU-27

| Country | ISO3 (World Bank, IMF) | ISO2 | Eurostat code | Euro | OECD |
|---|---|---|---|---|---|
| Austria | AUT | AT | AT | Yes | Yes |
| Belgium | BEL | BE | BE | Yes | Yes |
| Bulgaria | BGR | BG | BG | Yes (2026) | No |
| Croatia | HRV | HR | HR | Yes | No (accession) |
| Cyprus | CYP | CY | CY | Yes | No |
| Czechia | CZE | CZ | CZ | No | Yes |
| Denmark | DNK | DK | DK | No | Yes |
| Estonia | EST | EE | EE | Yes | Yes |
| Finland | FIN | FI | FI | Yes | Yes |
| France | FRA | FR | FR | Yes | Yes |
| Germany | DEU | DE | DE | Yes | Yes |
| Greece | GRC | GR | **EL** | Yes | Yes |
| Hungary | HUN | HU | HU | No | Yes |
| Ireland | IRL | IE | IE | Yes | Yes |
| Italy | ITA | IT | IT | Yes | Yes |
| Latvia | LVA | LV | LV | Yes | Yes |
| Lithuania | LTU | LT | LT | Yes | Yes |
| Luxembourg | LUX | LU | LU | Yes | Yes |
| Malta | MLT | MT | MT | Yes | No |
| Netherlands | NLD | NL | NL | Yes | Yes |
| Poland | POL | PL | PL | No | Yes |
| Portugal | PRT | PT | PT | Yes | Yes |
| Romania | ROU | RO | RO | No | No |
| Slovakia | SVK | SK | SK | Yes | Yes |
| Slovenia | SVN | SI | SI | Yes | Yes |
| Spain | ESP | ES | ES | Yes | Yes |
| Sweden | SWE | SE | SE | No | Yes |

Verify the euro and OECD columns before you present. Membership changes, and they matter as filters in your dashboard.

### 1.2 Outside Europe (13), a suggested mix of regions and income levels

| Country | ISO3 | ISO2 | Region | OECD |
|---|---|---|---|---|
| United States | USA | US | North America | Yes |
| Canada | CAN | CA | North America | Yes |
| Mexico | MEX | MX | Latin America | Yes |
| Brazil | BRA | BR | Latin America | No |
| Chile | CHL | CL | Latin America | Yes |
| Japan | JPN | JP | East Asia | Yes |
| South Korea | KOR | KR | East Asia | Yes |
| China | CHN | CN | East Asia | No |
| India | IND | IN | South Asia | No |
| Singapore | SGP | SG | Southeast Asia | No |
| Indonesia | IDN | ID | Southeast Asia | No |
| Australia | AUS | AU | Oceania | Yes |
| South Africa | ZAF | ZA | Africa | No |

You can swap any of these, but avoid Taiwan, because the World Bank does not publish data for it. A different region mix will also change your EU vs non-EU results, so pick on purpose and explain why.

**Coverage consequence:** the World Bank, WGI and IMF cover all 40 countries, so they are your **core layer**. Eurostat covers only the EU-27, so it is your **Europe extension layer**. The OECD covers about 29 of the 40, so it is **optional enrichment**. Any analysis across all 40 countries must use core-layer variables only.

---

## 2. APIs and exactly what to query

Time window: **2012–2024** (13 years). Recent years will have gaps because of publication lag, which is expected.

### 2.1 World Bank, World Development Indicators (core, no key)

**Endpoint**
```
https://api.worldbank.org/v2/country/{ISO3;ISO3;...}/indicator/{CODE}?format=json&date=2012:2024&per_page=1000
```
- Separate countries with semicolons: `SWE;PRT;USA;...`. All 40 fit in one call per indicator.
- The response is a list `[metadata, rows]`. Check `metadata["pages"]` and loop over `&page=N` if there is more than one page.
- Python shortcut: `pip install wbgapi`, then `wb.data.DataFrame(codes, economy=countries, time=range(2012, 2025))`.

**Indicators to pull (about 30)**

| Theme | Code | Indicator |
|---|---|---|
| Economy | NY.GDP.MKTP.CD | GDP (current US$) |
| Economy | NY.GDP.PCAP.PP.KD | GDP per capita, PPP (constant intl $) |
| Economy | NY.GDP.MKTP.KD.ZG | GDP growth (annual %) |
| Economy | FP.CPI.TOTL.ZG | Inflation, consumer prices (annual %) |
| Economy | NV.IND.MANF.ZS | Manufacturing, value added (% of GDP) |
| Economy | NV.SRV.TOTL.ZS | Services, value added (% of GDP) |
| Trade & investment | NE.TRD.GNFS.ZS | Trade (% of GDP) |
| Trade & investment | BX.KLT.DINV.WD.GD.ZS | FDI, net inflows (% of GDP) |
| Trade & investment | TX.VAL.TECH.MF.ZS | High-tech exports (% of manufactured exports) |
| Finance | FS.AST.PRVT.GD.ZS | Domestic credit to private sector (% of GDP) |
| Finance | FR.INR.LEND | Lending interest rate (%) |
| Finance | CM.MKT.LCAP.GD.ZS | Market cap of listed domestic companies (% of GDP) |
| Finance | CM.MKT.LDOM.NO | Number of listed domestic companies |
| Business | IC.BUS.NDNS.ZS | New business density (new registrations per 1,000 people aged 15–64) |
| Business | IC.TAX.TOTL.CP.ZS | Total tax and contribution rate (% of profit). *Series ends around 2019* |
| Business | IC.LGL.CRED.XQ | Strength of legal rights index. *Series ends around 2019* |
| Fiscal | GC.TAX.TOTL.GD.ZS | Tax revenue (% of GDP) |
| Labour | SL.UEM.TOTL.ZS | Unemployment (% of labour force, ILO model) |
| Labour | SL.UEM.1524.ZS | Youth unemployment (% of ages 15–24) |
| Labour | SL.TLF.CACT.ZS | Labour force participation (% of ages 15+) |
| Labour | SL.EMP.SELF.ZS | Self-employed (% of employment) |
| Social | SP.POP.TOTL | Population |
| Social | SP.POP.1564.TO.ZS | Population aged 15–64 (%) |
| Social | SP.POP.65UP.TO.ZS | Population aged 65+ (%) |
| Social | SP.URB.TOTL.IN.ZS | Urban population (%) |
| Social | SP.DYN.LE00.IN | Life expectancy at birth |
| Social | SI.POV.GINI | Gini index (inequality). *Sparse; not every year* |
| Education & innovation | SE.TER.ENRR | Tertiary school enrolment (% gross) |
| Education & innovation | SE.XPD.TOTL.GD.ZS | Government education spending (% of GDP) |
| Education & innovation | GB.XPD.RSDV.GD.ZS | R&D expenditure (% of GDP) |
| Education & innovation | IP.PAT.RESD | Patent applications, residents |
| Infrastructure | IT.NET.USER.ZS | Internet users (% of population) |
| Infrastructure | IT.CEL.SETS.P2 | Mobile subscriptions (per 100 people) |

Before writing the fetch code, check every code at `https://data.worldbank.org/indicator/{CODE}`. Some business indicators came from the discontinued Doing Business project and stop around 2019–2020. Use them only as "latest available" snapshots, never as a time series.

### 2.2 Worldwide Governance Indicators (core, no key)

These are the best free measures of institutional quality, which matters a lot for business. They are served by the same World Bank API under **source 3**.

```
https://api.worldbank.org/v2/country/{ISO3;...}/indicator/{CODE}?source=3&format=json&date=2012:2024&per_page=1000
```
With wbgapi: `wb.data.DataFrame(codes, economy=countries, db=3)`

| Code | Dimension |
|---|---|
| GE.EST | Government effectiveness |
| RQ.EST | Regulatory quality |
| RL.EST | Rule of law |
| CC.EST | Control of corruption |
| VA.EST | Voice and accountability |
| PV.EST | Political stability / absence of violence |

The scale runs from about −2.5 to +2.5, where higher is better.

### 2.3 IMF DataMapper, World Economic Outlook (core, no key)

This source fills recent-year gaps and adds government debt and the current account.

```
https://www.imf.org/external/datamapper/api/v1/{INDICATOR}/{ISO3}/{ISO3}/...
```
Example: `.../api/v1/GGXWDG_NGDP/SWE/PRT/USA`

| Code | Indicator |
|---|---|
| NGDP_RPCH | Real GDP growth (%) |
| PCPIPCH | Inflation, average consumer prices (%) |
| LUR | Unemployment rate (%) |
| GGXWDG_NGDP | General government gross debt (% of GDP) |
| BCA_NGDPD | Current account balance (% of GDP) |
| NGDPDPC | GDP per capita, current US$ |

The response looks like `{"values": {CODE: {ISO3: {year: value}}}}`. It also includes **projections** for future years, so drop any year later than the last actual year.

### 2.4 Eurostat (Europe layer, EU-27 only, no key)

```
https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{DATASET}?geo=SE&geo=PT&...&sinceTimePeriod=2012
```
- The response is in JSON-stat format. `pip install eurostat` and `eurostat.get_data_df("nama_10_pc")` turn it into a DataFrame.
- Use the Eurostat code **EL** for Greece. Drop aggregate rows such as `EU27_2020` and `EA20`, or keep them only as benchmark lines in charts.
- Every dataset has extra dimensions (`unit`, `na_item`, `nace_r2`...). Filter each one to a single value, or you will get duplicate rows.

| Dataset | Content | Why it matters for companies |
|---|---|---|
| nama_10_pc | GDP per capita in PPS | Comparable purchasing power |
| une_rt_a | Unemployment, annual | Labour availability |
| prc_hicp_aind | Harmonised inflation (HICP) | Cost pressure |
| lc_lci_lev | Labour cost levels (€ per hour) | Direct operating cost |
| earn_mw_cur | Statutory minimum wages | Cost floor |
| rd_e_gerdtot | R&D spending by sector | Innovation environment |
| isoc_ci_ifp_iu | Internet use by individuals | Digital customers |
| `isoc_e*` tables | Digital adoption by enterprises (e-commerce, cloud, AI) | Firm digitalisation |
| `sbs_*` tables | Structural business statistics: turnover, value added, employees by sector (NACE) | **Country × sector business outcomes** |
| `bd_*` tables | Business demography: firm birth, death and survival rates | **Business dynamism** |

Find the exact current codes for the `sbs_`, `bd_` and `isoc_e` tables in the Eurostat Data Browser (search "enterprise statistics by size class", "business demography", or "enterprises use of AI"). Eurostat renames these when methodology changes.

### 2.5 OECD Data Explorer (optional, about 29 of 40 countries, no key)

```
https://sdmx.oecd.org/public/rest/data/{AGENCY},{DATAFLOW},{VERSION}/{KEY}?startPeriod=2012&format=csvfilewithlabels
```
- Don't build these URLs by hand. Open a dataset on `data-explorer.oecd.org`, set your filters, then click **Developer API** and copy the query URL.
- Useful themes: corporate tax rates, structural and demographic business statistics, labour productivity, average wages.
- There is a strict **rate limit**. Download once, cache, and never call it inside a loop that runs on every dashboard refresh.

### 2.6 Company performance data (the outcome variable)

There are two levels. Do level A for sure, and level B if you want the ML model.

**A. Country-level business outcomes (no extra API needed).** Use World Bank `IC.BUS.NDNS.ZS`, `CM.MKT.LCAP.GD.ZS` and `CM.MKT.LDOM.NO`, plus Eurostat `sbs_` and `bd_` for the EU. This answers questions like "do countries with better rule of law have more new businesses?"

**B. Firm-level data from listed companies, using `yfinance` (no key, unofficial).**
1. Build `companies_seed.csv` by hand: about 15 large listed companies per country, taken from each country's main stock index (OMXS30 for Sweden, PSI for Portugal, S&P 500 for the US...). Columns: `ticker, country_iso3`.
   - Yahoo suffixes: `.ST` Stockholm, `.LS` Lisbon, `.DE` Xetra, `.PA` Paris, `.AS` Amsterdam, `.MI` Milan, `.MC` Madrid, `.T` Tokyo, `.KS` Korea, `.SS`/`.SZ` China, `.NS` India, `.SI` Singapore, `.AX` Australia, `.JO` Johannesburg, `.SA` Brazil, `.MX` Mexico, `.SN` Chile, `.TO` Toronto, `.JK` Jakarta.
2. For each ticker, pull:
   - `yf.Ticker(t).info`: `sector`, `industry`, `country`, `currency`, `marketCap`
   - `.income_stmt` (annual): Total Revenue, Operating Income, Net Income
   - `.balance_sheet` (annual): Total Assets, Stockholders Equity
3. Compute **ratios** (these don't depend on currency):
   - ROA = Net Income / Total Assets
   - Operating margin = Operating Income / Revenue
   - Revenue growth = Revenue_t / Revenue_t−1 − 1
4. Add `time.sleep(1)` between tickers, wrap every call in try/except, and save raw results immediately. yfinance is a scraper and can break or rate-limit you.

Limits to state in the report: you only get about 4 years of annual financials, the sample covers large listed firms only, and the country field is the headquarters, not where the company actually earns its money. That is the reason the main stock indices of small countries are a fair but imperfect sample.

*Alternatives if yfinance fails:* Financial Modeling Prep or Alpha Vantage. Both have free API-key tiers with small daily request limits, so check the current free-tier coverage before relying on either.

### 2.7 REST Countries (metadata, no key)

```
https://restcountries.com/v3.1/alpha?codes=SWE,PRT,USA,...&fields=cca3,cca2,name,region,subregion,latlng,currencies
```
Use it for region, subregion and coordinates for maps. You can also hard-code this data into your country table; for 40 countries that is fine.

### 2.8 Source coverage summary

| Source | Countries | Years | Key | Layer |
|---|---|---|---|---|
| World Bank WDI | 40/40 | 2012–2024 (lag 1–2 yrs) | No | Core |
| WGI | 40/40 | 2012–2023 | No | Core |
| IMF DataMapper | 40/40 | 2012–latest actual | No | Core |
| Eurostat | 27/40 | 2012–latest | No | Europe extension |
| OECD | ~29/40 | varies | No | Optional |
| yfinance | ~600 firms | ~4 fiscal years | No | Company outcome |
| REST Countries | 40/40 | static | No | Metadata |

---

## 3. Database design (your decisions)

Look at the data before you finalise the schema. Every source above arrives in roughly the same **long shape** once flattened:

```
country | indicator | year | value        ← World Bank, WGI, IMF, Eurostat, OECD
ticker  | fiscal_year | metric | value    ← company financials
```

That suggests a natural design, but these are the decisions you need to make and justify.

### 3.1 Decisions to make

| Decision | Options | Recommendation |
|---|---|---|
| Engine | SQLite / **DuckDB** / PostgreSQL | DuckDB: one file, fast SQL on pandas, works in Streamlit, no server. Choose PostgreSQL if you want to show server-database skills. |
| Shape of indicator data | Long (one row per value) / wide (one column per indicator) | **Store long, read wide.** Long tables never change when you add an indicator. Wide views are built for analysis. |
| One table per source or one table for all | Per source / unified | Per source in **staging**, unified in **core**. |
| Same concept from two sources (e.g. unemployment from WB and IMF) | Keep both / pick one | Keep both rows tagged by source, and add a `concept_map` table that says which source is preferred for each concept. |
| Raw data | Files only / raw tables | Files in `data/raw/` named by source and date. They are your audit trail. |
| Countries keyed by | ISO3 / ISO2 / internal id | ISO3 as the primary key, with other codes as columns. |
| Company outcome per year | Fiscal year / calendar year | Map fiscal year to the calendar year in which it mostly falls, and document the rule. |

### 3.2 Proposed layers

```
data/raw/            JSON/CSV exactly as received (never edited)
staging schema       stg_worldbank, stg_wgi, stg_imf, stg_eurostat, stg_oecd, stg_companies
core schema          dimensions + facts (below)
mart schema          views for dashboards and ML
```

### 3.3 Proposed core schema

```sql
CREATE TABLE dim_country (
    iso3          VARCHAR PRIMARY KEY,
    iso2          VARCHAR,
    eurostat_code VARCHAR,
    name          VARCHAR,
    region        VARCHAR,
    income_group  VARCHAR,
    is_eu         BOOLEAN,
    is_euro       BOOLEAN,
    is_oecd       BOOLEAN,
    lat DOUBLE, lon DOUBLE
);

CREATE TABLE dim_source (
    source_id     VARCHAR PRIMARY KEY,   -- 'WB', 'WGI', 'IMF', 'ESTAT', 'OECD', 'YF'
    name          VARCHAR,
    base_url      VARCHAR,
    retrieved_at  TIMESTAMP
);

CREATE TABLE dim_indicator (
    indicator_id     VARCHAR PRIMARY KEY,  -- e.g. 'WB:SL.UEM.TOTL.ZS'
    source_id        VARCHAR REFERENCES dim_source,
    source_code      VARCHAR,
    name             VARCHAR,
    concept          VARCHAR,             -- e.g. 'unemployment'
    theme            VARCHAR,             -- economy, social, labour, governance...
    unit             VARCHAR,
    higher_is_better BOOLEAN,
    coverage         VARCHAR              -- 'all40' or 'eu27' or 'oecd'
);

CREATE TABLE fact_country_year (
    iso3          VARCHAR REFERENCES dim_country,
    indicator_id  VARCHAR REFERENCES dim_indicator,
    year          INTEGER,
    value         DOUBLE,
    PRIMARY KEY (iso3, indicator_id, year)
);

CREATE TABLE concept_map (
    concept              VARCHAR PRIMARY KEY,
    preferred_indicator  VARCHAR REFERENCES dim_indicator
);

CREATE TABLE dim_company (
    ticker     VARCHAR PRIMARY KEY,
    name       VARCHAR,
    iso3       VARCHAR REFERENCES dim_country,
    sector     VARCHAR,
    industry   VARCHAR,
    currency   VARCHAR
);

CREATE TABLE fact_company_year (
    ticker            VARCHAR REFERENCES dim_company,
    year              INTEGER,
    revenue           DOUBLE,
    operating_income  DOUBLE,
    net_income        DOUBLE,
    total_assets      DOUBLE,
    equity            DOUBLE,
    roa               DOUBLE,
    operating_margin  DOUBLE,
    revenue_growth    DOUBLE,
    PRIMARY KEY (ticker, year)
);
```

### 3.4 Mart views

- `v_country_wide`: one row per country and year, one column per preferred concept. Use DuckDB `PIVOT` on `fact_country_year` joined to `concept_map`.
- `v_country_latest`: the latest non-null value per country and concept, for snapshot rankings and maps.
- `v_coverage`: the share of non-missing values per country and indicator. It drives the coverage heatmap in Dashboard 1.
- `v_model_dataset`: company-year rows joined to **last year's** country features (`year − 1`). Lagging avoids using information from the same year as the outcome.

---

## 4. Repository structure

```
global-business-comparison/
├── README.md
├── requirements.txt
├── config/
│   ├── countries.csv          # the 40 countries with all codes
│   ├── indicators.yaml        # source, code, concept, theme, unit, higher_is_better
│   └── companies_seed.csv     # ticker, iso3
├── data/
│   ├── raw/                   # untouched API output (gitignored if large)
│   └── gbc.duckdb             # the database
├── src/
│   ├── fetch/
│   │   ├── worldbank.py       # WDI + WGI
│   │   ├── imf.py
│   │   ├── eurostat_fetch.py
│   │   ├── oecd.py
│   │   └── companies.py       # yfinance
│   ├── load/
│   │   ├── schema.sql
│   │   └── load_db.py         # raw → staging → core
│   ├── transform/
│   │   └── marts.sql
│   └── model/
│       └── train.py           # stretch
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_analysis.ipynb
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Raw_Data.py
│       ├── 2_Insights.py
│       └── 3_What_If.py       # stretch
└── report/
    ├── white_paper.md
    └── slides/
```

Put every code, country and indicator list in `config/`, never inside your scripts. Adding a 41st country or a new indicator should mean editing one file.

### 4.1 Example fetcher (World Bank)

```python
import json, time, requests, pathlib

BASE = "https://api.worldbank.org/v2/country/{c}/indicator/{i}"

def fetch_wb(indicator, countries, start=2012, end=2024, source=None):
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 1000}
    if source:
        params["source"] = source          # 3 = WGI
    url = BASE.format(c=";".join(countries), i=indicator)
    rows, page = [], 1
    while True:
        r = requests.get(url, params={**params, "page": page}, timeout=30)
        r.raise_for_status()
        meta, data = r.json()
        rows += data or []
        if page >= meta["pages"]:
            break
        page += 1
        time.sleep(0.3)
    out = pathlib.Path(f"data/raw/worldbank/{indicator}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    return rows
```

Each row has `countryiso3code`, `date` and `value`, which already match your long table.

---

## 5. Analysis plan

| Question | Data | Method | Output |
|---|---|---|---|
| How do the 40 countries compare? | Core layer | Rankings, z-scores, radar profiles | Ranking tables, radar chart |
| Which country factors move with business dynamism? | New business density, market cap vs. country features | Spearman correlation, scatter plots with trendlines | Correlation heatmap |
| Which country factors relate to firm performance? | `v_model_dataset` | Correlation of median firm ROA per country with features; regression `ROA ~ country features + sector + log(size)` (statsmodels) | Coefficient chart |
| EU vs non-EU | `is_eu` flag | Box plots, Mann–Whitney test | Comparison panel |
| Inside Europe: Nordic vs South vs East | Eurostat layer | Group comparison on labour cost, digital adoption, business demography | Europe panel |
| Are there country "types"? | Standardised core features | PCA + k-means (k = 3–5) | Cluster map |

**Cleaning rules to apply and document:**
- Treat values more than one year older than the target year as missing, unless you are using a "latest available" snapshot.
- Drop an indicator from the 40-country analysis if it is missing for more than 25% of countries.
- Standardise features (z-scores) before correlation, clustering and modelling.
- Winsorise firm ratios at the 1st and 99th percentiles, because one company with tiny assets can produce an ROA of 300%.
- Report the number of observations (n) on every chart. With 40 countries, n is small.

---

## 6. Dashboards (Streamlit + Plotly)

Streamlit is the fastest route. Deploy it for free from your GitHub repo on Streamlit Community Cloud, so you can show a live link in your portfolio.

### Dashboard 1: Raw data explorer (`1_Raw_Data.py`)

The purpose is to show what you collected, honestly, including the gaps.

- **Sidebar filters:** source, theme, indicator, countries (with "EU only" and "non-EU only" shortcuts), year range.
- **Coverage heatmap:** countries × indicators, coloured by % of years available. This proves you understand your data quality.
- **Time series** of the selected indicator for the selected countries.
- **Raw table** with a CSV download button.
- **Source panel:** which API, which endpoint and when it was retrieved, taken from `dim_source`.
- **Company table:** list of firms per country and sector, with their latest ratios.

### Dashboard 2: Insights (`2_Insights.py`)

The purpose is to show the cleaned, filtered and correlated results.

- **KPI cards:** number of countries, firms and indicators, and median ROA for EU vs non-EU.
- **Choropleth world map:** pick any concept from `v_country_latest`.
- **Correlation heatmap:** country features × business outcomes, with a Spearman/Pearson toggle.
- **"Pick two" scatter:** any feature against any outcome, with a trendline, labelled points and r and n shown.
- **Country profile:** a radar chart comparing one country with the EU median and the non-EU median.
- **EU vs non-EU box plots.**
- **Clusters:** map coloured by k-means cluster, with a table of cluster averages.
- **Europe tab:** Eurostat-only views (labour cost, digital adoption, business demography).

### Dashboard 3: What-if (`3_What_If.py`, stretch)

- Inputs: sector, company size (total assets), current country, target country.
- Output: predicted ROA in each country, plus a bar chart of all 40 countries for that company profile and the top feature contributions (SHAP).
- Label on the page: *"Associational estimate from historical data, not a causal prediction."*

---

## 7. ML model (stretch goal)

**Goal:** estimate how a company with a given profile might perform under the conditions of each country.

**Dataset:** `v_model_dataset`, one row per company-year.

| Part | Content |
|---|---|
| Target | ROA (or operating margin), winsorised |
| Company features | sector (one-hot), log(total assets), previous-year revenue growth |
| Country features | lagged (t−1) core-layer concepts: GDP per capita PPP, growth, inflation, lending rate, unemployment, tax revenue, WGI scores, internet use, R&D, credit to private sector, market cap % GDP |
| Excluded | Eurostat-only features, so the model works for all 40 countries |

**Models, in order:**
1. **Baseline:** predict the sector median. Anything that can't beat this is not useful.
2. Ridge regression (interpretable).
3. Gradient boosting (`HistGradientBoostingRegressor`, or LightGBM).

**Validation, the important part:** use `GroupKFold` with **country as the group**. This tests whether the model generalises to a country it has never seen, which is exactly the "insert a company into a country" question. A normal random split leaks country information and gives misleadingly good scores.

**Metrics:** MAE and R² compared with the baseline. Expect a modest R²; firm performance is noisy, and saying so is fine.

**What-if mechanics:** take one company row, swap in the country features of each of the 40 countries, and predict 40 times. Rank the results and explain them with SHAP values (`pip install shap`).

**Caveats to write down:** about 600 firms × about 4 years is small. Headquarters country is not the same as operating country. There are survivorship and large-firm bias, and different accounting standards. The model learns association and cannot tell what *would* happen if a company actually moved.

---

## 8. Schedule (4 days core, ML only if time remains)

| Day | Must do | Done when |
|---|---|---|
| **1: Config + fetch** | Write `countries.csv`, `indicators.yaml` and `companies_seed.csv`. Build the World Bank, WGI and IMF fetchers and run them. Start yfinance running in the background. | Raw files for all core sources are in `data/raw/` |
| **2: Database** | Fetch Eurostat. Create the schema, load staging → core, build `concept_map` and the mart views, compute company ratios. **Freeze the data at the end of the day.** | `SELECT * FROM v_country_wide` returns 40 countries |
| **3: Analysis + dashboards** | Correlations, regression, EU vs non-EU, clusters. Build Dashboard 1 and Dashboard 2. | Both dashboards run locally with real data |
| **4: Polish + report** | Deploy on Streamlit Cloud, write results into the white paper, make the slides, clean the README. | Live link, report and slides finished |
| **Stretch** | ML model + Dashboard 3 | GroupKFold results beat the baseline |

Fallbacks if you fall behind:
- If yfinance is slow or broken by day 2, drop firm-level data and use level-A outcomes (new business density, market cap). The project still answers the question at country level.
- If Eurostat parsing eats time, keep only 3–4 Eurostat datasets.
- If Dashboard 2 runs long, cut the clusters tab before cutting the correlation and scatter views.

---

## 9. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| API down or rate-limited | Cache raw files and never refetch inside the dashboard. Use retries with `time.sleep`. |
| Different country codes | `dim_country` holds ISO3, ISO2 and the Eurostat code. Join through it every time. |
| Missing recent years | Use a "latest available" view and show coverage openly in Dashboard 1. |
| Same concept, different numbers across sources | `concept_map` picks one preferred source. Mention the differences in the report. |
| Eurostat duplicate rows | Filter every extra dimension (`unit`, `na_item`, `nace_r2`, `sex`, `age`) to one value. |
| Currency mixing in firm data | Use ratios only. Never compare raw revenue across currencies. |
| Outlier firms | Winsorise ratios and show medians rather than means. |
| Overclaiming | Say "associated with", never "causes". Show n on every chart. |

---

## 10. Final checklist

- [ ] `countries.csv` with 40 rows and all three code columns
- [ ] `indicators.yaml` with every code verified on the source website
- [ ] Raw files saved for every source, with retrieval dates
- [ ] DuckDB schema created and loaded, with mart views working
- [ ] Data dictionary (from `dim_indicator`) exported to the report
- [ ] Dashboard 1: raw explorer with coverage heatmap
- [ ] Dashboard 2: insights with correlations, scatter, map, EU vs non-EU
- [ ] Deployed Streamlit link in the README
- [ ] White paper updated with results and limitations
- [ ] Slides for the defense
- [ ] (Stretch) ML model with GroupKFold validation and the what-if page
