-- Schema for the global business comparison database (PostgreSQL).
-- Layers: staging (1:1 with the source files) -> core (dimensions + long facts) -> mart (views).
-- Rebuilt from scratch by src/load/build_db.py.

DROP SCHEMA IF EXISTS mart CASCADE;
DROP SCHEMA IF EXISTS core CASCADE;
DROP SCHEMA IF EXISTS staging CASCADE;
CREATE SCHEMA staging;
CREATE SCHEMA core;
CREATE SCHEMA mart;

-- ───────────────────────────── STAGING ─────────────────────────────
CREATE TABLE staging.stg_sources (
    source_id TEXT, name TEXT, base_url TEXT, retrieved_at TIMESTAMP
);
CREATE TABLE staging.stg_countries_config (
    iso3 TEXT, iso2 TEXT, eurostat_code TEXT, name TEXT, region TEXT, eu_group TEXT,
    is_eu BOOLEAN, is_euro BOOLEAN, is_oecd BOOLEAN, small_firm_sample BOOLEAN
);
CREATE TABLE staging.stg_wb_countries (
    iso3 TEXT, iso2 TEXT, name TEXT, region TEXT, income_group TEXT, lat TEXT, lon TEXT
);
CREATE TABLE staging.stg_indicator_config (          -- config/indicators.yaml
    source TEXT, code TEXT, theme TEXT, snapshot_only BOOLEAN
);
-- World Bank WDI and WGI share one shape
CREATE TABLE staging.stg_worldbank (indicator_code TEXT, indicator_name TEXT, iso3 TEXT, year INT, value DOUBLE PRECISION);
CREATE TABLE staging.stg_wgi       (indicator_code TEXT, indicator_name TEXT, iso3 TEXT, year INT, value DOUBLE PRECISION);
CREATE TABLE staging.stg_imf       (indicator_code TEXT, iso3 TEXT, year INT, value DOUBLE PRECISION);
CREATE TABLE staging.stg_imf_indicators (code TEXT, label TEXT, unit TEXT, description TEXT);
CREATE TABLE staging.stg_eurostat_datasets (dataset TEXT, label TEXT, theme TEXT);
CREATE TABLE staging.stg_eurostat (
    dataset TEXT, iso3 TEXT, geo TEXT, period TEXT, value DOUBLE PRECISION, flag TEXT, dims JSONB
);
CREATE TABLE staging.stg_yf_seed (
    ticker TEXT, iso3 TEXT, name TEXT, sector TEXT, industry TEXT, currency TEXT, market_cap DOUBLE PRECISION
);
CREATE TABLE staging.stg_yf_meta (
    ticker TEXT, iso3 TEXT, ok BOOLEAN, longname TEXT, country TEXT, sector TEXT, industry TEXT,
    currency TEXT, financialcurrency TEXT, marketcap DOUBLE PRECISION, exchange TEXT,
    fulltimeemployees DOUBLE PRECISION
);
CREATE TABLE staging.stg_yf_financials (
    ticker TEXT, iso3 TEXT, statement TEXT, period_end DATE, item TEXT, value DOUBLE PRECISION
);
CREATE TABLE staging.stg_esef_index (
    fxo_id TEXT, country TEXT, lei TEXT, entity_name TEXT, json_url TEXT, date_added TIMESTAMP
);
CREATE TABLE staging.stg_esef_facts (
    fxo_id TEXT, lei TEXT, entity_name TEXT, country TEXT, concept TEXT,
    period_start DATE, period_end DATE, unit TEXT, value DOUBLE PRECISION, iso3 TEXT
);

-- ───────────────────────────── CORE ─────────────────────────────
CREATE TABLE core.dim_source (
    source_id    TEXT PRIMARY KEY,           -- WB, WGI, IMF, ESTAT, YF, ESEF
    name         TEXT,
    base_url     TEXT,
    retrieved_at TIMESTAMP
);

CREATE TABLE core.dim_country (
    iso3              TEXT PRIMARY KEY,
    iso2              TEXT,
    eurostat_code     TEXT,                  -- NULL outside the EU; Greece is EL
    name              TEXT,
    region            TEXT,
    eu_group          TEXT,                  -- Nordic / Western / Southern / Eastern (EU only)
    income_group      TEXT,
    is_eu             BOOLEAN,
    is_euro           BOOLEAN,
    is_oecd           BOOLEAN,
    small_firm_sample BOOLEAN,               -- market too small to reach ~50 firms
    lat               DOUBLE PRECISION,
    lon               DOUBLE PRECISION
);

CREATE TABLE core.dim_indicator (
    indicator_id  TEXT PRIMARY KEY,          -- 'WB:SL.UEM.TOTL.ZS', 'ESTAT:une_rt_a:age=Y15-74|sex=T|unit=PC_ACT'
    source_id     TEXT NOT NULL REFERENCES core.dim_source,
    source_code   TEXT NOT NULL,             -- indicator code, or dataset code for Eurostat
    name          TEXT,
    description   TEXT,
    theme         TEXT,
    unit          TEXT,
    snapshot_only BOOLEAN DEFAULT FALSE,
    filters       JSONB                      -- Eurostat dimension values that define this series
);

CREATE TABLE core.fact_country_year (
    iso3         TEXT NOT NULL REFERENCES core.dim_country,
    indicator_id TEXT NOT NULL REFERENCES core.dim_indicator,
    period       TEXT NOT NULL,              -- '2019', or '2012-S1' for semi-annual series
    year         INT  NOT NULL,
    value        DOUBLE PRECISION,
    flag         TEXT,                       -- Eurostat quality flag (b break, p provisional, e estimated)
    PRIMARY KEY (iso3, indicator_id, period)
);

CREATE TABLE core.dim_company (
    company_id         TEXT PRIMARY KEY,     -- 'YF:7203.T' or 'ESEF:<lei>'
    source_id          TEXT NOT NULL REFERENCES core.dim_source,
    source_key         TEXT NOT NULL,        -- ticker (YF) or LEI (ESEF)
    name               TEXT,
    iso3               TEXT NOT NULL REFERENCES core.dim_country,   -- headquarters country
    sector             TEXT,                 -- NULL for ESEF (filings carry no sector)
    industry           TEXT,
    exchange           TEXT,
    currency           TEXT,                 -- trading currency
    financial_currency TEXT,                 -- currency of the statements
    market_cap         DOUBLE PRECISION,
    employees          DOUBLE PRECISION
);

CREATE TABLE core.fact_company_financial (
    company_id        TEXT NOT NULL REFERENCES core.dim_company,
    statement         TEXT NOT NULL,         -- income | balance | cashflow | other
    item              TEXT NOT NULL,         -- item name exactly as the source reports it
    fiscal_year_start DATE NOT NULL,
    period_end        DATE NOT NULL,         -- fiscal year end (balance sheet date)
    fy_start_derived  BOOLEAN NOT NULL,      -- TRUE: start computed as end - 1 year + 1 day; FALSE: reported
    currency          TEXT,
    value             DOUBLE PRECISION,
    filing_id         TEXT,                  -- ESEF filing the value came from (NULL for YF)
    PRIMARY KEY (company_id, statement, item, fiscal_year_start, period_end)
);
CREATE INDEX ON core.fact_company_financial (company_id, period_end);

-- Which source items feed the standard company metrics used by mart.v_company_year.
-- Lowest priority number wins when a source has several candidates.
CREATE TABLE core.company_metric_map (
    source_id TEXT NOT NULL REFERENCES core.dim_source,
    statement TEXT NOT NULL,
    item      TEXT NOT NULL,
    metric    TEXT NOT NULL,
    priority  INT  NOT NULL DEFAULT 1,
    PRIMARY KEY (source_id, statement, item)
);
