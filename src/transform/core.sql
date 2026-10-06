-- staging -> core. Only countries in dim_country are loaded (config/countries.csv decides the scope).

INSERT INTO core.dim_source SELECT source_id, name, base_url, retrieved_at FROM staging.stg_sources;

INSERT INTO core.dim_country
SELECT c.iso3, c.iso2, NULLIF(c.eurostat_code, ''), c.name, c.region, NULLIF(c.eu_group, ''),
       w.income_group, c.is_eu, c.is_euro, c.is_oecd, c.small_firm_sample,
       NULLIF(w.lat, '')::double precision, NULLIF(w.lon, '')::double precision
FROM staging.stg_countries_config c
LEFT JOIN staging.stg_wb_countries w USING (iso3);

-- ── indicators ──────────────────────────────────────────────────────────────
INSERT INTO core.dim_indicator (indicator_id, source_id, source_code, name, theme, snapshot_only)
SELECT 'WB:' || s.indicator_code, 'WB', s.indicator_code, MAX(s.indicator_name), MAX(k.theme),
       COALESCE(BOOL_OR(k.snapshot_only), FALSE)
FROM staging.stg_worldbank s
LEFT JOIN staging.stg_indicator_config k ON k.source = 'WB' AND k.code = s.indicator_code
GROUP BY s.indicator_code;

INSERT INTO core.dim_indicator (indicator_id, source_id, source_code, name, theme, snapshot_only)
SELECT 'WGI:' || s.indicator_code, 'WGI', s.indicator_code, MAX(s.indicator_name), MAX(k.theme),
       COALESCE(BOOL_OR(k.snapshot_only), FALSE)
FROM staging.stg_wgi s
LEFT JOIN staging.stg_indicator_config k ON k.source = 'WGI' AND k.code = s.indicator_code
GROUP BY s.indicator_code;

INSERT INTO core.dim_indicator (indicator_id, source_id, source_code, name, description, theme, unit)
SELECT 'IMF:' || d.indicator_code, 'IMF', d.indicator_code, i.label, i.description,
       (SELECT k.theme FROM staging.stg_indicator_config k WHERE k.source = 'IMF' AND k.code = d.indicator_code),
       i.unit
FROM (SELECT DISTINCT indicator_code FROM staging.stg_imf) d
LEFT JOIN staging.stg_imf_indicators i ON i.code = d.indicator_code;

INSERT INTO core.dim_indicator (indicator_id, source_id, source_code, name, description, theme, unit, filters)
SELECT 'ESTAT:' || e.dataset || ':' || e.dims_key, 'ESTAT', e.dataset, d.label, e.dims_key, d.theme,
       e.dims ->> 'unit', e.dims
FROM (SELECT DISTINCT dataset, dims,
             (SELECT string_agg(k || '=' || v, '|' ORDER BY k) FROM jsonb_each_text(dims) AS t(k, v)) AS dims_key
      FROM staging.stg_eurostat) e
JOIN staging.stg_eurostat_datasets d USING (dataset);

-- ── country-year facts ──────────────────────────────────────────────────────
INSERT INTO core.fact_country_year (iso3, indicator_id, period, year, value)
SELECT iso3, 'WB:' || indicator_code, year::text, year, value
FROM staging.stg_worldbank WHERE value IS NOT NULL AND iso3 IN (SELECT iso3 FROM core.dim_country);

INSERT INTO core.fact_country_year (iso3, indicator_id, period, year, value)
SELECT iso3, 'WGI:' || indicator_code, year::text, year, value
FROM staging.stg_wgi WHERE value IS NOT NULL AND iso3 IN (SELECT iso3 FROM core.dim_country);

INSERT INTO core.fact_country_year (iso3, indicator_id, period, year, value)
SELECT iso3, 'IMF:' || indicator_code, year::text, year, value
FROM staging.stg_imf WHERE value IS NOT NULL AND iso3 IN (SELECT iso3 FROM core.dim_country);

INSERT INTO core.fact_country_year (iso3, indicator_id, period, year, value, flag)
SELECT iso3,
       'ESTAT:' || dataset || ':' || (SELECT string_agg(k || '=' || v, '|' ORDER BY k) FROM jsonb_each_text(dims) AS t(k, v)),
       period, LEFT(period, 4)::int, value, NULLIF(flag, '')
FROM staging.stg_eurostat
WHERE value IS NOT NULL AND iso3 IN (SELECT iso3 FROM core.dim_country);

-- ── companies ───────────────────────────────────────────────────────────────
INSERT INTO core.dim_company
SELECT DISTINCT ON (s.ticker)
       'YF:' || s.ticker, 'YF', s.ticker, COALESCE(m.longname, s.name), s.iso3,
       COALESCE(m.sector, s.sector), COALESCE(m.industry, s.industry), m.exchange,
       COALESCE(m.currency, s.currency), m.financialcurrency,
       COALESCE(m.marketcap, s.market_cap), m.fulltimeemployees
FROM staging.stg_yf_seed s
LEFT JOIN staging.stg_yf_meta m ON m.ticker = s.ticker
WHERE s.iso3 IN (SELECT iso3 FROM core.dim_country)
  AND EXISTS (SELECT 1 FROM staging.stg_yf_financials f WHERE f.ticker = s.ticker)
ORDER BY s.ticker;

INSERT INTO core.dim_company (company_id, source_id, source_key, name, iso3)
SELECT DISTINCT ON (lei) 'ESEF:' || lei, 'ESEF', lei, entity_name, iso3
FROM staging.stg_esef_facts
WHERE iso3 IN (SELECT iso3 FROM core.dim_country)
ORDER BY lei, entity_name;

-- ── company financials: yfinance ───────────────────────────────────────────
-- yfinance gives only the fiscal year end, so the start is derived (end - 1 year + 1 day).
INSERT INTO core.fact_company_financial
SELECT DISTINCT ON (f.ticker, f.statement, f.item, f.period_end)
       'YF:' || f.ticker, f.statement, f.item,
       (f.period_end - INTERVAL '1 year' + INTERVAL '1 day')::date, f.period_end, TRUE,
       COALESCE(c.financial_currency, c.currency), f.value, NULL
FROM staging.stg_yf_financials f
JOIN core.dim_company c ON c.company_id = 'YF:' || f.ticker
ORDER BY f.ticker, f.statement, f.item, f.period_end;

-- ── company financials: ESEF ──────────────────────────────────────────────
-- Keep annual durations and all instants. Reporting currency = the unit most used for Assets.
-- The same fact appears in several filings (prior-year comparatives): keep the latest filing.
CREATE TEMP TABLE esef_unit AS
SELECT DISTINCT ON (lei) lei, unit AS reporting_unit
FROM (SELECT lei, unit, COUNT(*) n FROM staging.stg_esef_facts WHERE concept = 'Assets' GROUP BY lei, unit) x
ORDER BY lei, n DESC;

CREATE TEMP TABLE esef_latest AS
SELECT DISTINCT ON (e.lei, e.concept, e.period_start, e.period_end)
       e.lei, e.concept, e.period_start, e.period_end, e.unit, e.value, e.fxo_id
FROM staging.stg_esef_facts e
JOIN esef_unit u ON u.lei = e.lei
LEFT JOIN (SELECT DISTINCT fxo_id, date_added FROM staging.stg_esef_index) i ON i.fxo_id = e.fxo_id
WHERE e.iso3 IN (SELECT iso3 FROM core.dim_country)
  AND (e.period_start IS NULL OR e.period_end - e.period_start BETWEEN 350 AND 380)
  AND (e.unit = u.reporting_unit OR e.concept IN ('NumberOfEmployees', 'AverageNumberOfEmployees'))
ORDER BY e.lei, e.concept, e.period_start, e.period_end, i.date_added DESC NULLS LAST;

CREATE TEMP TABLE esef_fy AS
SELECT lei, period_end, MIN(period_start) AS fy_start
FROM esef_latest WHERE period_start IS NOT NULL GROUP BY lei, period_end;

INSERT INTO core.fact_company_financial
SELECT 'ESEF:' || l.lei,
       CASE WHEN l.concept IN ('NumberOfEmployees', 'AverageNumberOfEmployees') THEN 'other'
            WHEN l.concept LIKE 'CashFlows%' THEN 'cashflow'
            WHEN l.period_start IS NOT NULL THEN 'income'
            ELSE 'balance' END,
       l.concept,
       COALESCE(l.period_start, f.fy_start, (l.period_end - INTERVAL '1 year' + INTERVAL '1 day')::date),
       l.period_end,
       (l.period_start IS NULL AND f.fy_start IS NULL),
       l.unit, l.value, l.fxo_id
FROM esef_latest l
LEFT JOIN esef_fy f ON f.lei = l.lei AND f.period_end = l.period_end;

-- ESEF companies with no Assets fact get no reporting currency, so none of their facts load: drop them.
DELETE FROM core.dim_company c
WHERE NOT EXISTS (SELECT 1 FROM core.fact_company_financial f WHERE f.company_id = c.company_id);

-- ── metric map for the standard company metrics ────────────────────────────
INSERT INTO core.company_metric_map (source_id, statement, item, metric, priority) VALUES
 ('YF',   'income',  'Total Revenue',                       'revenue',          1),
 ('YF',   'income',  'Operating Income',                    'operating_income', 1),
 ('YF',   'income',  'Net Income',                          'net_income',       1),
 ('YF',   'balance', 'Total Assets',                        'total_assets',     1),
 ('YF',   'balance', 'Stockholders Equity',                 'equity',           1),
 ('YF',   'balance', 'Common Stock Equity',                 'equity',           2),
 ('ESEF', 'income',  'Revenue',                             'revenue',          1),
 ('ESEF', 'income',  'RevenueFromContractsWithCustomers',   'revenue',          2),
 ('ESEF', 'income',  'ProfitLossFromOperatingActivities',   'operating_income', 1),
 ('ESEF', 'income',  'ProfitLossAttributableToOwnersOfParent', 'net_income',    1),
 ('ESEF', 'income',  'ProfitLoss',                          'net_income',       2),
 ('ESEF', 'balance', 'Assets',                              'total_assets',     1),
 ('ESEF', 'balance', 'EquityAttributableToOwnersOfParent', 'equity',           1),
 ('ESEF', 'balance', 'Equity',                              'equity',           2);
