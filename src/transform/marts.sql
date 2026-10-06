-- Views for dashboards and analysis. No data is stored here.

-- latest non-null value per country and indicator (rankings, maps)
CREATE VIEW mart.v_country_latest AS
SELECT DISTINCT ON (f.iso3, f.indicator_id)
       f.iso3, f.indicator_id, i.name AS indicator_name, i.theme, f.period, f.year, f.value
FROM core.fact_country_year f
JOIN core.dim_indicator i USING (indicator_id)
ORDER BY f.iso3, f.indicator_id, f.period DESC;

-- share of years 2012-2024 with data, per country and indicator (coverage heatmap).
-- Eurostat series only apply to EU countries, so non-EU countries are left out of that grid.
CREATE VIEW mart.v_coverage AS
WITH grid AS (
    SELECT c.iso3, i.indicator_id, i.source_id, i.name AS indicator_name, i.theme
    FROM core.dim_country c
    CROSS JOIN core.dim_indicator i
    WHERE i.source_id <> 'ESTAT' OR c.is_eu
), have AS (
    SELECT iso3, indicator_id, COUNT(DISTINCT year) AS n_years, MIN(year) AS first_year, MAX(year) AS last_year
    FROM core.fact_country_year
    WHERE year BETWEEN 2012 AND 2024
    GROUP BY iso3, indicator_id
)
SELECT g.*, COALESCE(h.n_years, 0) AS n_years, h.first_year, h.last_year,
       ROUND(100.0 * COALESCE(h.n_years, 0) / 13, 1) AS pct_years
FROM grid g LEFT JOIN have h USING (iso3, indicator_id);

-- one row per company and fiscal year, standard metrics side by side.
-- fiscal_year = calendar year in which most of the fiscal year falls (midpoint rule).
-- Ratios are raw (not winsorised); revenue_growth needs the previous year to be ~12 months earlier.
CREATE VIEW mart.v_company_year AS
WITH m AS (
    SELECT f.company_id, f.fiscal_year_start, f.period_end, mm.metric, f.value, f.currency, mm.priority
    FROM core.fact_company_financial f
    JOIN core.dim_company c USING (company_id)
    JOIN core.company_metric_map mm
      ON mm.source_id = c.source_id AND mm.statement = f.statement AND mm.item = f.item
), best AS (
    SELECT DISTINCT ON (company_id, period_end, metric) *
    FROM m ORDER BY company_id, period_end, metric, priority, fiscal_year_start
), wide AS (
    SELECT company_id, period_end, MIN(fiscal_year_start) AS fiscal_year_start, MAX(currency) AS currency,
           MAX(value) FILTER (WHERE metric = 'revenue')          AS revenue,
           MAX(value) FILTER (WHERE metric = 'operating_income') AS operating_income,
           MAX(value) FILTER (WHERE metric = 'net_income')       AS net_income,
           MAX(value) FILTER (WHERE metric = 'total_assets')     AS total_assets,
           MAX(value) FILTER (WHERE metric = 'equity')           AS equity
    FROM best GROUP BY company_id, period_end
), lagged AS (
    SELECT w.*,
           LAG(revenue)    OVER (PARTITION BY company_id ORDER BY period_end) AS prev_revenue,
           LAG(period_end) OVER (PARTITION BY company_id ORDER BY period_end) AS prev_period_end
    FROM wide w
)
SELECT l.company_id, c.source_id, c.name, c.iso3, c.sector,
       EXTRACT(YEAR FROM l.fiscal_year_start + (l.period_end - l.fiscal_year_start) / 2)::int AS fiscal_year,
       l.fiscal_year_start, l.period_end, l.currency,
       l.revenue, l.operating_income, l.net_income, l.total_assets, l.equity,
       l.net_income / NULLIF(l.total_assets, 0)      AS roa,
       l.operating_income / NULLIF(l.revenue, 0)     AS operating_margin,
       CASE WHEN l.period_end - l.prev_period_end BETWEEN 350 AND 380
            THEN l.revenue / NULLIF(l.prev_revenue, 0) - 1 END AS revenue_growth
FROM lagged l JOIN core.dim_company c USING (company_id);

-- how many companies each country has, per source, with usable data (assets and net income)
CREATE VIEW mart.v_company_coverage AS
SELECT c.iso3, y.source_id,
       COUNT(DISTINCT y.company_id) AS companies,
       COUNT(*)                     AS firm_years,
       ROUND(COUNT(*)::numeric / COUNT(DISTINCT y.company_id), 1) AS years_per_company
FROM mart.v_company_year y
JOIN core.dim_company c USING (company_id)
WHERE y.net_income IS NOT NULL AND y.total_assets IS NOT NULL
GROUP BY c.iso3, y.source_id;
