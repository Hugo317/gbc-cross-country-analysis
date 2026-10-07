# Country conditions and company performance

How do the social and economic conditions of a country relate to the performance of the companies that operate there? This project compares 34 countries (21 EU and 13 non-EU) using 35 country indicators and about 10,000 firm-years from 2,590 listed companies, and tests 17 hypotheses with methods built for a small number of countries (wild cluster bootstrap, Benjamini–Hochberg correction), with a replication on a second data source.

**Live dashboard:** [gbc-cross-country-analysis.onrender.com](https://gbc-cross-country-analysis.onrender.com). It runs on a free host that sleeps when idle, so the first visit can take about a minute.

## Main findings

- Firms in more developed countries earn slightly less on their assets (about -1.1 pp ROA per +1 SD of development). It survives dropping any single country and is confirmed on a second data source.
- The effect sits in small firms; the largest third of firms show almost none (exploratory).
- Results are associations, not causation. Only 34 countries are independent observations, and several findings came from exploring the same data, which the dashboard labels.

## Run it

```bash
uv run --group dash python app/dash_app.py        # http://127.0.0.1:8050
uv run --group dash python app/build_extract.py   # rebuild app/data/ after the analysis changes (needs data/gbc.duckdb)
```

The dashboard reads only `app/data/`, so it runs without the database.

## Layout

| Folder | What's in it |
|---|---|
| `src/` | data fetching, loading into PostgreSQL / DuckDB, transforms and analysis helpers |
| `notebooks/` | data catalog, EDA and the hypothesis notebooks |
| `results/` | one JSON per hypothesis, read by the dashboard |
| `app/` | the Plotly Dash dashboard, its data extract and deploy files (`Dockerfile`, `render.yaml` at the root) |

Data sources: World Bank WDI and WGI, IMF DataMapper, Eurostat, Yahoo Finance (via yfinance, unofficial) and ESEF/xBRL filings. See the Method & caveats tab of the dashboard for the limits of the data.
