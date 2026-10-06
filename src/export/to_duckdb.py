"""Export the PostgreSQL core tables and mart views to a single DuckDB file for the dashboards.

Usage:  uv run python src/export/to_duckdb.py
Output: data/gbc.duckdb  (rebuilt from scratch every run; never edit it by hand)
Env:    GBC_DSN  PostgreSQL connection string (default: "dbname=gbc")
"""
import os
import pathlib

import duckdb
import pandas as pd
import psycopg

ROOT = pathlib.Path(__file__).resolve().parents[2]
DSN = os.environ.get("GBC_DSN", "dbname=gbc")
OUT = ROOT / "data" / "gbc.duckdb"

# Postgres relation -> table name in DuckDB (views are materialised as tables)
RELATIONS = {
    "core.dim_source": "dim_source",
    "core.dim_country": "dim_country",
    "core.dim_indicator": "dim_indicator",
    "core.fact_country_year": "fact_country_year",
    "core.dim_company": "dim_company",
    "core.fact_company_financial": "fact_company_financial",
    "core.company_metric_map": "company_metric_map",
    "mart.v_country_latest": "v_country_latest",
    "mart.v_coverage": "v_coverage",
    "mart.v_company_year": "v_company_year",
    "mart.v_company_coverage": "v_company_coverage",
}


def main():
    if OUT.exists():
        OUT.unlink()
    duck = duckdb.connect(str(OUT))
    with psycopg.connect(DSN) as pg, pg.cursor() as cur:
        for rel, name in RELATIONS.items():
            cur.execute(f"SELECT * FROM {rel}")
            df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            duck.register("df", df)
            duck.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
            duck.unregister("df")
            print(f"  {name}: {len(df):,} rows", flush=True)
    duck.close()
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
