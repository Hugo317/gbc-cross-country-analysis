"""Build the PostgreSQL database from the saved raw/flat files: schema -> staging -> core -> marts.

Usage:  uv run python src/load/build_db.py            (rebuilds everything from scratch)
Env:    GBC_DSN  PostgreSQL connection string (default: "dbname=gbc")
"""
import datetime as dt
import glob
import io
import json
import os
import pathlib

import pandas as pd
import psycopg
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, FLAT, CONFIG = ROOT / "data" / "raw", ROOT / "data" / "flat", ROOT / "config"
DSN = os.environ.get("GBC_DSN", "dbname=gbc")

EUROSTAT_THEMES = {
    "nama_10_pc": "economy", "une_rt_a": "labour", "prc_hicp_aind": "prices",
    "lc_lci_lev": "labour_cost", "earn_mw_cur": "labour_cost", "rd_e_gerdtot": "innovation",
    "isoc_ci_ifp_iu": "digital", "isoc_eb_ai": "digital", "sbs_na_sca_r2": "business_structure",
    "sbs_sc_ovw": "business_structure", "bd_size": "business_demography", "nasa_10_ki": "corporate_performance",
}


def copy_df(conn, df, table, columns=None):
    """Bulk load a DataFrame with COPY. Empty strings and NaN become NULL."""
    columns = columns or list(df.columns)
    buf = io.StringIO()
    df[columns].to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({','.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '')") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)
    print(f"  {table}: {len(df):,} rows", flush=True)


def mtime(path):
    return dt.datetime.fromtimestamp(pathlib.Path(path).stat().st_mtime)


def read_wb_dir(sub):
    rows = []
    for path in sorted(glob.glob(str(RAW / sub / "*.json"))):
        if pathlib.Path(path).name.startswith("_"):
            continue
        for r in json.loads(pathlib.Path(path).read_text()):
            rows.append({"indicator_code": r["indicator"]["id"], "indicator_name": r["indicator"]["value"],
                         "iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]})
    return pd.DataFrame(rows)


def load_staging(conn):
    print("loading staging")
    sources = pd.DataFrame([
        ("WB", "World Bank WDI", "https://api.worldbank.org/v2", mtime(RAW / "wb" / "SP.POP.TOTL.json")),
        ("WGI", "Worldwide Governance Indicators (World Bank source 3)", "https://api.worldbank.org/v2", mtime(RAW / "wgi" / "GOV_WGI_GE.EST.json")),
        ("IMF", "IMF DataMapper (World Economic Outlook)", "https://www.imf.org/external/datamapper/api/v1", mtime(RAW / "imf" / "NGDP_RPCH.json")),
        ("ESTAT", "Eurostat", "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data", mtime(FLAT / "eurostat_long.parquet")),
        ("YF", "Yahoo Finance via yfinance", "https://finance.yahoo.com", mtime(FLAT / "company_financials_long.parquet")),
        ("ESEF", "ESEF filings (filings.xbrl.org)", "https://filings.xbrl.org/api", mtime(FLAT / "esef_facts_long.parquet")),
    ], columns=["source_id", "name", "base_url", "retrieved_at"])
    copy_df(conn, sources, "staging.stg_sources")

    copy_df(conn, pd.read_csv(CONFIG / "countries.csv"), "staging.stg_countries_config")
    wbc = pd.DataFrame([{"iso3": r["id"], "iso2": r["iso2Code"], "name": r["name"], "region": r["region"]["value"],
                         "income_group": r["incomeLevel"]["value"], "lat": r["latitude"], "lon": r["longitude"]}
                        for r in json.loads((RAW / "wb_meta" / "countries.json").read_text())])
    copy_df(conn, wbc, "staging.stg_wb_countries")

    cfg = yaml.safe_load((CONFIG / "indicators.yaml").read_text())
    cfg_rows = [{"source": s, "code": i["code"], "theme": i.get("theme"), "snapshot_only": i.get("snapshot_only", False)}
                for s, items in cfg.items() for i in items]
    copy_df(conn, pd.DataFrame(cfg_rows), "staging.stg_indicator_config")

    copy_df(conn, read_wb_dir("wb"), "staging.stg_worldbank")
    copy_df(conn, read_wb_dir("wgi"), "staging.stg_wgi")

    imf = []
    for path in sorted(glob.glob(str(RAW / "imf" / "*.json"))):
        code = pathlib.Path(path).stem
        if code.startswith("_"):
            continue
        imf += [{"indicator_code": code, "iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
                for r in json.loads(pathlib.Path(path).read_text())]
    copy_df(conn, pd.DataFrame(imf), "staging.stg_imf")
    ind = json.loads((RAW / "imf" / "_indicators.json").read_text())
    copy_df(conn, pd.DataFrame([{"code": k, "label": v.get("label"), "unit": v.get("unit"), "description": v.get("description")}
                                for k, v in ind.items()]), "staging.stg_imf_indicators")

    es = pd.read_parquet(FLAT / "eurostat_long.parquet")
    copy_df(conn, es, "staging.stg_eurostat", ["dataset", "iso3", "geo", "period", "value", "flag", "dims"])
    labels = [{"dataset": p.stem, "label": json.loads(p.read_text())["label"], "theme": EUROSTAT_THEMES.get(p.stem)}
              for p in sorted((RAW / "eurostat").glob("*.json"))]
    copy_df(conn, pd.DataFrame(labels), "staging.stg_eurostat_datasets")

    seed = pd.read_csv(CONFIG / "companies_seed.csv")
    copy_df(conn, seed, "staging.stg_yf_seed")
    meta = pd.read_parquet(FLAT / "company_meta.parquet").rename(columns={
        "longName": "longname", "financialCurrency": "financialcurrency", "marketCap": "marketcap",
        "fullTimeEmployees": "fulltimeemployees"})
    copy_df(conn, meta, "staging.stg_yf_meta")
    fin = pd.read_parquet(FLAT / "company_financials_long.parquet")
    copy_df(conn, fin, "staging.stg_yf_financials", ["ticker", "iso3", "statement", "period_end", "item", "value"])

    idx = pd.read_parquet(FLAT / "esef_index.parquet")
    copy_df(conn, idx, "staging.stg_esef_index", ["fxo_id", "country", "lei", "entity_name", "json_url", "date_added"])
    ef = pd.read_parquet(FLAT / "esef_facts_long.parquet")
    copy_df(conn, ef, "staging.stg_esef_facts", ["fxo_id", "lei", "entity_name", "country", "concept",
                                                 "period_start", "period_end", "unit", "value", "iso3"])


def run_sql(conn, path):
    print(f"running {path.relative_to(ROOT)}", flush=True)
    with conn.cursor() as cur:
        cur.execute(path.read_text())


def summary(conn):
    print("\ncore row counts")
    with conn.cursor() as cur:
        for t in ["dim_source", "dim_country", "dim_indicator", "fact_country_year", "dim_company",
                  "fact_company_financial", "company_metric_map"]:
            cur.execute(f"SELECT COUNT(*) FROM core.{t}")
            print(f"  core.{t}: {cur.fetchone()[0]:,}")


def main():
    with psycopg.connect(DSN, autocommit=False) as conn:
        run_sql(conn, ROOT / "src" / "load" / "schema.sql")
        load_staging(conn)
        run_sql(conn, ROOT / "src" / "transform" / "core.sql")
        run_sql(conn, ROOT / "src" / "transform" / "marts.sql")
        conn.commit()
        summary(conn)


if __name__ == "__main__":
    main()
