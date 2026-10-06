"""Extra World Bank indicators for notebook 1.16 (tax, energy/climate, FX, banking, investment).

Output: data/flat/extra_country_year.parquet   iso3 | indicator | year | value, plus data/raw/wb_extra/<code>.json (cache).
Kept outside the Postgres schema (the schema is owned by the project owner); notebook 1.16 reads it directly.
"""
import json
import pathlib
import time

import duckdb
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "wb_extra"
OUT = ROOT / "data" / "flat" / "extra_country_year.parquet"
CODES = {
    "IC.TAX.TOTL.CP.ZS": "total_tax_rate_pct_profit", "GC.TAX.YPKG.ZS": "taxes_on_income_share_of_revenue",
    "EN.GHG.CO2.PC.CE.AR5": "co2_per_capita", "EG.FEC.RNEW.ZS": "renewable_share", "EG.EGY.PRIM.PP.KD": "energy_intensity",
    "PA.NUS.FCRF": "fx_lcu_per_usd", "FR.INR.RINR": "real_interest_rate", "FB.AST.NPER.ZS": "bank_npl_share",
    "NE.GDI.TOTL.ZS": "gross_capital_formation", "NE.CON.GOVT.ZS": "gov_consumption", "SL.TLF.CACT.FE.ZS": "female_participation",
    "CM.MKT.TRNR": "stock_turnover", "IT.NET.BBND.P2": "broadband_per_100", "SE.SEC.ENRR": "secondary_enrolment",
}


def fetch(code, iso3s):
    f = RAW / f"{code}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = f"https://api.worldbank.org/v2/country/{';'.join(iso3s)}/indicator/{code}"
    rows = []
    for attempt in range(3):
        try:
            r = requests.get(url, params={"format": "json", "per_page": 20000, "date": "2010:2025"}, timeout=60)
            r.raise_for_status()
            js = r.json()
            rows = js[1] if len(js) > 1 and js[1] else []
            break
        except Exception as exc:
            print("retry", code, exc)
            time.sleep(3)
    RAW.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    return rows


def main():
    iso3s = duckdb.connect(str(ROOT / "data" / "gbc.duckdb"), read_only=True).sql("SELECT iso3 FROM dim_country").df().iso3.tolist()
    out = []
    for code, short in CODES.items():
        rows = fetch(code, iso3s)
        got = [(r["countryiso3code"], short, int(r["date"]), r["value"]) for r in rows if r["value"] is not None]
        print(f"{short:34s} {len(got):5d} values, {len({g[0] for g in got})} countries")
        out += got
    df = pd.DataFrame(out, columns=["iso3", "indicator", "year", "value"])
    df.to_parquet(OUT)
    print(len(df), "rows ->", OUT)


if __name__ == "__main__":
    main()
