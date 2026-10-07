"""Country x sector panels from Eurostat national accounts by industry (NACE A64), 1995 onward.

Output: data/flat/sector_country_year.parquet   iso3 | nace | indicator | year | value (million EUR, current prices)
Raw per-country JSON cache: data/raw/eurostat_sector/<dataset>_<item>_<geo>.json
Kept outside the Postgres schema (the schema is owned by the project owner). EU countries only; the 13 non-EU countries need EU KLEMS / OECD STAN.
"""
import json
import pathlib
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "eurostat_sector"
OUT = ROOT / "data" / "flat" / "sector_country_year.parquet"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
ISO2 = {"AUT": "AT", "BEL": "BE", "CZE": "CZ", "DNK": "DK", "EST": "EE", "FIN": "FI", "FRA": "FR", "DEU": "DE", "GRC": "EL", "HUN": "HU",
        "IRL": "IE", "ITA": "IT", "LVA": "LV", "LTU": "LT", "LUX": "LU", "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO", "ESP": "ES", "SWE": "SE"}
# (dataset, indicator name, filters)
SERIES = [
    ("nama_10_a64", "value_added", {"na_item": "B1G", "unit": "CP_MEUR"}),
    ("nama_10_a64", "net_operating_surplus", {"na_item": "B2A3N", "unit": "CP_MEUR"}),
    ("nama_10_a64", "compensation_employees", {"na_item": "D1", "unit": "CP_MEUR"}),
    ("nama_10_nfa_st", "net_fixed_assets", {"asset10": "N11N", "unit": "CRC_MEUR"}),
]


def fetch(ds, name, flt, iso3, iso2):
    f = RAW / f"{ds}_{name}_{iso3}.json"
    if f.exists():
        return json.loads(f.read_text())
    js = {}
    for attempt in range(3):
        try:
            r = requests.get(API + ds, params={"format": "JSON", "freq": "A", "geo": iso2, **flt}, timeout=120)
            js = r.json()
            if "error" not in js:
                break
            print("  api msg", ds, name, iso3, js["error"][0].get("label", "")[:80])
            js = {}
            time.sleep(5)
        except Exception as exc:
            print("  retry", ds, name, iso3, exc)
            time.sleep(3)
    RAW.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(js))
    return js


def parse(js, iso3, name):
    if not js or "value" not in js:
        return []
    dims, size = js["id"], js["size"]
    idx = {d: list(js["dimension"][d]["category"]["index"]) for d in dims}
    rows = []
    for k, v in js["value"].items():
        k = int(k)
        pos = []
        for s in reversed(size):
            pos.append(k % s)
            k //= s
        cat = {d: idx[d][i] for d, i in zip(dims, reversed(pos))}
        rows.append((iso3, cat["nace_r2"], name, int(cat["time"]), v))
    return rows


def main():
    out = []
    for ds, name, flt in SERIES:
        for iso3, iso2 in ISO2.items():
            rows = parse(fetch(ds, name, flt, iso3, iso2), iso3, name)
            out += rows
        n = sum(1 for r in out if r[2] == name)
        print(f"{name:26s} {n:6d} values")
    df = pd.DataFrame(out, columns=["iso3", "nace", "indicator", "year", "value"])
    df.to_parquet(OUT)
    print(len(df), "rows ->", OUT)


if __name__ == "__main__":
    main()
