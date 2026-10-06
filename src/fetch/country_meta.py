"""Fetch descriptive metadata the dimension tables need.

- World Bank country metadata (income group, coordinates, region) -> data/raw/wb_meta/countries.json
- IMF DataMapper indicator labels and units                       -> data/raw/imf/_indicators.json
"""
import json
import pathlib

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    isos = pd.read_csv(ROOT / "config" / "countries.csv").iso3.tolist()
    r = requests.get("https://api.worldbank.org/v2/country/" + ";".join(isos),
                     params={"format": "json", "per_page": 100}, timeout=60)
    r.raise_for_status()
    out = ROOT / "data" / "raw" / "wb_meta" / "countries.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r.json()[1]))
    print("country metadata:", len(r.json()[1]), "countries")

    r = requests.get("https://www.imf.org/external/datamapper/api/v1/indicators", timeout=60)
    r.raise_for_status()
    out = ROOT / "data" / "raw" / "imf" / "_indicators.json"
    out.write_text(json.dumps(r.json()["indicators"]))
    print("IMF indicators:", len(r.json()["indicators"]))


if __name__ == "__main__":
    main()
