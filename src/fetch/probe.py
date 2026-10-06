"""Probe every candidate indicator: does it return data for the 40 countries?

Writes data/profile/probe.csv and prints a summary. Raw responses are saved to
data/raw/ so the later fetch step can reuse them.
"""
import json
import pathlib
import time

import pandas as pd
import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
START, END = 2012, 2024
WB_URL = "https://api.worldbank.org/v2/country/{c}/indicator/{i}"
IMF_URL = "https://www.imf.org/external/datamapper/api/v1/{i}"


def load_config():
    countries = pd.read_csv(ROOT / "config" / "countries.csv")
    indicators = yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text())
    return countries["iso3"].tolist(), indicators


def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def save_raw(source, code, payload):
    out = ROOT / "data" / "raw" / source.lower() / f"{code}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))


def fetch_wb(code, isos, source=None):
    params = {"format": "json", "date": f"{START}:{END}", "per_page": 1000}
    if source:
        params["source"] = source
    url = WB_URL.format(c=";".join(isos), i=code)
    rows, page = [], 1
    while True:
        payload = get_json(url, {**params, "page": page})
        if not isinstance(payload, list) or len(payload) < 2:
            return rows  # API error message, e.g. unknown indicator
        meta, data = payload
        rows += data or []
        if page >= meta["pages"]:
            return rows
        page += 1
        time.sleep(0.3)


def fetch_imf(code, isos):
    # The API ignores country filters in the path, so fetch all and filter here.
    payload = get_json(IMF_URL.format(i=code))
    values = payload.get("values", {}).get(code, {})
    return [
        {"countryiso3code": iso, "date": year, "value": val}
        for iso, series in values.items()
        if iso in isos
        for year, val in series.items()
        if START <= int(year) <= END
    ]


def summarise(source, code, rows, n_countries):
    df = pd.DataFrame(rows)
    if df.empty or "value" not in df:
        return dict(source=source, code=code, countries=0, years=0, pct_filled=0.0, last_year=None)
    df = df.dropna(subset=["value"])
    df["year"] = df["date"].astype(int)
    expected = n_countries * (END - START + 1)
    return dict(
        source=source,
        code=code,
        countries=df["countryiso3code"].nunique(),
        years=df["year"].nunique(),
        pct_filled=round(100 * len(df) / expected, 1),
        last_year=int(df["year"].max()) if len(df) else None,
    )


def main():
    isos, indicators = load_config()
    results = []
    for source, items in indicators.items():
        for item in items:
            code = item["code"]
            try:
                if source == "IMF":
                    rows = fetch_imf(code, isos)
                else:
                    rows = fetch_wb(code, isos, source=3 if source == "WGI" else None)
                save_raw(source, code, rows)
                res = summarise(source, code, rows, len(isos))
            except Exception as exc:  # keep probing the rest
                res = dict(source=source, code=code, countries=0, years=0,
                           pct_filled=0.0, last_year=None, error=str(exc)[:120])
            res["theme"] = item.get("theme")
            res["snapshot_only"] = item.get("snapshot_only", False)
            results.append(res)
            print(f"{source:4} {code:22} countries={res['countries']:>2} "
                  f"filled={res['pct_filled']:>5}% last={res['last_year']}")
            time.sleep(0.3)

    out = ROOT / "data" / "profile" / "probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
