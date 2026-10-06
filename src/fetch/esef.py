"""Fetch EU listed-company financials from filings.xbrl.org (ESEF annual reports, IFRS).

1. Index: list every filing for our EU countries      -> data/flat/esef_index.parquet
2. Select: per country, the top MAX_PER_COUNTRY entities (most filings, then most recent),
   and each entity's latest FILINGS_PER_ENTITY filings.
3. Download each filing's xBRL-JSON (gzipped copy kept in data/raw/esef/), extract a whitelist
   of IFRS concepts at consolidated level (no extra dimensions).
                                                       -> data/flat/esef_facts_long.parquet

Period ends are converted to inclusive dates. Caveats: the API's `period_end` field is unreliable (we saw 2029, 4172), so the real
reporting period is read from the facts. ESEF has no sector field.
"""
import datetime as dt
import gzip
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "https://filings.xbrl.org"
RAW = ROOT / "data" / "raw" / "esef"
FLAT = ROOT / "data" / "flat"
MAX_PER_COUNTRY = 100
FILINGS_PER_ENTITY = 4
WORKERS = 4

CONCEPTS = [
    "Revenue", "RevenueFromContractsWithCustomers", "InterestRevenueExpense", "InterestIncome",
    "GrossProfit", "ProfitLossFromOperatingActivities", "ProfitLossBeforeTax", "ProfitLoss",
    "ProfitLossAttributableToOwnersOfParent", "IncomeTaxExpenseContinuingOperations",
    "DepreciationAndAmortisationExpense", "Assets", "CurrentAssets", "NoncurrentAssets",
    "Liabilities", "CurrentLiabilities", "Equity", "EquityAttributableToOwnersOfParent",
    "CashAndCashEquivalents", "CashFlowsFromUsedInOperatingActivities",
    "CashFlowsFromUsedInInvestingActivities", "CashFlowsFromUsedInFinancingActivities",
    "NumberOfEmployees", "AverageNumberOfEmployees",
]
WANTED = {f"ifrs-full:{c}" for c in CONCEPTS}


def get(url, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def build_index(iso2_list):
    out = RAW.parent.parent / "flat" / "esef_index.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows = []
    for cc in iso2_list:
        page = 1
        while True:
            flt = json.dumps([{"name": "country", "op": "eq", "val": cc}])
            j = get(f"{BASE}/api/filings", params={"filter": flt, "page[size]": 500,
                                                  "page[number]": page, "include": "entity"}).json()
            ent = {e["id"]: e["attributes"] for e in j.get("included", [])}
            data = j.get("data", [])
            for x in data:
                eid = (x.get("relationships", {}).get("entity", {}).get("data") or {}).get("id")
                a = x["attributes"]
                rows.append({"fxo_id": a["fxo_id"], "country": a["country"], "lei": ent.get(eid, {}).get("identifier"),
                             "entity_name": ent.get(eid, {}).get("name"), "json_url": a.get("json_url"),
                             "date_added": a.get("date_added")})
            if len(data) < 500:
                break
            page += 1
        print(f"index {cc}: {sum(r['country'] == cc for r in rows)} filings", flush=True)
    df = pd.DataFrame(rows).dropna(subset=["json_url", "lei"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def select(index):
    parts = []
    for cc, g in index.groupby("country"):
        ranks = (g.groupby("lei").agg(n=("fxo_id", "size"), last=("date_added", "max"))
                 .sort_values(["n", "last"], ascending=False).head(MAX_PER_COUNTRY))
        sub = g[g.lei.isin(ranks.index)].sort_values("date_added", ascending=False)
        parts.append(sub.groupby("lei").head(FILINGS_PER_ENTITY))
    return pd.concat(parts)


def inclusive_end(ts):
    """xBRL-JSON period ends at midnight are exclusive (2022-01-01T00:00:00 = close of 2021-12-31)."""
    day = dt.date.fromisoformat(ts[:10])
    return (day - dt.timedelta(days=1) if ts[10:] == "T00:00:00" else day).isoformat()


def extract(fxo_id, lei, name, country, js):
    rows = []
    for f in js.get("facts", {}).values():
        d = f["dimensions"]
        if d.get("concept") not in WANTED or "unit" not in d:
            continue
        if set(d) - {"concept", "entity", "period", "unit", "language"}:
            continue   # skip dimensional (segment/component) facts
        try:
            val = float(f["value"])
        except (TypeError, ValueError):
            continue
        start, _, end = d["period"].partition("/")
        rows.append({"fxo_id": fxo_id, "lei": lei, "entity_name": name, "country": country,
                     "concept": d["concept"].split(":")[1], "period_start": start[:10] if end else None,
                     "period_end": inclusive_end(end or start), "unit": d["unit"].split(":")[-1], "value": val})
    return rows


def process(row):
    cache = RAW / f"{row.fxo_id.replace('/', '_')}.json.gz"
    try:
        if cache.exists():
            js = json.loads(gzip.decompress(cache.read_bytes()))
        else:
            js = get(BASE + row.json_url).json()
            RAW.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(gzip.compress(json.dumps(js).encode()))
            time.sleep(0.3)
        return extract(row.fxo_id, row.lei, row.entity_name, row.country, js), None
    except Exception as exc:
        return [], f"{row.fxo_id}: {str(exc)[:80]}"


def main():
    countries = pd.read_csv(ROOT / "config" / "countries.csv")
    eu = countries[countries.is_eu]
    iso2_to_iso3 = eu.set_index("iso2")["iso3"].to_dict()
    index = build_index(list(iso2_to_iso3))
    chosen = select(index)
    print(f"selected {chosen.lei.nunique()} entities, {len(chosen)} filings", flush=True)

    rows, errors = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, (r, err) in enumerate(pool.map(process, chosen.itertuples()), 1):
            rows += r
            if err:
                errors.append(err)
            if i % 100 == 0:
                print(f"{i}/{len(chosen)} filings, {len(errors)} errors", flush=True)

    df = pd.DataFrame(rows)
    df["iso3"] = df.country.map(iso2_to_iso3)
    FLAT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FLAT / "esef_facts_long.parquet", index=False)
    (ROOT / "data" / "profile" / "esef_errors.txt").write_text("\n".join(errors))
    print(f"Wrote {len(df)} facts from {df.lei.nunique()} entities; {len(errors)} filing errors")


if __name__ == "__main__":
    main()
