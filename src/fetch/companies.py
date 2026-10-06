"""Fetch annual income statements and balance sheets for every ticker in config/companies_seed.csv.

Raw per-ticker result -> data/raw/yfinance_fin/<ticker>.json (cached; rerun skips finished tickers).
Flat long table      -> data/flat/company_financials_long.parquet
    ticker | iso3 | statement | period_end | item | value
Per-ticker metadata  -> data/flat/company_meta.parquet
"""
import json
import pathlib
import time

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "yfinance_fin"
FLAT = ROOT / "data" / "flat"
META_KEYS = ["longName", "country", "sector", "industry", "currency", "financialCurrency",
             "marketCap", "exchange", "fullTimeEmployees"]


def frame_to_records(df, statement):
    """yfinance statement (rows=items, columns=period end) -> list of dicts."""
    out = []
    if df is None or df.empty:
        return out
    for period, col in df.items():
        for item, val in col.items():
            if pd.notna(val):
                out.append({"statement": statement, "period_end": pd.Timestamp(period).strftime("%Y-%m-%d"),
                            "item": str(item), "value": float(val)})
    return out


def fetch_one(ticker):
    cache = RAW / f"{ticker}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    res = {"ticker": ticker, "ok": False, "meta": {}, "records": []}
    for attempt in range(2):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            res["meta"] = {k: info.get(k) for k in META_KEYS}
            res["records"] = (frame_to_records(t.income_stmt, "income")
                              + frame_to_records(t.balance_sheet, "balance"))
            res["ok"] = bool(res["records"])
            res.pop("error", None)
            break
        except Exception as exc:
            res["error"] = str(exc)[:120]
            time.sleep(3)
    RAW.mkdir(parents=True, exist_ok=True)
    if "error" not in res:   # don't cache transient failures
        cache.write_text(json.dumps(res))
    time.sleep(1)
    return res


def main():
    countries = set(pd.read_csv(ROOT / "config" / "countries.csv").iso3)
    seed = pd.read_csv(ROOT / "config" / "companies_seed.csv")
    seed = seed[seed.iso3.isin(countries)]
    rows, meta = [], []
    for i, (ticker, iso3) in enumerate(zip(seed.ticker, seed.iso3), 1):
        res = fetch_one(ticker)
        for r in res["records"]:
            rows.append({"ticker": ticker, "iso3": iso3, **r})
        meta.append({"ticker": ticker, "iso3": iso3, "ok": res["ok"], **res["meta"]})
        if i % 25 == 0:
            print(f"{i}/{len(seed)} done", flush=True)

    FLAT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(FLAT / "company_financials_long.parquet", index=False)
    pd.DataFrame(meta).to_parquet(FLAT / "company_meta.parquet", index=False)
    print(f"Wrote {len(rows)} rows for {sum(m['ok'] for m in meta)}/{len(meta)} tickers with financials")


if __name__ == "__main__":
    main()
