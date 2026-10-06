"""Monthly adjusted close prices (local currency) for every Yahoo company in the DuckDB export.

Used by notebook 1.8 (market outcomes) and 1.10 (shocks). Kept as a flat file, outside the Postgres schema.
Output: data/flat/prices_monthly.parquet   ticker | month | adj_close
Cached per batch in data/raw/yfinance_prices/ so reruns skip finished batches.
"""
import pathlib
import time

import duckdb
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "yfinance_prices"
OUT = ROOT / "data" / "flat" / "prices_monthly.parquet"
START, BATCH = "2018-12-01", 60


def main():
    con = duckdb.connect(str(ROOT / "data" / "gbc.duckdb"), read_only=True)
    tickers = con.sql("SELECT source_key FROM dim_company WHERE source_id = 'YF' ORDER BY 1").df().source_key.tolist()
    RAW.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(0, len(tickers), BATCH):
        f = RAW / f"batch_{i // BATCH:03d}.parquet"
        if not f.exists():
            batch = tickers[i:i + BATCH]
            for attempt in range(3):
                try:
                    px = yf.download(batch, start=START, interval="1mo", auto_adjust=True, progress=False, threads=True)["Close"]
                    break
                except Exception as exc:
                    print("retry", i, exc)
                    time.sleep(5)
            else:
                continue
            if isinstance(px, pd.Series):
                px = px.to_frame(batch[0])
            long = px.rename_axis("month").reset_index().melt("month", var_name="ticker", value_name="adj_close").dropna()
            long.to_parquet(f)
            time.sleep(2)
        parts.append(pd.read_parquet(f))
        print(f"{min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)
    out = pd.concat(parts)
    out["month"] = pd.to_datetime(out.month).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    out = out[out.adj_close > 0].drop_duplicates(["ticker", "month"])
    out.to_parquet(OUT)
    print(out.ticker.nunique(), "tickers,", len(out), "rows ->", OUT)


if __name__ == "__main__":
    main()
