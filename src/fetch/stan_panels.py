"""Country x sector panels for the 13 non-EU countries from OECD STAN (2025 edition, ISIC Rev.4).

Output: data/flat/sector_country_year_stan.parquet   iso3 | isic | indicator | year | value | price_base
Raw download cache: data/raw/oecd_stan/stan_<measure>.csv
Same indicator names as src/fetch/sector_panels.py but without net_fixed_assets (STAN capital stock has no current-price series).
Coverage of net operating surplus: AUS, JPN, KOR, USA only; value added also CAN, CHL, MEX. BRA, CHN, IND, SGP, IDN, ZAF are not in STAN.
Values are in national currency (ratios are unit-free); kept outside the Postgres schema.
"""
import pathlib

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "oecd_stan"
OUT = ROOT / "data" / "flat" / "sector_country_year_stan.parquet"
URL = "https://sdmx.oecd.org/public/rest/data/OECD.STI.PIE,DSD_STAN@DF_STAN_2025,/A.{areas}.....?startPeriod=1990&format=csvfile"
AREAS = "USA+CAN+MEX+BRA+CHL+JPN+KOR+CHN+IND+SGP+IDN+AUS+ZAF"
MEASURES = {"B1G": "value_added", "B2A3N": "net_operating_surplus", "D1": "compensation_employees"}  # STAN capital stock is volume-only, so no rate of return here: use the operating margin


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / "stan_all.csv"
    if not f.exists():
        r = requests.get(URL.format(areas=AREAS), timeout=300)
        r.raise_for_status()
        f.write_bytes(r.content)
    df = pd.read_csv(f, usecols=["REF_AREA", "ACTIVITY", "MEASURE", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"], low_memory=False)
    df = df[df.MEASURE.isin(MEASURES) & (df.PRICE_BASE == "V")]
    df["indicator"] = df.MEASURE.map(MEASURES)
    df["value"] = df.OBS_VALUE * 10.0 ** df.UNIT_MULT.fillna(0)
    out = df.rename(columns={"REF_AREA": "iso3", "ACTIVITY": "isic", "TIME_PERIOD": "year", "PRICE_BASE": "price_base"})[
        ["iso3", "isic", "indicator", "year", "value", "price_base"]].dropna(subset=["value"])
    out["year"] = out.year.astype(int)
    out.to_parquet(OUT)
    print(len(out), "rows ->", OUT)


if __name__ == "__main__":
    main()
