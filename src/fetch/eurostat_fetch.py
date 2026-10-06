"""Fetch the Eurostat datasets for the EU-27 and flatten them to one long table.

Raw JSON-stat responses go to data/raw/eurostat/<dataset>.json.
The flat table goes to data/flat/eurostat_long.parquet.

Every dataset is filtered down to explicit dimension values so no duplicate rows
appear. Edit SPECS to add or change a dataset.
"""
import itertools
import json
import pathlib
import time

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}"
SINCE = 2012

# dataset -> filters on the non-geo/time dimensions (lists allowed)
SPECS = {
    # country-level corporate performance (non-financial corporations), covers all EU-27
    "nasa_10_ki": {"unit": ["PC"], "sector": ["S11"],
                   "na_item": ["B2G_B3G_RAT_S11", "ROCE_S11", "ROE_S11", "IRG_S11", "DIR_S11"]},
    "nama_10_pc": {"unit": ["CP_PPS_EU27_2020_HAB", "CP_EUR_HAB"], "na_item": ["B1GQ"]},
    "une_rt_a": {"unit": ["PC_ACT"], "sex": ["T"], "age": ["Y15-74", "Y15-24"]},
    "prc_hicp_aind": {"unit": ["RCH_A_AVG"], "coicop": ["CP00"]},
    "lc_lci_lev": {"unit": ["EUR"], "lcstruct": ["D1_D4_MD5"], "nace_r2": ["B-S_X_O"]},
    "earn_mw_cur": {"currency": ["EUR", "PPS"]},
    "rd_e_gerdtot": {"sectperf": ["TOTAL"], "unit": ["PC_GDP", "EUR_HAB"]},
    "isoc_ci_ifp_iu": {"indic_is": ["I_IU3"], "unit": ["PC_IND"], "ind_type": ["IND_TOTAL"]},
    "isoc_eb_ai": {"size_emp": ["GE10"], "unit": ["PC_ENT"], "indic_is": ["E_AI_TANY"]},
    # business structure: legacy table (to 2020) and current table (2021+), different codes
    "sbs_na_sca_r2": {"nace_r2": ["B-N_S95_X_K"],
                      "indic_sb": ["V11110", "V12110", "V12150", "V16110", "V92110", "V91110"]},
    "sbs_sc_ovw": {"nace_r2": ["B-S_X_O_S94"], "size_emp": ["TOTAL"],
                   "indic_sbs": ["ENT_NR", "EMP_NR", "NETTUR_MEUR", "AV_MEUR", "GOR_PC", "AV_SAL_TEUR"]},
    # business demography (births/deaths/survival), current methodology
    "bd_size": {"nace_r2": ["B-S_X_O_S94"], "age": ["TOTAL"], "sizeclas": ["TOTAL"],
                "indic_sbs": ["ENT_NR", "ENT_BRTH_NR", "ENT_DTH_NR", "ENT_BRTHR_PC", "ENT_DTHR_PC",
                              "ENT_SRVLR_BRTH_CHB_PC"]},
}


def countries():
    df = pd.read_csv(ROOT / "config" / "countries.csv").dropna(subset=["eurostat_code"])
    return df.set_index("eurostat_code")["iso3"].to_dict()


def fetch(dataset, geos, filters):
    params = [("format", "JSON"), ("lang", "EN"), ("sinceTimePeriod", str(SINCE))]
    params += [("geo", g) for g in geos]
    params += [(k, v) for k, vals in filters.items() for v in vals]
    for attempt in range(3):
        r = requests.get(BASE.format(dataset), params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 413:
            raise RuntimeError("response too large; tighten the filters")
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


def flatten(dataset, js, geo_to_iso3):
    """JSON-stat -> long DataFrame (one row per observation)."""
    ids, sizes = js["id"], js["size"]
    cats = {d: list(js["dimension"][d]["category"]["index"]) for d in ids}
    labels = {d: js["dimension"][d]["category"].get("label", {}) for d in ids}
    status = js.get("status", {})
    rows = []
    for flat_idx, combo in enumerate(itertools.product(*[range(s) for s in sizes])):
        key = str(flat_idx)
        if key not in js["value"]:
            continue
        rec = {d: cats[d][i] for d, i in zip(ids, combo)}
        rec["value"] = js["value"][key]
        rec["flag"] = status.get(key)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "dataset", dataset)
    df["iso3"] = df["geo"].map(geo_to_iso3)
    df = df.rename(columns={"time": "period"}).drop(columns=["freq"], errors="ignore")
    return df, {d: labels[d] for d in ids}


def main():
    geo_map = countries()
    raw_dir = ROOT / "data" / "raw" / "eurostat"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset, filters in SPECS.items():
        try:
            js = fetch(dataset, list(geo_map), filters)
            (raw_dir / f"{dataset}.json").write_text(json.dumps(js))
            out = flatten(dataset, js, geo_map)
            if isinstance(out, pd.DataFrame):
                print(f"{dataset:16} EMPTY")
                continue
            df, _ = out
            frames.append(df)
            print(f"{dataset:16} rows={len(df):>6} countries={df['iso3'].nunique():>2} "
                  f"periods={df['period'].min()}..{df['period'].max()}")
        except Exception as exc:
            print(f"{dataset:16} FAILED: {exc}")
        time.sleep(0.5)

    # dimensions differ per dataset, so keep them as a JSON string column in the flat table
    base = ["dataset", "iso3", "geo", "period", "value", "flag"]
    long = []
    for df in frames:
        dims = [c for c in df.columns if c not in base]
        df = df.assign(dims=df[dims].apply(lambda r: json.dumps(r.to_dict(), sort_keys=True), axis=1))
        long.append(df[base + ["dims"]])
    all_df = pd.concat(long, ignore_index=True)
    out = ROOT / "data" / "flat" / "eurostat_long.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out, index=False)
    print(f"\nWrote {out}: {len(all_df)} rows")


if __name__ == "__main__":
    main()
