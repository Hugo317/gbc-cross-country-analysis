"""Country x Yahoo-sector panel built from Eurostat (EU, NACE A64) and OECD STAN (non-EU, ISIC4).

build() -> DataFrame  iso3 | sector | year | value_added | net_operating_surplus | compensation_employees | net_fixed_assets | margin | ror | source
margin = net operating surplus / value added (all countries); ror = net operating surplus / net fixed assets (Eurostat countries only).
A sector-country-year is kept only if every slot of the sector has data, so composition does not change over time.
Coarse mapping from industry to the 11 Yahoo sectors: approximate (public administration, education, households and
activities not in the list are excluded; "Technology" and "Consumer" splits are judgment calls).
Run: uv run python src/analysis/sector_panel.py   -> data/flat/sector_yahoo_country_year.parquet and sector_section_country_year.parquet (letter level, includes ror)
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAT = ROOT / "data" / "flat"

# sector -> list of slots; each slot is a list of alternative codes (first one present is used)
NACE = {
    "Consumer Defensive": [["A01"], ["A02"], ["A03"], ["C10-C12"]],
    "Energy": [["B"], ["C19"]],
    "Utilities": [["D35"], ["E36"], ["E37-E39"]],
    "Basic Materials": [["C16"], ["C17"], ["C18"], ["C20"], ["C22"], ["C23"], ["C24"]],
    "Consumer Cyclical": [["C13-C15"], ["C29"], ["C31_C32"], ["G45"], ["G47"], ["I"], ["R90-R92"], ["R93"]],
    "Industrials": [["C25"], ["C27"], ["C28"], ["C30"], ["C33"], ["F"], ["G46"], ["H49"], ["H50"], ["H51"], ["H52"], ["H53"],
                    ["M69_M70"], ["M71"], ["M72"], ["M73"], ["M74_M75"], ["N"]],
    "Technology": [["C26"], ["J58"], ["J62_J63"]],
    "Communication Services": [["J59_J60"], ["J61"]],
    "Healthcare": [["C21"], ["Q86"], ["Q87_Q88"]],
    "Real Estate": [["L68"]],
    "Financial Services": [["K64"], ["K65"], ["K66"]],
}
ISIC = {
    "Consumer Defensive": [["A"], ["C10T12"]],
    "Energy": [["B"], ["C19"]],
    "Utilities": [["D"], ["E"]],
    "Basic Materials": [["C16"], ["C17"], ["C18"], ["C20"], ["C22"], ["C23"], ["C24"]],
    "Consumer Cyclical": [["C13T15"], ["C29"], ["C31_32"], ["G45"], ["G47"], ["I"], ["R90T92"], ["R93"]],
    "Industrials": [["C25"], ["C27"], ["C28"], ["C30"], ["C33"], ["F"], ["G46"], ["H49"], ["H50"], ["H51"], ["H52"], ["H53"],
                    ["M69_70"], ["M71"], ["M72"], ["M73"], ["M74_75"], ["N"]],
    "Technology": [["C26"], ["J58"], ["J62_63"]],
    "Communication Services": [["J59_60"], ["J61"]],
    "Healthcare": [["C21"], ["Q86"], ["Q87_88"]],
    "Real Estate": [["L68A", "L"]],
    "Financial Services": [["K64"], ["K65"], ["K66"]],
}
INDICATORS = ["value_added", "net_operating_surplus", "compensation_employees", "net_fixed_assets"]


def _aggregate(df, code_col, mapping, source):
    wide = df.pivot_table(index=["iso3", "year", code_col], columns="indicator", values="value")
    out = []
    for sector, slots in mapping.items():
        parts = []
        for slot in slots:
            avail = [c for c in slot if c in wide.index.get_level_values(code_col)]
            if not avail:
                parts = None
                break
            parts.append(wide.xs(avail[0], level=code_col))
        if parts is None:
            continue
        cat = pd.concat(parts, keys=range(len(parts)), names=["slot"]).reset_index()
        g = cat.groupby(["iso3", "year"])
        n = g.size()
        sums = g[[c for c in INDICATORS if c in cat]].agg(lambda s: s.sum(min_count=len(s)))
        sums = sums[n.reindex(sums.index) == len(parts)]
        sums["sector"] = sector
        out.append(sums.reset_index())
    res = pd.concat(out)
    res["source"] = source
    return res


def build():
    eu = _aggregate(pd.read_parquet(FLAT / "sector_country_year.parquet"), "nace", NACE, "Eurostat")
    stan = _aggregate(pd.read_parquet(FLAT / "sector_country_year_stan.parquet"), "isic", ISIC, "OECD STAN")
    df = pd.concat([eu, stan], ignore_index=True)
    for c in INDICATORS:
        if c not in df:
            df[c] = pd.NA
    df["margin"] = df.net_operating_surplus / df.value_added
    df["ror"] = df.net_operating_surplus / df.net_fixed_assets
    return df[["iso3", "sector", "year", *INDICATORS, "margin", "ror", "source"]]


def build_sections():
    """NACE/ISIC section level (letters A..S): the only level where net fixed assets exist for most Eurostat countries, so ror lives here."""
    letters = list("ABCDEFGHIJKLMNQR")
    eu = pd.read_parquet(FLAT / "sector_country_year.parquet").rename(columns={"nace": "section"})
    st = pd.read_parquet(FLAT / "sector_country_year_stan.parquet").rename(columns={"isic": "section"})
    eu["source"], st["source"] = "Eurostat", "OECD STAN"
    both = pd.concat([eu[eu.section.isin(letters)], st[st.section.isin(letters)]])
    w = both.pivot_table(index=["iso3", "section", "year", "source"], columns="indicator", values="value").reset_index()
    for c in INDICATORS:
        if c not in w:
            w[c] = pd.NA
    w["margin"] = w.net_operating_surplus / w.value_added
    w["ror"] = w.net_operating_surplus / w.net_fixed_assets
    return w


if __name__ == "__main__":
    build_sections().to_parquet(FLAT / "sector_section_country_year.parquet")
    df = build()
    df.to_parquet(FLAT / "sector_yahoo_country_year.parquet")
    print(len(df), "rows")
    print(df.groupby("iso3").agg(sectors=("sector", "nunique"), y0=("year", "min"), y1=("year", "max"), margin_n=("margin", "count"), ror_n=("ror", "count")).T.to_string())
