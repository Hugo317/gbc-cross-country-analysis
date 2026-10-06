"""Shared data preparation for the analysis notebooks (reads data/gbc.duckdb).

country  : load_country()  -> panel (country-year), profile (2019-2023 mean per country), feature lists
firms    : load_firms()    -> Yahoo company-years with winsorised ratios and the PREVIOUS year's
                              country features (raw and standardised as *_z)
"""
import pathlib

import duckdb
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE_YEARS = (2019, 2023)   # window averaged for the country profile
WINSOR = (0.01, 0.99)          # firm ratios are clipped to these percentiles
MAX_MISSING = 0.25             # drop country features missing for more than 25% of countries
MIN_FIRMS = 10                 # a country needs this many companies to get a country-level outcome

# (short name, indicator_id, transform, group). Where sources overlap, one is kept
# (World Bank growth/inflation/unemployment; IMF duplicates are not used).
FEATURES = [
    ("gdp_per_capita_ppp", "WB:NY.GDP.PCAP.PP.KD", "log", "economy"), ("gdp_usd", "WB:NY.GDP.MKTP.CD", "log", "economy"),
    ("gdp_growth", "WB:NY.GDP.MKTP.KD.ZG", None, "economy"), ("inflation", "WB:FP.CPI.TOTL.ZG", None, "economy"),
    ("manufacturing_share", "WB:NV.IND.MANF.ZS", None, "economy"), ("services_share", "WB:NV.SRV.TOTL.ZS", None, "economy"),
    ("trade_gdp", "WB:NE.TRD.GNFS.ZS", None, "trade"), ("fdi_gdp", "WB:BX.KLT.DINV.WD.GD.ZS", None, "trade"),
    ("hightech_exports", "WB:TX.VAL.TECH.MF.ZS", None, "trade"),
    ("private_credit_gdp", "WB:FS.AST.PRVT.GD.ZS", None, "finance"), ("lending_rate", "WB:FR.INR.LEND", None, "finance"),
    ("tax_revenue_gdp", "WB:GC.TAX.TOTL.GD.ZS", None, "fiscal"), ("gov_debt_gdp", "IMF:GGXWDG_NGDP", None, "fiscal"),
    ("current_account_gdp", "IMF:BCA_NGDPD", None, "economy"),
    ("unemployment", "WB:SL.UEM.TOTL.ZS", None, "labour"), ("youth_unemployment", "WB:SL.UEM.1524.ZS", None, "labour"),
    ("labour_participation", "WB:SL.TLF.CACT.ZS", None, "labour"), ("self_employed", "WB:SL.EMP.SELF.ZS", None, "labour"),
    ("population", "WB:SP.POP.TOTL", "log", "social"), ("pop_working_age", "WB:SP.POP.1564.TO.ZS", None, "social"),
    ("pop_65plus", "WB:SP.POP.65UP.TO.ZS", None, "social"), ("urban_share", "WB:SP.URB.TOTL.IN.ZS", None, "social"),
    ("life_expectancy", "WB:SP.DYN.LE00.IN", None, "social"), ("gini", "WB:SI.POV.GINI", None, "social"),
    ("tertiary_enrolment", "WB:SE.TER.ENRR", None, "education"), ("education_spend_gdp", "WB:SE.XPD.TOTL.GD.ZS", None, "education"),
    ("rd_gdp", "WB:GB.XPD.RSDV.GD.ZS", None, "innovation"), ("patents_residents", "WB:IP.PAT.RESD", "log", "innovation"),
    ("internet_users", "WB:IT.NET.USER.ZS", None, "infrastructure"), ("mobile_subscriptions", "WB:IT.CEL.SETS.P2", None, "infrastructure"),
    ("gov_effectiveness", "WGI:GOV_WGI_GE.EST", None, "governance"), ("regulatory_quality", "WGI:GOV_WGI_RQ.EST", None, "governance"),
    ("rule_of_law", "WGI:GOV_WGI_RL.EST", None, "governance"), ("corruption_control", "WGI:GOV_WGI_CC.EST", None, "governance"),
    ("voice_accountability", "WGI:GOV_WGI_VA.EST", None, "governance"), ("political_stability", "WGI:GOV_WGI_PV.EST", None, "governance"),
]
# features behind the two summary indices (equal-weight mean of standardised features)
DEV_FEATURES = ["gdp_per_capita_ppp", "life_expectancy", "internet_users", "rule_of_law", "gov_effectiveness", "corruption_control"]
INST_FEATURES = ["gov_effectiveness", "regulatory_quality", "rule_of_law", "corruption_control", "voice_accountability", "political_stability"]
# country-level business outcomes ("level A"), kept apart from the features
OUTCOMES_A = [("new_business_density", "WB:IC.BUS.NDNS.ZS", None), ("market_cap_gdp", "WB:CM.MKT.LCAP.GD.ZS", None),
              ("listed_companies", "WB:CM.MKT.LDOM.NO", "log")]


def connect(read_only=True):
    return duckdb.connect(str(ROOT / "data" / "gbc.duckdb"), read_only=read_only)


def load_country(con):
    """Return a dict with dim, panel, profile_all, profile, feat_cols, group, dropped."""
    dim = con.sql("SELECT * FROM dim_country").df().set_index("iso3")
    group = {s: g for s, i, t, g in FEATURES}
    spec = {s: (i, t) for s, i, t, g in FEATURES} | {s: (i, t) for s, i, t in OUTCOMES_A}
    ids = ",".join(f"'{i}'" for i, t in spec.values())
    fc = con.sql(f"SELECT iso3, indicator_id, year, value FROM fact_country_year WHERE indicator_id IN ({ids})").df()
    id2short = {i: s for s, (i, t) in spec.items()}
    panel = fc.assign(short=fc.indicator_id.map(id2short)).pivot_table(index=["iso3", "year"], columns="short", values="value")
    panel.columns.name = None
    for s, (i, t) in spec.items():
        if t == "log":
            panel[s] = np.log(panel[s].where(panel[s] > 0))
    panel = panel.reset_index()
    win = panel[panel.year.between(*PROFILE_YEARS)]
    profile_all = win.groupby("iso3")[list(spec)].agg(lambda s: s.mean() if s.notna().sum() >= 3 else np.nan)
    coverage = profile_all.notna().mean()
    dropped = [s for s in group if coverage[s] < 1 - MAX_MISSING]
    feat_cols = [s for s in group if s not in dropped]
    profile = profile_all[feat_cols]
    zp = (profile - profile.mean()) / profile.std()
    out = dict(dim=dim, panel=panel, profile_all=profile_all, profile=profile, feat_cols=feat_cols, group=group,
               dropped=dropped, outcomes=[o[0] for o in OUTCOMES_A])
    for name, cols in (("dev", DEV_FEATURES), ("inst", INST_FEATURES)):
        raw = zp[cols].mean(axis=1)
        out[name + "_scale"] = raw.std()
        out[name + "_idx"] = (raw - raw.mean()) / raw.std()      # standardised: mean 0, SD 1 across countries
    return out


def load_firms(con, country, source="YF"):
    """Company-years of one source (YF = Yahoo, the default; ESEF = EU filings, no sector) with winsorised outcomes and previous-year country features (*_z = standardised)."""
    panel, profile, feat_cols = country["panel"], country["profile"], country["feat_cols"]
    dim = country["dim"]
    cy = con.sql(f"SELECT * FROM v_company_year WHERE source_id = '{source}' AND roa IS NOT NULL").df()
    for col in ["roa", "operating_margin", "revenue_growth"]:
        lo, hi = cy[col].quantile(WINSOR)
        cy[col + "_w"] = cy[col].clip(lo, hi)
    cy["sector"] = cy.sector.fillna("Unknown")
    cy["roa_pp"] = cy.roa_w * 100                                   # ROA in percentage points
    cy["roa_sector_adj_pp"] = (cy.roa_w - cy.groupby("sector").roa_w.transform("median")) * 100
    cy["country"] = cy.iso3.map(dim.name)
    # extra ratios (winsorised like the others)
    cy["turnover"] = cy.revenue / cy.total_assets                       # sales per unit of assets
    cy["equity_ratio"] = cy.equity / cy.total_assets                    # share of assets financed by equity
    cy["net_margin"] = cy.net_income / cy.revenue.where(cy.revenue > 0)
    for col in ["turnover", "equity_ratio", "net_margin"]:
        lo, hi = cy[col].quantile(WINSOR)
        cy[col + "_w"] = cy[col].clip(lo, hi)
    cy["loss"] = (cy.net_income < 0).astype(float)
    cy["loss_pp"] = cy.loss * 100                                       # share of loss-making firm-years, in %
    cy["is_financial"] = cy.sector.eq("Financial Services")
    # size without exchange rates: percentile of total assets within country and fiscal year
    # (within the country over all years when the country-year has fewer than 10 companies)
    g = cy.groupby(["iso3", "fiscal_year"]).total_assets
    cy["size_rank"] = np.where(g.transform("count") >= 10, g.rank(pct=True), cy.groupby("iso3").total_assets.rank(pct=True))
    # previous year's country features (t-1; if missing, t-2)
    grid = pd.MultiIndex.from_product([panel.iso3.unique(), range(2010, 2027)], names=["iso3", "year"])
    full = panel.set_index(["iso3", "year"]).reindex(grid)[feat_cols]
    lag = full.groupby(level="iso3").ffill(limit=1).groupby(level="iso3").shift(1)
    fy = cy.merge(lag.reset_index().rename(columns={"year": "fiscal_year"}), on=["iso3", "fiscal_year"], how="left")
    mu, sd = profile.mean(), profile.std()
    for f in feat_cols:
        fy[f + "_z"] = (fy[f] - mu[f]) / sd[f]
    # summary indices of the previous year's conditions, in SDs across countries
    fy["dev_z"] = fy[[f + "_z" for f in DEV_FEATURES]].mean(axis=1) / country["dev_scale"]
    fy["inst_z"] = fy[[f + "_z" for f in INST_FEATURES]].mean(axis=1) / country["inst_scale"]
    # same-year national growth and inflation (for the growth analysis)
    same = panel[["iso3", "year", "gdp_growth", "inflation"]].rename(columns={"year": "fiscal_year", "gdp_growth": "gdp_growth_now", "inflation": "inflation_now"})
    fy = fy.merge(same, on=["iso3", "fiscal_year"], how="left")
    fy["growth_real_w"] = (1 + fy.revenue_growth_w) / (1 + fy.inflation_now / 100) - 1   # revenue growth net of national inflation
    fy["growth_pp"] = fy.revenue_growth_w * 100
    fy["growth_real_pp"] = fy.growth_real_w * 100
    return fy


def country_year_lagged(country, first=2012, last=2026):
    """Country-year table of the PREVIOUS year's features (same rule as load_firms: t-1, else t-2), standardised (*_z), with dev_z / inst_z.
    Column `year` is the year the firm outcome refers to; used by notebooks that work with calendar years (prices, shocks)."""
    panel, profile, feat_cols = country["panel"], country["profile"], country["feat_cols"]
    grid = pd.MultiIndex.from_product([panel.iso3.unique(), range(first - 2, last + 1)], names=["iso3", "year"])
    full = panel.set_index(["iso3", "year"]).reindex(grid)[feat_cols]
    lag = full.groupby(level="iso3").ffill(limit=1).groupby(level="iso3").shift(1).reset_index()
    mu, sd = profile.mean(), profile.std()
    for f in feat_cols:
        lag[f + "_z"] = (lag[f] - mu[f]) / sd[f]
    lag["dev_z"] = lag[[f + "_z" for f in DEV_FEATURES]].mean(axis=1) / country["dev_scale"]
    lag["inst_z"] = lag[[f + "_z" for f in INST_FEATURES]].mean(axis=1) / country["inst_scale"]
    return lag[lag.year >= first]
