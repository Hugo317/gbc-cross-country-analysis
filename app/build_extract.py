"""Build the small, self-contained data extract the dashboard reads (so it can be hosted without data/gbc.duckdb).

Run from the repo root after the analysis is updated:  uv run --group dash python app/build_extract.py
Writes app/data/{country,firms}.parquet, app/data/extra.json and copies results/H*.json, W*.json and AB1.json.
"""
import json
import pathlib
import shutil
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from src.analysis import prep  # noqa: E402
import utils  # noqa: E402
import within, compare  # noqa: E402

OUT = ROOT / "app" / "data"
OUT.mkdir(exist_ok=True)

con = prep.connect()
C = prep.load_country(con)
F = prep.load_firms(con, C)

agg = F.groupby("iso3").agg(firms=("company_id", "nunique"), roa=("roa_pp", "mean"), margin=("net_margin_w", "mean"), loss=("loss_pp", "mean"))
country = C["dim"].join(agg).join(C["profile"])
country["dev_index"], country["inst_index"] = C["dev_idx"], C["inst_idx"]
country = country[country.firms >= prep.MIN_FIRMS]
country.index.name = "iso3"
country.reset_index().to_parquet(OUT / "country.parquet")

keep = ["company_id", "iso3", "sector", "fiscal_year", "roa_pp", "net_margin_w", "loss_pp", "size_rank", "dev_z"]
F[keep].dropna(subset=["roa_pp"]).to_parquet(OUT / "firms.parquet")

# development -> ROA by company size (H13), computed once here instead of at app start
d = F.dropna(subset=["roa_pp", "net_margin_w", "loss_pp", "size_rank", "dev_z"]).copy()
d["size_group"] = pd.cut(d.size_rank, [0, 1 / 3, 2 / 3, 1.0001], labels=["small", "medium", "large"], include_lowest=True)
rows = []
for g, gd in d.groupby("size_group", observed=True):
    m = utils.fit_cluster("roa_pp ~ dev_z + C(sector) + C(fiscal_year)", gd)
    rows.append({"size": g, "coef": m.params["dev_z"], "lo": m.params["dev_z"] - 1.96 * m.bse["dev_z"], "hi": m.params["dev_z"] + 1.96 * m.bse["dev_z"]})
d["dev_group"] = pd.qcut(d.iso3.map(C["dev_idx"]), 3, labels=["least developed", "middle", "most developed"])
grid = d.pivot_table(index="size_group", columns="dev_group", values="roa_pp", aggfunc="median", observed=True)
cnt = d.pivot_table(index="size_group", columns="dev_group", values="roa_pp", aggfunc="count", observed=True)

# company-level comparisons inside countries (notebooks 3.1 and 2.1)
names = C["dim"].name
nf = F.groupby("iso3").company_id.nunique()
Fw = F[F.iso3.isin(nf[nf >= within.MIN_FIRMS].index)]
TERMS = {"W1": "equity_ratio_w", "W2": "size_rank"}
w_slopes, w_summary = [], {}
for h, t in TERMS.items():
    s = within.country_slopes(Fw, [t])
    s = s[s.term == t].assign(trait=h, country=lambda d: d.iso3.map(names), iso2=lambda d: d.iso3.map(C["dim"].iso2))
    w_slopes.append(s)
    r = within.repeats(s)
    w_summary[h] = {**r["re"], **{k: v for k, v in r.items() if k != "re"}}
sp = within.sector_profile(Fw)
ag = within.sector_agreement(sp)
sector = {"mean": sp.mean().round(3).to_dict(), "countries": sp.notna().sum().to_dict(), "share_positive": ((sp > 0).sum() / sp.notna().sum()).round(3).to_dict(),
          "agreement": float(ag.mean()), "agreement_positive": int((ag > 0).sum()), "agreement_n": int(len(ag)), "placebo": float(pd.Series(within.sector_placebo(Fw)).mean())}
cm = compare._country_means(F.dropna(subset=["roa_pp"]), "roa_pp", utils.CONTROLS)
eu = pd.DataFrame({"iso3": cm.index, "name": cm.index.map(names), "adj_roa": cm.values, "eu": cm.index.map(C["dim"].is_eu).astype(bool)})
ctrl = utils.wild_cluster("eu", F.assign(eu=F.iso3.map(C["dim"].is_eu).astype(float)), outcome="roa_pp", extra="dev_z", B=999)
company = {"eu_controlled": {"coef": ctrl["coef"], "p": ctrl["p_wild"]}, "slopes": pd.concat(w_slopes)[["iso3", "iso2", "country", "trait", "coef", "se", "n", "firms"]].to_dict(orient="records"), "summary": w_summary, "sector": sector,
           "eu_countries": eu.to_dict(orient="records")}

hyp = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("H*.json")}
fam = pd.Series({h: r["p_primary"] for h, r in hyp.items() if h not in {"H11", "H15"}})
extra = {"size_effects": rows, "grid": grid.round(3).to_dict(orient="split"), "grid_n": cnt.to_dict(orient="split"),
         "q": utils.bh(fam).to_dict(), "feat_cols": C["feat_cols"], "group": C["group"],
         "company": company, "years": [int(F.fiscal_year.min()), int(F.fiscal_year.max())], "n_companies": int(F.company_id.nunique()), "n_firm_years": int(len(F))}
(OUT / "extra.json").write_text(json.dumps(extra))
res = OUT / "results"
res.mkdir(exist_ok=True)
for p in [*(ROOT / "results").glob("H*.json"), *(ROOT / "results").glob("W*.json"), ROOT / "results" / "AB1.json"]:
    shutil.copy(p, res / p.name)
print("ok", {p.name: p.stat().st_size for p in OUT.iterdir() if p.is_file()})
