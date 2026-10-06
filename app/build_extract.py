"""Build the small, self-contained data extract the dashboard reads (so it can be hosted without data/gbc.duckdb).

Run from the repo root after the analysis is updated:  uv run --group dash python app/build_extract.py
Writes app/data/{country,firms}.parquet, app/data/extra.json and copies results/H*.json.
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

hyp = {p.stem: json.loads(p.read_text()) for p in (ROOT / "results").glob("H*.json")}
fam = pd.Series({h: r["p_primary"] for h, r in hyp.items() if h not in {"H11", "H15"}})
extra = {"size_effects": rows, "grid": grid.round(3).to_dict(orient="split"), "grid_n": cnt.to_dict(orient="split"),
         "q": utils.bh(fam).to_dict(), "feat_cols": C["feat_cols"], "group": C["group"],
         "years": [int(F.fiscal_year.min()), int(F.fiscal_year.max())], "n_companies": int(F.company_id.nunique()), "n_firm_years": int(len(F))}
(OUT / "extra.json").write_text(json.dumps(extra))
res = OUT / "results"
res.mkdir(exist_ok=True)
for p in (ROOT / "results").glob("H*.json"):
    shutil.copy(p, res / p.name)
print("ok", {p.name: p.stat().st_size for p in OUT.iterdir() if p.is_file()})
