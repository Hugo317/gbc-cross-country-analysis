"""Statistics helpers shared by the hypothesis notebooks (01_1 ... 01_7)."""
import json
import pathlib

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONTROLS = "C(sector) + size_rank + C(fiscal_year)"      # standard firm-level controls


# ───────────── regressions ─────────────
def fit_cluster(formula, data, cluster="iso3"):
    """OLS with standard errors clustered by country."""
    return smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster].values})


def wild_cluster(focal, data, outcome="roa_pp", extra="", controls=CONTROLS, B=1999, seed=0, cluster="iso3"):
    """Coefficient of `focal` with a wild cluster bootstrap p-value (Rademacher weights, null imposed).
    Appropriate with few clusters (here 34 countries), where cluster-robust p-values are too optimistic."""
    need = [outcome, focal] + (["size_rank"] if "size_rank" in controls else [])
    d = data.dropna(subset=need)
    rhs = f"{focal}" + (f" + {extra}" if extra else "") + (f" + {controls}" if controls else "")
    y, X = patsy.dmatrices(f"{outcome} ~ {rhs}", d, return_type="dataframe")
    X = X.loc[:, (X != 0).any(axis=0)]                    # a dummy can end up empty once rows with missing values are dropped
    g = pd.factorize(d.loc[X.index, cluster])[0]          # patsy drops rows with missing values: align the clusters
    o = np.argsort(g, kind="stable")
    Xv, yv, gv = X.values[o], y.values[o, 0], g[o]
    k = list(X.columns).index(focal)
    XtXi = np.linalg.pinv(Xv.T @ Xv)
    starts = np.r_[0, np.flatnonzero(np.diff(gv)) + 1]
    sizes = np.diff(np.r_[starts, len(yv)])
    G, (N, K) = len(starts), Xv.shape
    adj = G / (G - 1) * (N - 1) / (N - K)

    def tstat(yy):
        beta = XtXi @ (Xv.T @ yy)
        u = yy - Xv @ beta
        S = np.add.reduceat(Xv * u[:, None], starts, axis=0)
        V = adj * XtXi @ (S.T @ S) @ XtXi
        return beta[k], beta[k] / np.sqrt(V[k, k]), np.sqrt(V[k, k])

    b, t, se = tstat(yv)
    Xr = np.delete(Xv, k, axis=1)
    fit_r = Xr @ np.linalg.lstsq(Xr, yv, rcond=None)[0]
    res_r = yv - fit_r
    rng = np.random.default_rng(seed)
    hits = sum(abs(tstat(fit_r + np.repeat(rng.choice([-1.0, 1.0], G), sizes) * res_r)[1]) >= abs(t) for _ in range(B))
    p_cr = 2 * (1 - stats.t.cdf(abs(t), G - 1))
    return dict(coef=float(b), se=float(se), ci_low=float(b - 1.96 * se), ci_high=float(b + 1.96 * se), t=float(t),
                p_cluster=float(p_cr), p_wild=float((hits + 1) / (B + 1)), clusters=int(G), n=int(N))


# ───────────── correlations ─────────────
def spearman(x, y, min_n=8):
    m = pd.notna(x) & pd.notna(y)
    if m.sum() < min_n:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return float(r), float(p), int(m.sum())


def boot_ci(x, y, n=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    r = [stats.spearmanr(x[i], y[i])[0] for i in idx]
    return tuple(np.nanpercentile(r, [2.5, 97.5]))


def partial_rho(x, y, c):
    """Spearman-type partial correlation of x and y controlling for c (ranks, linear residuals)."""
    x, y, c = (np.asarray(v, float) for v in (x, y, c))
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(c))
    rx, ry, rc = (stats.rankdata(v[m]) for v in (x, y, c))
    res = lambda a: a - np.polyval(np.polyfit(rc, a, 1), rc)
    return float(stats.pearsonr(res(rx), res(ry))[0])


def bh(pvals):
    """Benjamini-Hochberg q-values; NaN p-values are left as NaN."""
    p = pd.Series(pvals, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.sum():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def scan(feature_table, outcome, features, min_n=12):
    """Spearman of every feature with one country-level outcome (pairwise complete); returns a table with BH q."""
    rows = []
    for f in features:
        r, p, n = spearman(feature_table[f], outcome, min_n)
        rows.append({"feature": f, "rho": r, "p": p, "n": n})
    t = pd.DataFrame(rows).set_index("feature").dropna(subset=["rho"])
    t["q"] = bh(t.p)
    return t.sort_values("rho", key=abs, ascending=False)


# ───────────── country types ─────────────
def country_clusters(profile, var_share=0.80, k_range=(3, 5), min_size=3, seed=0):
    """K-means on the principal components explaining `var_share` of variance; every cluster needs >= min_size
    countries. Clusters are numbered from 1 = highest GDP per person."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    zz = ((profile - profile.mean()) / profile.std()).fillna(0)
    U, S, Vt = np.linalg.svd(zz.values, full_matrices=False)
    npc = int(np.searchsorted(np.cumsum(S**2 / (S**2).sum()), var_share) + 1)
    pcs = U[:, :npc] * S[:npc]
    sil, labs = {}, {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(k, n_init=50, random_state=seed).fit(pcs)
        sil[k], labs[k] = silhouette_score(pcs, km.labels_), km.labels_
    valid = [k for k in sil if np.bincount(labs[k]).min() >= min_size]
    best = max(valid or list(sil), key=lambda k: sil[k])
    lab = pd.Series(labs[best], index=profile.index)
    order = profile.gdp_per_capita_ppp.groupby(lab).mean().sort_values(ascending=False).index
    lab = lab.map({c: i + 1 for i, c in enumerate(order)})
    return lab, dict(k=best, silhouette=sil, n_components=npc)


# ───────────── results registry (read by 01_hypotheses.ipynb) ─────────────
def save_result(hid, **fields):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{hid}.json").write_text(json.dumps(fields, indent=2, default=float))


def load_results():
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))} if RESULTS.exists() else {}


def verdict(coef, p_wild, expected_sign, robust_share, clusters, alpha=0.05):
    """Pre-agreed rule. expected_sign: +1, -1 or 0 (direction not predicted).
    supported        : expected direction (or any if 0), wild p < alpha, and the sign holds in >= 80% of robustness checks
    partly supported : expected direction and (wild p < 0.10, or p < alpha but robustness < 80%)
    contradicted     : significant (p < alpha) in the OPPOSITE direction to the prediction
    not supported    : otherwise (no clear effect)"""
    if clusters < 15:
        return "inconclusive"
    sign_ok = expected_sign == 0 or np.sign(coef) == expected_sign
    if not sign_ok and p_wild < alpha:
        return "contradicted"
    if sign_ok and p_wild < alpha and robust_share >= 0.8:
        return "supported"
    if sign_ok and (p_wild < 0.10 or p_wild < alpha):
        return "partly supported"
    return "not supported"
