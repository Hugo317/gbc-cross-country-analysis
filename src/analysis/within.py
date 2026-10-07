"""Within-country company comparisons and their replication across countries.

Design: inside one country, country conditions are shared by all companies, so differences in ROA between
companies cannot come from them. We estimate the same company-level relationship separately in every country,
then ask whether it repeats: random-effects pooling across countries (country = replication unit),
share of countries with the same sign, and leave-one-country-out agreement.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

MIN_FIRMS = 30          # companies a country needs for its own estimate (standard errors are clustered by company)


def country_slopes(firms, terms, outcome="roa_pp", controls="C(sector) + C(fiscal_year)", min_firms=MIN_FIRMS):
    """OLS of `outcome` on `terms` + controls, one regression per country, errors clustered by company.
    Returns a table (index iso3) with coef/se per term in long form: columns term, coef, se, n, firms."""
    rows = []
    rhs = " + ".join(list(terms) + ([controls] if controls else []))
    for iso, g in firms.dropna(subset=[outcome] + list(terms)).groupby("iso3"):
        if g.company_id.nunique() < min_firms:
            continue
        m = smf.ols(f"{outcome} ~ {rhs}", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.company_id)[0]})
        for t in terms:
            rows.append({"iso3": iso, "term": t, "coef": m.params[t], "se": m.bse[t], "n": int(m.nobs), "firms": g.company_id.nunique()})
    return pd.DataFrame(rows)


def random_effects(coef, se):
    """DerSimonian-Laird random-effects mean of per-country estimates, with the Hartung-Knapp interval and p
    (t with k-1 df: suited to a modest number of countries). Also tau (spread of true country slopes) and I2."""
    b, v = np.asarray(coef, float), np.asarray(se, float) ** 2
    k, w = len(b), 1 / v
    fixed = (w * b).sum() / w.sum()
    Q = (w * (b - fixed) ** 2).sum()
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * b).sum() / ws.sum()
    se_hk = np.sqrt((ws * (b - mu) ** 2).sum() / ((k - 1) * ws.sum()))
    tcrit = stats.t.ppf(0.975, k - 1)
    return dict(mean=float(mu), ci_low=float(mu - tcrit * se_hk), ci_high=float(mu + tcrit * se_hk), p=float(2 * stats.t.sf(abs(mu / se_hk), k - 1)),
                tau=float(np.sqrt(tau2)), I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, k=int(k), Q_p=float(stats.chi2.sf(Q, k - 1)))


def repeats(slopes):
    """How often a company-level relationship repeats across countries. `slopes`: coef/se per country (one term).
    Returns same-sign share, binomial sign-test p, and the leave-one-country-out agreement: for each country, does the
    random-effects mean of the OTHER countries have the same sign as its own estimate?"""
    s = slopes.set_index("iso3")
    re = random_effects(s.coef, s.se)
    sign = np.sign(re["mean"])
    same = (np.sign(s.coef) == sign)
    sig = (s.coef / s.se).abs() > 1.96
    loo = pd.Series({c: np.sign(random_effects(s.coef.drop(c), s.se.drop(c))["mean"]) == np.sign(s.coef[c]) for c in s.index})
    return dict(re=re, same_sign_share=float(same.mean()), sign_test_p=float(stats.binomtest(int(same.sum()), len(s), 0.5).pvalue),
                significant_same_sign=int((same & sig).sum()), significant_opposite=int((~same & sig).sum()), loo_agreement=float(loo.mean()))


def sector_profile(firms, min_obs=8, outcome="roa_pp"):
    """Country x sector table of the sector's median outcome minus the country median (sectors with >= min_obs company-years)."""
    g = firms.groupby(["iso3", "sector"])[outcome].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= min_obs]
    g["prem"] = g["median"] - g.iso3.map(firms.groupby("iso3")[outcome].median())
    return g.pivot(index="iso3", columns="sector", values="prem")


def sector_agreement(profile, min_shared=5):
    """For each country: Spearman correlation of its sector ranking with the average ranking of all other countries."""
    out = {}
    for c in profile.index:
        other = profile.drop(c).mean()
        m = profile.loc[c].notna() & other.notna()
        if m.sum() >= min_shared:
            out[c] = stats.spearmanr(profile.loc[c][m], other[m])[0]
    return pd.Series(out)


def sector_placebo(firms, reps=30, seed=0):
    """Mean agreement when sector labels are shuffled inside each country (no real sector effect): returns the list of means."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(reps):
        d = firms.copy()
        d["sector"] = d.groupby("iso3").sector.transform(lambda s: rng.permutation(s.values))
        res.append(sector_agreement(sector_profile(d)).mean())
    return res
