"""Head-to-head comparison of two groups of countries ("A vs B") on a firm outcome.

Countries, not firms, are the units that differ between the groups, so inference is clustered by country:
the primary p-value is a wild cluster bootstrap (utils.wild_cluster); a country-level permutation test is
reported next to it as an assumption-light cross-check. Used by notebooks 2.1 and 2.2 and meant to be called
by the dashboard later.
"""
import itertools
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import utils

MIN_COUNTRIES = 3          # per group: below this nothing is computed
WARN_COUNTRIES = 8         # per group: below this the result carries a warning


def _country_means(d, outcome, controls):
    """Country mean of the outcome after removing the controls (sector, size, year)."""
    resid = smf.ols(f"{outcome} ~ {controls}" if controls else f"{outcome} ~ 1", d).fit().resid
    return resid.groupby(d.loc[resid.index, "iso3"]).mean()


def permutation_p(means_a, means_b, n_perm=19999, seed=0):
    """Two-sided p for a difference in group means, shuffling group labels over COUNTRIES (exact if few enough)."""
    allv = np.r_[means_a.values, means_b.values]
    na, n = len(means_a), len(allv)
    obs = allv[na:].mean() - allv[:na].mean()
    if math.comb(n, na) <= 20000:        # exact: every possible split of the countries
        diffs = np.array([np.delete(allv, idx).mean() - allv[list(idx)].mean() for idx in itertools.combinations(range(n), na)])
        return float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(allv)
        diffs[i] = p[na:].mean() - p[:na].mean()
    return float((np.sum(np.abs(diffs) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))


def compare_groups(firms, group_a, group_b, outcome="roa_pp", controls=utils.CONTROLS, B=999, seed=0, perm=True):
    """Difference in `outcome` between country group B and group A (B minus A), controlling for `controls`.

    firms   : firm-year table with iso3 and the outcome (prep.load_firms)
    group_a, group_b : iterables of iso3 codes (must not overlap)
    Returns a dict: diff, ci_low, ci_high, p_wild, p_perm, countries_a/b, firm_years_a/b, raw_mean_a/b, warnings.
    """
    a, b = set(group_a), set(group_b)
    if a & b:
        raise ValueError(f"groups overlap: {sorted(a & b)}")
    present = set(firms.iso3.unique())
    a, b = a & present, b & present
    if min(len(a), len(b)) < MIN_COUNTRIES:
        raise ValueError(f"each group needs at least {MIN_COUNTRIES} countries with firm data (got {len(a)} and {len(b)})")
    d = firms[firms.iso3.isin(a | b)].dropna(subset=[outcome]).copy()
    d["grp_b"] = d.iso3.isin(b).astype(float)
    r = utils.wild_cluster("grp_b", d, outcome=outcome, controls=controls, B=B, seed=seed)
    out = dict(diff=r["coef"], ci_low=r["ci_low"], ci_high=r["ci_high"], p_wild=r["p_wild"], p_cluster=r["p_cluster"],
               countries_a=len(a), countries_b=len(b),
               firm_years_a=int((d.grp_b == 0).sum()), firm_years_b=int((d.grp_b == 1).sum()),
               raw_mean_a=float(d.loc[d.grp_b == 0, outcome].mean()), raw_mean_b=float(d.loc[d.grp_b == 1, outcome].mean()))
    if perm:
        cm = _country_means(d, outcome, controls)
        out["p_perm"] = permutation_p(cm[cm.index.isin(a)], cm[cm.index.isin(b)], seed=seed)
        out["country_means"] = cm
    w = []
    if min(len(a), len(b)) < WARN_COUNTRIES:
        w.append(f"a group has fewer than {WARN_COUNTRIES} countries: the bootstrap is unreliable, trust the permutation p")
    if max(len(a), len(b)) > 3 * min(len(a), len(b)):
        w.append("groups are very unbalanced")
    if firms[firms.iso3.isin(a | b)].groupby("iso3").size().min() < 50:
        w.append("a country with fewer than 50 firm-years is included")
    out["warnings"] = w
    return out


def min_detectable(country_sd, n_a, n_b, power_z=2.8):
    """Smallest true group difference (same unit as the outcome) detectable with ~80% power at alpha 0.05,
    from the spread of country means (normal approximation: 2.8 x SE of a difference of two means)."""
    return power_z * country_sd * math.sqrt(1 / n_a + 1 / n_b)


def median_split(values):
    """Split a country-indexed Series at its median: returns (low_group, high_group) lists of iso3."""
    v = values.dropna()
    m = v.median()
    return list(v.index[v <= m]), list(v.index[v > m])
