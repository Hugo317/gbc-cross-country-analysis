"""Dashboard (Plotly Dash): how country conditions relate to company performance.

Run:   uv run --group dash python app/dash_app.py   ->  http://127.0.0.1:8050
Reads only app/data/ (built by app/build_extract.py), so it can be hosted without data/gbc.duckdb.
"""
import json
import os
import pathlib
import re

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, Dash, Input, Output, callback, ctx, dcc, html

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
AUTHOR, GITHUB, REPO = "Hugo", "https://github.com/Hugo317", "https://github.com/Hugo317/gbc-cross-country-analysis"

# ---------------------------------------------------------------- palette / plotly template
BG, CARD, LINE, TEXT, MUTED = "#0b0b0d", "#16161a", "#2b2b31", "#ece9e6", "#9b9aa0"
BLOOD, RED, REDHI, AMBER = "#7a0c12", "#b0121b", "#e5383b", "#e9a23b"
CAT = [REDHI, "#d9d4cf", AMBER, "#8d8d96", "#ff8a8d", "#5b5b64", "#b0121b", "#bfa6a0"]
DISTINCT = [REDHI, "#4cc9f0", AMBER, "#90be6d", "#b388eb", "#e9e4dc", "#2ec4b6", "#ff7eb6"]   # categorical colours that stay apart on a dark background
CC = [REDHI, "#4cc9f0", AMBER]                                                                  # one fixed colour per selected country, shared by both country charts
RED_SCALE = [[0, "#1a1114"], [0.5, BLOOD], [1, "#ff4d50"]]
MAP_SCALE = [[0, "#5a1a20"], [0.5, "#a3121b"], [1, "#ff6b6e"]]
pio.templates["gbc"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
    colorway=CAT, xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED), automargin=True), yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=MUTED), automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=RED)))
TPL = "gbc"
CFG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["select2d", "lasso2d", "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
       "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "gbc-chart"}}

# ---------------------------------------------------------------- data
country = pd.read_parquet(DATA / "country.parquet").set_index("iso3")
F = pd.read_parquet(DATA / "firms.parquet")
X = json.loads((DATA / "extra.json").read_text())
HYP = {p.stem: json.loads(p.read_text()) for p in sorted((DATA / "results").glob("H*.json"), key=lambda p: int(p.stem[1:]))}
NOT_A_TEST = {"H11", "H15"}                      # robustness / data-quality entries, outside the multiple-testing family
Q = pd.Series(X["q"])
N_PASS = int(sum(1 for h in Q.index if Q[h] < 0.10 and HYP[h]["verdict"] == "supported"))
EXPLORATORY = {"H1", "H2", "H3", "H5", "H13"}    # suggested by exploratory scans of the same data
VERDICT_COLOR = {"supported": REDHI, "contradicted": AMBER, "not supported": "#6b6b74"}
FEATS = X["feat_cols"]
pretty = lambda s: s.replace("_", " ")
LABELS = {"gdp_per_capita_ppp": "GDP per person (PPP)", "gdp_usd": "Size of the economy (GDP, US$)", "gdp_growth": "GDP growth", "inflation": "Inflation",
          "manufacturing_share": "Manufacturing share of the economy", "services_share": "Services share of the economy", "trade_gdp": "Trade openness (trade as % of GDP)",
          "fdi_gdp": "Foreign investment (% of GDP)", "hightech_exports": "High-tech exports", "private_credit_gdp": "Bank credit to businesses and households (% of GDP)",
          "tax_revenue_gdp": "Tax revenue (% of GDP)", "gov_debt_gdp": "Government debt (% of GDP)", "current_account_gdp": "Current account (% of GDP)", "unemployment": "Unemployment",
          "youth_unemployment": "Youth unemployment", "labour_participation": "Share of people working or looking for work", "self_employed": "Self-employed share", "population": "Population",
          "pop_working_age": "Working-age share of population", "pop_65plus": "Share of population aged 65+", "urban_share": "Share living in cities", "life_expectancy": "Life expectancy",
          "gini": "Income inequality (Gini)", "tertiary_enrolment": "University enrolment", "education_spend_gdp": "Education spending (% of GDP)", "rd_gdp": "Research spending (% of GDP)",
          "patents_residents": "Patents filed by residents", "internet_users": "Internet users", "mobile_subscriptions": "Mobile phone subscriptions", "gov_effectiveness": "Government effectiveness",
          "regulatory_quality": "Regulatory quality", "rule_of_law": "Rule of law", "corruption_control": "Control of corruption", "voice_accountability": "Voice and accountability",
          "political_stability": "Political stability", "dev_index": "Development index", "inst_index": "Institutions index"}
LOGS = {"population", "gdp_usd", "patents_residents"}                   # stored as natural logs in the extract
nice = lambda f: LABELS.get(f, pretty(f).capitalize())
INDICES = {"dev_index": "Development index", "inst_index": "Institutions index"}
OUTCOMES = {"roa": "Return on assets, ROA (% points)", "margin": "Net profit margin (%)", "loss": "Loss-making company-years (%)"}
FEATURE_OPTIONS = [{"label": nice(f), "value": f} for f in FEATS] + [{"label": v, "value": k} for k, v in INDICES.items()]
OUTCOME_OPTIONS = [{"label": v, "value": k} for k, v in OUTCOMES.items()]
MAP_OPTIONS = OUTCOME_OPTIONS + [{"label": v, "value": k} for k, v in INDICES.items()] + [{"label": nice(f), "value": f} for f in FEATS]
Z = ((country[FEATS] - country[FEATS].mean()) / country[FEATS].std())
GROUPS = sorted(set(X["group"].values()))
PRESETS = [("Richer countries, lower returns", "dev_index", "roa"), ("Strong institutions vs loss-making firms", "inst_index", "loss"),
           ("Bank credit vs profit margin", "private_credit_gdp", "margin"), ("Trade openness vs ROA", "trade_gdp", "roa"),
           ("Life expectancy vs ROA", "life_expectancy", "roa")]


SECTORS = sorted(s for s in F.sector.dropna().unique() if s != "Unknown")
SECTOR_OPTIONS = [{"label": "All sectors", "value": "all"}] + [{"label": s, "value": s} for s in SECTORS]
YEARS = (int(F.fiscal_year.clip(2022, 2025).min()), int(F.fiscal_year.clip(2022, 2025).max()))
_rows = F.groupby("fiscal_year").size()
MAIN_YEARS = (int(_rows[_rows >= 500].index.min()), int(_rows[_rows >= 500].index.max()))      # years with a real number of filings
WINDOW = f"fiscal {MAIN_YEARS[0]}–{MAIN_YEARS[1]}"                                               # the one coverage window quoted everywhere
MIN_FIRMS = 10
MIN_SECTOR_FIRMS = 5                      # a sector or year slice needs this many firms in a country to be shown
MINUS = "−"


def human(v):
    for k, u in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= k:
            return f"{v / k:g}{u}"
    return f"{v:g}"


def log_ticks(lo, hi):
    ks = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    return [10.0 ** k for k in ks], [human(10.0 ** k) for k in ks]


def sg(x, d=1):
    """Signed number with a true minus sign; tiny values print as plain zero."""
    if x is None or pd.isna(x):
        return "n/a"
    if round(x, d) == 0:
        return f"{0:.{d}f}"
    return ("+" if x > 0 else MINUS) + f"{abs(x):.{d}f}"


def graph(fig=None, id=None, h=360):
    """dcc.Graph with a fixed pixel height so a chart can never grow the page."""
    kw = {"config": CFG, "style": {"height": f"{h}px"}}
    if id:
        kw["id"] = id
    if fig is not None:
        fig.update_layout(height=h)
        kw["figure"] = fig
    return dcc.Graph(**kw)


def with_labels(d, x, y, k=10):
    """Show text labels only for the k most extreme points; the rest are on hover. Keeps scatter plots readable."""
    zx, zy = (d[x] - d[x].mean()) / d[x].std(), (d[y] - d[y].mean()) / d[y].std()
    keep = (zx.abs() + zy.abs()).nlargest(k).index
    return d.assign(lab=np.where(d.index.isin(keep), d.iso2, ""))


def sample_txt(r):
    """n counts countries for the country-level tests and company-years for the firm-level ones."""
    if r["id"] == "H15":
        return f'{r["n"]:,} matched company-years (484 companies in 18 countries)'
    if r["effect_unit"].startswith("Spearman"):
        return f'n = {r["n"]} countries'
    return f'n = {r["n"]:,} company-years in {r["clusters"]} countries'


def firm_slice(sector="all", years=None):
    d = F if sector in (None, "all") else F[F.sector == sector]
    return d if years is None else d[d.fiscal_year.between(*years)]


def country_stats(sector="all", years=None):
    """Country table whose firm outcomes are recomputed for a sector and/or fiscal-year window (thin countries drop out)."""
    filtered = sector not in (None, "all") or years is not None
    a = firm_slice(sector, years).groupby("iso3").agg(firms=("company_id", "nunique"), roa=("roa_pp", "mean"), margin=("net_margin_w", "mean"), loss=("loss_pp", "mean"))
    a["margin"] = a["margin"] * 100
    a = a[a.firms >= (MIN_SECTOR_FIRMS if filtered else MIN_FIRMS)]
    return country.drop(columns=["firms", "roa", "margin", "loss"]).join(a, how="inner")


def cap(text):
    return html.Small(text, className="text-muted d-block mb-3 caption")


def ci(r):
    return "n/a" if r["ci_low"] is None or r["ci_high"] is None else f'{sg(r["ci_low"], 2)} to {sg(r["ci_high"], 2)}'


def kpi(title, value, sub=""):
    return dbc.Card(dbc.CardBody([html.Small(title, className="text-muted"), html.H3(value), html.Small(sub, className="text-muted")]), className="kpi h-100")


def stats_toggle(*lines):
    return html.Details([html.Summary("Stats detail"), *[html.Div(l) for l in lines]], className="stats")



# ---------------------------------------------------------------- plain-language content
# (id, headline, number, unit): the number and unit shown on each finding card
FINDINGS = [
    ("H0", "Firms in more developed countries earn slightly less on their assets", lambda r: f'{sg(r["effect"])} pp ROA', "points of ROA per step up in development (+1 SD)"),
    ("H13", "The effect shows up in small firms, while the largest third of firms show almost none.", lambda r: f'{sg(r["effect"], 2)} pp', "straight-line estimate of how much the effect improves from the smallest to the largest firm (grouped in thirds: −1.7, −1.2 and −0.2 pp)"),
    ("H11", "The main finding survives dropping any single country", lambda r: f'{sg(r["effect"])} pp', "weakest of 34 runs, each leaving one country out"),
    ("H5", "Countries with deeper bank credit have deeper stock markets", lambda r: f'ρ = {sg(r["effect"], 2)}', "match between credit and stock-market size, 20 countries"),
    ("H6", "In the EU, higher labour cost goes with higher productivity", lambda r: f'ρ = {sg(r["effect"], 2)}', "match between hourly labour cost and output per worker"),
    ("H1", "Surprise: stronger institutions go with MORE loss-making firms (small open economies largely account for it)", lambda r: f'{sg(r["effect"], 2)} pp', "extra chance of a loss per step up in institutions (+1 SD)"),
    ("H15", "The two data sources agree on the 484 companies in both, so the Yahoo numbers look reliable", lambda r: f'ρ = {sg(r["effect"], 2)}', "match between Yahoo and ESEF ROA, same company-year"),
]
SHORT = {"H0": "Development and ROA", "H13": "Small vs large firms", "H5": "Credit and stock markets", "H6": "EU labour cost and output", "H1": "Institutions and losses"}
PLAIN = {
    "H0": "Across 34 countries, companies in richer, more developed countries tend to earn a bit less profit for each unit of assets they own. Each step up the development "
          "scale goes with about 1.1 percentage points lower return on assets. A second data source (the official filings, ESEF) shows the same pattern, and it survives removing any single "
          "country. It is a pattern, not proof of cause: it could reflect tougher competition, higher costs, or the kinds of companies that are listed.",
    "H13": "The pattern is not the same for everyone. Small companies carry it: for them, a more developed country goes with about 1.7 points lower ROA. For medium firms it is about 1.2 points lower, and for the largest third about 0.2. (The big number above is a straight-line estimate of the improvement from the smallest to the largest firm; grouping firms in thirds, as in the charts below, gives these rounder numbers.) For the largest third of "
           "companies the effect is close to zero. We noticed this while exploring the data, so treat it as a strong lead rather than a settled fact.",
    "H11": "This is a stress test of the main finding. We re-ran the analysis 34 times, leaving out one country each time, and also tried other ways of measuring development. The result stayed negative every time. Even the weakest case, with India dropped, is still about −0.9 points. So the main finding doesn't depend on one or two unusual countries.",
    "H5": "Countries where banks lend more to businesses and households tend to have bigger stock markets. Across the 20 countries with data, the link is strong (+0.77, where +1 would be a perfect match), and it stays strong after we account for how rich each country is. Related checks on new-business activity found no clear link (+0.27 and −0.31, not significant). We found this result by exploring the same data, so we label it exploratory.",
    "H6": "Across the EU, countries where an hour of work costs more also produce more per worker, and the two move closely together (a match of +0.90). A 10% higher labour cost goes with about 8.7% higher output per worker. This fits the idea that higher productivity pays for expensive labour.",
    "H1": "We expected stronger institutions (rule of law, effective government) to go with fewer loss-making companies, but the data show the opposite: each step up in institutions comes with about 3 points more chance of a loss. That pattern is largely accounted for by small, open economies. Once we account for how open each economy is to trade and how big it is, the gap shrinks to about zero. Treat this as a warning about misleading patterns, not as a real effect of institutions.",
    "H7": "We tested whether knowing a country's conditions helps predict company ROA in a country the model has never seen. In our test it did not. Predictions got slightly worse, with the average miss growing by about 0.4 ROA points. Country conditions describe patterns across countries, but in this test they did not help forecast results for a single country.",
    "H16": "We tested 12 newer indicators (energy, climate, tax, banking, currency, broadband and more) to see whether any of them relates to company ROA beyond what development already explains. The strongest, broadband use, shows a match of −0.40. But we tried 12 indicators, and some would look good by luck, so after accounting for that, the result is not significant. We found no reliable extra link.",
    "H14": "The idea was that the gap between more and less developed countries narrows over time. The estimate leans the other way (the gap widens by about 0.05 points a year), but the range includes zero, so there is no clear trend either way.",
    "H15": "This is a data check, not a finding about countries. We compared Yahoo Finance profit figures with the officially filed accounts (ESEF) for the same companies and years, and they match almost perfectly, so the Yahoo numbers look reliable for these companies. The catch is that for the 484 companies in both sources, the second source is not an independent confirmation.",
}
VERDICT_TEXT = {"supported": "Holds up", "contradicted": "Surprise: explained away", "not supported": "No clear evidence"}
CAVEAT_H = {"H5"}                                                   # supported, but a related check did not confirm


def status_badge(h):
    """One status vocabulary for cards, pages and the Method table: (text, css class)."""
    v = HYP[h]["verdict"]
    if v == "contradicted":
        return "Surprise: explained away", "bg-amber"
    if v != "supported":
        r = HYP[h]
        lo, hi, ex = r.get("ci_low"), r.get("ci_high"), r.get("expected_sign", 0)
        if lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi) and ex and (hi < 0 or lo > 0) and ((hi < 0) == (ex > 0)):
            return "Not supported: points the wrong way", "bg-ash"
        return "No clear evidence", "bg-ash"
    if h in NOT_A_TEST:
        return "Check passed", "bg-red"
    return ("Holds up, with a caveat" if h in CAVEAT_H else "Holds up"), "bg-red"


def plain_unit(u):
    if u.startswith("pp "):
        u = "percentage points of " + u[3:]
    return (u.replace("+1 pp", "+1 percentage point").replace("pp", "points").replace(" SD", " standard deviation").replace("MAE gain", "gain in prediction accuracy")
             .replace("extra adjusted R²", "extra share of variation explained").replace("Spearman ρ", "match (ρ)"))


def fm(x, d=1):
    return f"{x:.{d}f}".replace("-", MINUS)


def fix_minus(t):
    return re.sub(r"(?<![\w.])-(?=\d)", MINUS, t)


# headline number and unit text for ideas whose raw unit is not "points"
BIGOVER = {"H3": (lambda r: f'{sg(r["effect"] * 100, 0)}%', "share of the development–ROA link explained by sector mix, size and leverage"),
           "H4": (lambda r: sg(r["effect"], 2), "extra share of the variation in ROA explained by sector × country type"),
           "H7": (lambda r: f'{sg(r["effect"], 2)} pp', "change in prediction accuracy for unseen countries (negative = predictions got worse)"),
           "H16": (lambda r: f'ρ = {sg(r["effect"], 2)}', "match between broadband use and typical company ROA (strongest of 12 indicators tested)")}
GLOSSARY = [("ROA (return on assets)", "Profit divided by everything the company owns. 5% means 5 units of profit for every 100 units of assets."),
            ("pp (percentage points)", "A gap between two percentages. Going from 5% to 4% ROA is 1 pp lower."),
            ("SD (standard deviation)", "A yardstick for 'one typical step'. A country 1 SD above average is about as far above average as the typical country differs from it."),
            ("ρ (rho)", "How closely two things move together, from −1 to +1. 0 means no link, +1 means they always rise together."),
            ("p and q", "How easily chance alone could explain a result. Small is better. q is p after correcting for having tested many ideas."),
            ("Wild cluster bootstrap", "A re-sampling method that suits having only 34 countries."),
            ("Exploratory", "Suggested by looking at the same data, so treat it as a lead rather than proof."),
            ("Spearman correlation", "A rank-based version of ρ: do countries that rank high on one thing also rank high on the other?"),
            ("R² (share explained)", "How much of the ups and downs in ROA a model accounts for. 0 = nothing, 1 = everything."),
            ("Prediction error (MAE)", "The average miss, in ROA points, when a model predicts a company's ROA. Lower is better."),
            ("Leave-one-country-out", "Re-running a result again and again, each time dropping one country, to see whether any single country drives it."),
            ("Interaction", "Whether an effect differs between groups, for example between small and large firms."),
            ("Permutation test", "Shuffling labels at random many times to see how often chance alone produces a result as big."),
            ("Benjamini–Hochberg", "The correction that keeps us honest when many ideas are tested at once; q is the corrected p."),
            ("Evidence strength", "How unlikely it is that chance alone produced the result: the longer the bar, the stronger."),
            ("Confirmed on 2nd data source", "The same test on the official EU filings (ESEF) gave an estimate pointing the same way."),
            ("Yahoo / ESEF", "The two sources of company accounts: Yahoo Finance (all 34 countries) and official EU filings (ESEF).")]


def glossary():
    return dbc.Accordion([dbc.AccordionItem(html.Dl([x for t, d in GLOSSARY for x in (html.Dt(t), html.Dd(d))], className="glossary"), title="Words used on this page")],
                         start_collapsed=True, className="mt-3")


HOLDS = [f[0] for f in FINDINGS if f[0] in HYP]
CONFIRMED = [h for h in HOLDS if h not in EXPLORATORY and h != "H0"]
EXPLO = [h for h in HOLDS if h in EXPLORATORY]
FDICT = {f[0]: f for f in FINDINGS}
REST = [h for h in HYP if h not in HOLDS]
HNUM = lambda h: int(h[1:])


def hlink(h, text=None, cls="hlink"):
    return dcc.Link(text or h, href=f"/h/{h}", className=cls)


# ---------------------------------------------------------------- overview
def finding_card(h, headline, big, unit, hero=False):
    r = HYP[h]
    warn = r["verdict"] == "contradicted"
    bt, bc = status_badge(h)
    tags = [dbc.Badge(bt, className="me-1 " + bc)] + ([dbc.Badge("data check, not a new claim", color="secondary", className="me-1")] if h in NOT_A_TEST else [])
    if h in EXPLORATORY:
        tags.append(dbc.Badge("exploratory", color="warning", text_color="dark", className="me-1"))
    rep = r.get("replication_coef")
    if rep is not None and not pd.isna(rep) and r.get("replication_same_sign") and not r["effect_unit"].startswith("Spearman") and r["verdict"] != "not supported":
        tags.append(dbc.Badge("confirmed on 2nd data source", className="me-1 bg-ash"))
    qtxt = "" if h in NOT_A_TEST else f'q {"< 0.001" if Q[h] < 0.001 else f"= {Q[h]:.3f}"} (corrected for testing 15 ideas). '
    stats = stats_toggle(f'{qtxt}p = {r["p_primary"]:.2g} ({r["p_method"]}); 95% range {ci(r)}; {sample_txt(r)}.')
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Small([hlink(h), f' · {r["title"]}'], className="text-muted"), html.H5(headline, className="mt-1"),
        html.H2(big, className="big warn" if warn else "big"), html.Small(unit, className="text-muted d-block mb-2"), html.Div(tags), stats,
        html.Div(hlink(h, "Read the full finding →", "more-link"), className="mt-2")]),
        className="finding h-100 shadow-sm" + (" warn" if warn else "")), md=12 if hero else 6, lg=12 if hero else 4, className="mb-3")


def evidence_fig():
    ids = [h for h in HOLDS if h not in NOT_A_TEST]
    d = pd.DataFrame({"h": ids, "label": [f'{h} · {SHORT.get(h, HYP[h]["title"][:24])}' for h in ids], "strength": [-np.log10(max(HYP[h]["p_primary"], 1e-9)) for h in ids],
                      "verdict": [HYP[h]["verdict"] for h in ids]})
    d = d.sort_values("strength", ascending=False)
    fig = px.bar(d, x="strength", y="label", orientation="h", color="verdict", color_discrete_map=VERDICT_COLOR, template=TPL,
                 labels={"strength": "evidence strength", "label": ""})
    fig.add_vline(x=-np.log10(0.05), line_dash="dot", line_color=MUTED)
    fig.add_annotation(x=-np.log10(0.05), y=1.0, yref="paper", text="conventional threshold", showarrow=False, yshift=12, font=dict(color=MUTED, size=11), xanchor="left")
    fig.update_layout(margin=dict(l=10, r=10, t=34, b=60), showlegend=False, yaxis_autorange="reversed")
    fig.update_xaxes(title_standoff=14)
    return fig


def rest_table():
    head = html.Thead(html.Tr([html.Th(x) for x in ["", "Question", "Result", "Effect", "p"]]))
    unit = plain_unit
    def eff(h):
        r = HYP[h]
        if h in BIGOVER:
            return BIGOVER[h][0](r)
        return f'ρ = {sg(r["effect"], 2)}' if r["effect_unit"].startswith("Spearman") else f'{sg(r["effect"], 2)} pp'
    rows = [html.Tr([html.Td(hlink(h)), html.Td(HYP[h]["title"]), html.Td(status_badge(h)[0]), html.Td(eff(h), title=BIGOVER[h][1] if h in BIGOVER else unit(HYP[h]["effect_unit"])),
                     html.Td("≥ 0.99" if HYP[h]["p_primary"] >= 0.99 else f'{HYP[h]["p_primary"]:.2f}' if HYP[h]["p_primary"] >= 0.01 else f'{HYP[h]["p_primary"]:.1g}')]) for h in REST]
    return html.Div([dbc.Table([head, html.Tbody(rows)], size="sm", responsive=True, className="small text-muted"),
                     html.Small("p: how easily chance alone could produce the result (small = unlikely to be chance). A large p such as ≥ 0.99 means chance explains it easily; for H7 it tests whether predictions improved, and they did not.", className="text-muted")])


def hero_card():
    h, hl, fn, unit = FDICT["H0"]
    r = HYP[h]
    d = with_labels(country, "dev_index", "roa", 9)
    fig = px.scatter(d.reset_index(), x="dev_index", y="roa", text="lab", hover_name="name", template=TPL, color_discrete_sequence=[REDHI],
                     labels={"dev_index": "development index (SD)", "roa": "average ROA (%)"})
    b = np.polyfit(d.dev_index, d.roa, 1)
    xs = np.linspace(d.dev_index.min(), d.dev_index.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False, hoverinfo="skip"))
    fig.update_traces(textposition="top center", textfont=dict(size=10, color=TEXT), marker=dict(size=9, opacity=.9), selector=dict(mode="markers+text"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(title_standoff=14)
    fig.update_yaxes(title_standoff=14)
    return dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([html.Small([hlink(h), f' · {r["title"]}'], className="text-muted"), html.H3(hl, className="mt-1"), html.H1(fn(r), className="big"),
                 html.Small(unit, className="text-muted d-block mb-2"), dbc.Badge(status_badge("H0")[0], className="me-1 bg-red"), dbc.Badge("confirmed on 2nd data source", className="bg-ash"),
                 stats_toggle(f'p = {r["p_primary"]:.2g} ({r["p_method"]}); 95% range {ci(r)}; {sample_txt(r)}.'),
                 html.Div(hlink(h, "Read the full finding →", "more-link"), className="mt-2")], md=5),
        dbc.Col([graph(fig, h=320), cap("Each dot is a country (the most extreme ones are labelled; hover for the rest). Further right = more developed; higher up = firms earn more on their assets. The dashed line slopes down.")], md=7)],
        align="center")), className="finding hero-card shadow-sm")


layout_overview = html.Div([
    dbc.Row([dbc.Col(xs=6, md=True, children=kpi("Countries", f"{len(country)}", "21 EU + 13 non-EU")),
             dbc.Col(xs=6, md=True, children=kpi("Companies", f'{X["n_companies"]:,}', "listed, Yahoo Finance")),
             dbc.Col(xs=6, md=True, children=kpi("Company-years", f'{X["n_firm_years"]:,}', f"mostly {WINDOW}")),
             dbc.Col(xs=6, md=True, children=kpi("Country indicators", f"{len(FEATS)}", "World Bank, WGI, IMF, Eurostat")),
             dbc.Col(xs=6, md=True, children=kpi("Findings that hold", f"{N_PASS}", f"of {len(Q)} tests; 1 more passed but is explained away (H1)"))], className="g-3 mb-4"),
    dbc.Alert([html.B("Question: "), "how do the social and economic conditions of a country relate to the performance of the companies that operate there? ",
               "Click an H-number on any finding to read the full story."], className="alert-quest"),
    glossary(),
    html.H4("The headline result", className="mt-4"),
    hero_card(),
    html.H4("Also holds up", className="mt-4"), html.P("Checks that back the headline up, and findings that were planned in advance.", className="text-muted small"),
    dbc.Row([finding_card(*FDICT[h][:2], FDICT[h][2](HYP[h]), FDICT[h][3]) for h in CONFIRMED], className="g-3"),
    html.H4("Exploratory findings", className="mt-4"),
    html.P("These came from looking at the same data, so treat them as leads rather than proof.", className="text-muted small"),
    dbc.Row([finding_card(*FDICT[h][:2], FDICT[h][2](HYP[h]), FDICT[h][3]) for h in EXPLO], className="g-3"),
    html.H6("How strong is the evidence?", className="mt-3"), graph(evidence_fig(), h=60 + 46 * len([h for h in HOLDS if h not in NOT_A_TEST]) + 60),
    cap("Each bar is one finding; the longer the bar, the harder the result is to explain by chance. Evidence strength is how unlikely chance alone is to produce the result (higher = stronger). Bars past the dotted line pass the conventional threshold. The orange bar (H1) passed the test but is explained away, so it does not count as holding."),
    dbc.Accordion([dbc.AccordionItem([html.P(f"{len(REST)} further questions were tested; the data show no clear effect. They are listed for completeness: "
                                             "a clear answer of \"no\" is still an answer.", className="small text-muted"), rest_table()],
                                     title=f"Also tested: no clear evidence ({len(REST)})")], start_collapsed=True, className="mt-3"),
    html.Small("Results are associations, not causation. Only 34 countries are independent observations, so inference uses a method built for few groups (wild cluster bootstrap).",
               className="text-muted d-block mt-3"),
])

# ---------------------------------------------------------------- hypothesis pages (one per Hx, opened from the Overview)
_se = pd.DataFrame(X["size_effects"])
_size_fig = px.bar(_se, x="size", y="coef", error_y=_se.hi - _se.coef, error_y_minus=_se.coef - _se.lo, template=TPL, color_discrete_sequence=[REDHI],
                   labels={"coef": "ROA change (points)", "size": "company size within its country"})
_size_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
_size_fig.update_xaxes(title_standoff=14)
_g = X["grid"]
_heat = px.imshow(pd.DataFrame(_g["data"], index=_g["index"], columns=_g["columns"]), text_auto=".1f", color_continuous_scale=RED_SCALE, aspect="auto", template=TPL,
                  labels={"x": "country development", "y": "company size"})
_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)


def range_fig(r):
    lo, hi, e = r["ci_low"], r["ci_high"], r["effect"]
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return None
    width = max(hi - lo, 1e-9)
    show_zero = min(abs(lo), abs(hi)) <= 3 * width or lo <= 0 <= hi          # zoom on the range when zero is far away (e.g. a correlation of +0.99)
    col = AMBER if r["verdict"] == "contradicted" else REDHI
    fig = go.Figure(go.Scatter(x=[e], y=[""], mode="markers", marker=dict(size=14, color=col), error_x=dict(type="data", symmetric=False, array=[hi - e], arrayminus=[e - lo], color=col, thickness=4, width=10),
                               hovertemplate=f"estimate {sg(e, 2)}<br>range {sg(lo, 2)} to {sg(hi, 2)}<extra></extra>"), layout=dict(template=TPL))
    lo_a, hi_a = (min(lo, 0), max(hi, 0)) if show_zero else (lo, hi)
    pad = (hi_a - lo_a) * 0.15 + 1e-9
    if show_zero:
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        fig.add_annotation(x=0, y=1.0, yref="paper", text="no effect", showarrow=False, yshift=12, font=dict(color=MUTED, size=11), xanchor="left")
    fig.update_xaxes(range=[lo_a - pad, hi_a + pad], title="estimate (dot) and 95% range (line)", title_standoff=12, **({"tickformat": ".0%"} if r["id"] == "H3" else {}))
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=50), showlegend=False)
    return fig


def nf(h, x, d=2):
    """Number formatter: H3 is a share, shown as a percentage."""
    return f"{sg(x * 100, 0)}%" if h == "H3" else sg(x, d)


def ci_h(h, r):
    return f'{nf(h, r["ci_low"])} to {nf(h, r["ci_high"])}' if h == "H3" else ci(r)


def fmtp(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3g}"


def sign(x):
    return (x > 0) - (x < 0)


def plain_text(h, r):
    if h in PLAIN:
        return PLAIN[h]
    e, lo, hi, ex = r["effect"], r["ci_low"], r["ci_high"], r.get("expected_sign", 0)
    t = "The data do not back this idea up. "
    if lo is not None and hi is not None and not pd.isna(lo) and lo <= 0 <= hi:
        t += f"The best estimate is {nf(h, e)}, but the plausible range ({nf(h, lo)} to {nf(h, hi)}) includes zero, so we cannot tell it apart from no effect. " + ("It leans the opposite way to what the idea predicts. " if ex and sign(e) != ex and abs(e) > 0.1 * (hi - lo) else "")
    elif ex and sign(e) != ex:
        t += f"The estimate ({sg(e, 2)}) even points the opposite way to what we expected. "
    else:
        t += f"The best estimate is {nf(h, e)}, which is too weak or too uncertain to count as a finding. "
    return t + "Read this as 'no clear evidence', not as proof that there is no link."


def hyp_page(h):
    r = HYP[h]
    ok = r["verdict"] == "supported"
    headline = FDICT[h][1] if h in FDICT else f'Idea tested: {r["statement"]}'
    big = FDICT[h][2](r) if h in FDICT else BIGOVER[h][0](r) if h in BIGOVER else (f'ρ = {sg(r["effect"], 2)}' if r["effect_unit"].startswith("Spearman") else f'{sg(r["effect"], 2)} pp')
    unit = FDICT[h][3] if h in FDICT else BIGOVER[h][1] if h in BIGOVER else plain_unit(r["effect_unit"])
    bt, bc = status_badge(h)
    tags = [dbc.Badge(bt, className="me-2 " + bc)]
    if h in EXPLORATORY:
        tags.append(dbc.Badge("exploratory", color="warning", text_color="dark", className="me-2"))
    if h in NOT_A_TEST:
        tags.append(dbc.Badge("data check, not a new claim", color="secondary", className="me-2"))
    rep = r.get("replication_coef")
    if rep is not None and not pd.isna(rep) and r.get("replication_same_sign") and not r["effect_unit"].startswith("Spearman") and r["verdict"] != "not supported":
        tags.append(dbc.Badge("confirmed on 2nd data source", className="me-2 bg-ash"))
    own_unit = not r["effect_unit"].startswith("Spearman")
    rep_txt = None
    later = [n for n in r["notes"] if n.startswith("Replication on 2023")]
    if own_unit and rep is not None and not pd.isna(rep) and later:
        rep_txt = f"A check on the later years only (2023–2024, the same firms, so not independent) gives {sg(rep, 2)}."
    elif own_unit and rep is not None and not pd.isna(rep):
        rep_txt = f'The second data source (official ESEF filings) gives {nf(h, rep)}' + \
                  ((", pointing the same way." if sign(rep) == sign(r["effect"]) else ", pointing the other way.") if r["verdict"] != "not supported"
                   else (", with the same sign." if sign(rep) == sign(r["effect"]) else ", with the opposite sign.")) + \
                  (" Both sources show the same pattern, which is the opposite of what we expected." if r["verdict"] == "contradicted" and sign(rep) == sign(r["effect"]) else "")
    elif own_unit and h not in NOT_A_TEST:
        rep_txt = "This idea could not be re-tested on a second data source."
    fig = range_fig(r)
    robust = r.get("robust_share")
    bullets = [html.Li(f'Our best estimate is {nf(h, r["effect"])}. A reasonable range for the true value is {ci_h(h, r)}.' if fig is not None else (f'This measure has no range; the adjusted p-value (q) is {r["p_primary"]:.2f}.' if h == "H16" else f'This measure has no range; the p-value is {r["p_primary"]:.2f}.'))]
    if ok and robust is not None and not pd.isna(robust):
        bullets.append(html.Li(f"It held in {robust:.0%} of the alternative ways we cut the data."))
    if h == "H1":
        bullets.append(html.Li("Once we account for trade openness and population size, the effect shrinks to about zero (+0.12, p = 0.93). So we treat it as a surprise, not a finding."))
    if rep_txt:
        bullets.append(html.Li(rep_txt))
    bullets.append(html.Li(f'Basis: {sample_txt(r)}.'))
    extra = []
    if h == "H0":
        d = with_labels(country, "dev_index", "roa", 12)
        f = px.scatter(d.reset_index(), x="dev_index", y="roa", text="lab", hover_name="name", template=TPL, color_discrete_sequence=[REDHI],
                       labels={"dev_index": "development index (SD)", "roa": "average ROA (%)"})
        b = np.polyfit(d.dev_index, d.roa, 1)
        xs = np.linspace(d.dev_index.min(), d.dev_index.max(), 20)
        f.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False, hoverinfo="skip"))
        f.update_traces(textposition="top center", textfont=dict(size=10, color=TEXT), selector=dict(mode="markers+text"))
        f.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        f.update_xaxes(title_standoff=14)
        f.update_yaxes(title_standoff=14)
        extra = [html.H5("The pattern, country by country", className="mt-4"), graph(f, h=380), cap("Each dot is a country. Further right = more developed; higher up = firms earn more on their assets.")]
    elif h == "H13":
        extra = [html.H5("Who carries the effect?", className="mt-4"),
                 dbc.Row([dbc.Col([html.H6("Change in ROA by company size"), graph(_size_fig, h=340),
                                   cap("Per step up in development, ROA changes by this many points for small, medium and large firms. Whiskers show the 95% range. The effect shrinks as firms get bigger.")], md=6),
                          dbc.Col([html.H6("Typical ROA (%): company size × country development"), graph(_heat, h=340),
                                   cap("Rows are company size, columns are how developed the country is. Brighter = higher ROA.")], md=6)])]
    elif h == "H1":
        d = with_labels(country, "inst_index", "loss", 10)
        f = px.scatter(d.reset_index(), x="inst_index", y="loss", text="lab", hover_name="name", template=TPL, color_discrete_sequence=[AMBER],
                       labels={"inst_index": "institutions index (SD)", "loss": "share of loss-making company-years (%)"})
        f.update_traces(textposition="top center", textfont=dict(size=10, color=TEXT), selector=dict(mode="markers+text"))
        f.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        f.update_xaxes(title_standoff=14)
        f.update_yaxes(title_standoff=14)
        extra = [html.H5("The surprising pattern", className="mt-4"), graph(f, h=380), cap("Each dot is a country. Further right = stronger institutions; higher up = more loss-making companies.")]
    caveats = [] if h == "H15" else ["This shows an association, not a cause."]
    if h in CAVEAT_H:
        caveats.append("The caveat: related checks on new-business activity did not confirm a link (institutions vs new-business density +0.27; business birth rate in the EU −0.31; neither is significant).")
    if h in EXPLORATORY:
        caveats.append("It was suggested by looking at the same data, so it is a lead rather than proof.")
    if h in NOT_A_TEST:
        caveats.append("It is a stress test or data check of an earlier finding, so it is not counted among the 15 tests that are corrected for multiple testing.")
    if r["effect_unit"].startswith("Spearman") and h != "H15":
        caveats.append(f'It is based on {r["n"]} countries, so it is a country-level pattern, not a company-level one.')
    if h == "H15":
        caveats.append("Because the two sources agree this closely, results confirmed on ESEF are not independent for the 484 companies that appear in both.")
    return html.Div([
        dcc.Link("← Back to all findings", href="/", className="more-link"),
        html.Div([html.Small(f'{h} · {r["title"]}', className="text-muted d-block mt-3"), html.H2(headline, className="mt-1"), html.Div(tags, className="mb-3")]),
        dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([html.Small("The number", className="text-muted"),
                                                html.H1(big, className="big" + (" warn" if r["verdict"] == "contradicted" else "" if ok else " mute")), html.Small(unit, className="text-muted")]),
                                  className="finding h-100" + (" warn" if r["verdict"] == "contradicted" else "")), md=4),
                 dbc.Col(dbc.Card(dbc.CardBody([html.H5("In plain words"), html.P(plain_text(h, r), className="mb-0")]), className="h-100"), md=8)], className="g-3 mb-4"),
        html.H5("How sure are we?"),
        html.Ul(bullets),
        graph(fig, h=170) if fig is not None else html.Div(),
        cap("If the line crosses the dotted 'no effect' mark, the true effect could be zero.") if fig is not None and fig.layout.annotations else html.Div(),
        *extra,
        html.H5("Keep in mind", className="mt-4"), html.Ul([html.Li(c) for c in caveats]),
        dbc.Accordion([dbc.AccordionItem([html.P(f'Estimate {fm(r["effect"], 3)} {r["effect_unit"]}; 95% CI {ci(r)}; {"adjusted p (q)" if h == "H16" else "p"} {fmtp(r["p_primary"])} ({r["p_method"]}); {sample_txt(r)}.' +
                                                 ("" if h in NOT_A_TEST or h not in Q.index else f' q = {Q[h]:.3f} (Benjamini–Hochberg across 15 tests).'), className="small"),
                                          html.Ul([html.Li(fix_minus(n)) for n in r["notes"]], className="small"),
                                          html.P('In these notes, "same direction" and "opposite direction" refer to the sign we expected in advance, not to the main estimate.', className="small text-muted") if any("direction" in n for n in r["notes"]) else html.Div(),
                                          html.P("The by-size numbers in these notes come from a slightly different model than the bar chart on this page (about −2.0 vs −1.7 for small firms); the pattern is the same.", className="small text-muted") if h == "H13" else html.Div()], title="Technical details")], start_collapsed=True, className="mt-3"),
        glossary(),
        html.Small(f'Full analysis in notebook {r["notebook"]}', className="text-muted d-block mt-3"),
    ])


# ---------------------------------------------------------------- explore: scatter + map + size
layout_explore = html.Div([
    html.H5("Explore: pick any country condition and any firm outcome"),
    html.P(f"Each dot is a country (average over all available years, mainly {WINDOW}). Start from a preset or choose your own.", className="text-muted"),
    html.Div([dbc.Button(lbl, id={"type": "preset", "index": i}, className="btn-preset me-2 mb-2", n_clicks=0) for i, (lbl, _, _) in enumerate(PRESETS)]),
    dbc.Row([dbc.Col([dbc.Label("Country condition (x)"), dcc.Dropdown(FEATURE_OPTIONS, "dev_index", id="x", clearable=False)], md=4),
             dbc.Col([dbc.Label("Firm outcome (y)"), dcc.Dropdown(OUTCOME_OPTIONS, "roa", id="y", clearable=False)], md=4),
             dbc.Col([dbc.Label("Colour by"), dcc.Dropdown([{"label": "Region", "value": "region"}, {"label": "Income group", "value": "income_group"},
                                                           {"label": "EU or not", "value": "eu_group"}], "region", id="colour", clearable=False)], md=4)], className="mb-3"),
    dbc.Row([dbc.Col([dbc.Label("Sector (recomputes firm outcomes)"), dcc.Dropdown(SECTOR_OPTIONS, "all", id="sector", clearable=False)], md=4),
             dbc.Col([dbc.Button("Download country table (CSV)", id="dl-btn", className="btn-preset mt-4"), dcc.Download(id="dl")], md=4)], className="mb-3"),
    graph(id="scatter", h=540), html.Div(id="scatter-note", className="text-muted"),
    cap("Each bubble is a country; bigger bubbles have more firms. Labels show the most extreme countries; hover for the rest. A slope means the condition on the x-axis goes with the firm outcome on the y-axis. It does not prove cause."),
    html.Hr(),
    html.H5("Map"),
    dbc.Row([dbc.Col([dbc.Label("Show"), dcc.Dropdown(MAP_OPTIONS, "roa", id="map-metric", clearable=False)], md=6),
             dbc.Col([dbc.Label("Area"), dbc.RadioItems(options=[{"label": "Europe", "value": "europe"}, {"label": "World", "value": "world"}], value="europe", id="map-scope", inline=True)], md=6)],
            className="mb-2"),
    graph(id="map", h=520), cap("Darker = lower, brighter red = higher. Grey = not in this study, or (when a sector is chosen) fewer than 5 firms in that sector. The map follows the Sector filter above."),
    html.Hr(),
    html.H5("Who carries the effect? Small firms (all sectors)"),
    html.P(["Small companies carry the main pattern; the largest third of firms show almost none. ", hlink("H13", "Read the full finding →", "more-link")]),
    dbc.Row([dbc.Col([html.H6("Change in ROA by company size"), graph(_size_fig, h=340),
                      cap("Per step up in development, ROA changes by this many points for small, medium and large firms. Whiskers show the 95% range. The effect shrinks as firms get bigger.")], md=6),
             dbc.Col([html.H6("Typical ROA (%): company size × country development"), graph(_heat, h=340),
                      cap("Rows are company size, columns are how developed the country is. Brighter = higher ROA.")], md=6)]),
    html.Small("Exploratory: this refinement grew out of looking at the main finding.", className="text-muted"),
])

# ---------------------------------------------------------------- countries (single view or side-by-side)
layout_countries = html.Div([
    html.H5("Countries: look at one, or compare up to three"),
    dbc.Row([dbc.Col([dbc.Label("Countries (max 3)"), dcc.Dropdown(sorted(country.name), ["Germany", "Poland"], id="countries", multi=True)], md=6),
             dbc.Col([dbc.Label("Indicator theme"), dcc.Dropdown([{"label": "All themes", "value": "all"}] + [{"label": g.capitalize(), "value": g} for g in GROUPS], "governance", id="theme", clearable=False)], md=3),
             dbc.Col([dbc.Label("Sector"), dcc.Dropdown(SECTOR_OPTIONS, "all", id="c-sector", clearable=False)], md=3)], className="mb-3"),
    dbc.Label("Fiscal years"), dcc.RangeSlider(YEARS[0], YEARS[1], 1, value=list(YEARS), marks={y: str(y) for y in range(YEARS[0], YEARS[1] + 1)}, id="years", className="mb-4"),
    html.Div(id="country-cards"), dcc.Graph(id="country-bars", config=CFG),
    cap("Bars show how far each indicator sits from the average of the 34 countries, in standard deviations (SD). Right = above average, left = below."),
    html.H6("Median firm return (ROA) by year"), graph(id="country-roa", h=320),
    cap("Median return on assets of listed firms in each country and fiscal year. A dot is missing when too few firms filed that year."),
])

# ---------------------------------------------------------------- company level: companies compared inside the same country
CO = X["company"]
WS = pd.DataFrame(CO["slopes"])
W_UNIT = {"W1": 0.1, "W2": 1.0}                                     # equity ratio is shown per +0.1
W_LABEL = {"W1": ("equity ratio", "change in ROA (points) per +0.1 equity ratio"), "W2": ("company size", "change in ROA (points), largest vs smallest company")}
W_TEXT = {"W1": ("1 · Companies with more equity financing tend to earn more, in most countries",
                  "Equity ratio = equity / total assets. Inside each country, companies funded more by equity earn higher return on assets, after allowing for sector and year."),
          "W2": ("1 · Bigger companies do not consistently earn more or less: it depends on the country",
                 "Company size = a company's rank by total assets within its own country (0 = smallest, 1 = largest). Inside each country we ask whether bigger companies earn more ROA, after allowing for sector and year.")}
W1S, W2S = CO["summary"]["W1"], CO["summary"]["W2"]
AB1 = json.loads((DATA / "results" / "AB1.json").read_text())
EUC = pd.DataFrame(CO["eu_countries"])
SEC = CO["sector"]
EUC_CTRL = CO["eu_controlled"]
N_CO = len(set(WS.iso3))


def forest_fig(h):
    u, s = W_UNIT[h], CO["summary"][h]
    d = WS[WS.trait == h].assign(c=lambda d: d.coef * u, lo=lambda d: (d.coef - 1.96 * d.se) * u, hi=lambda d: (d.coef + 1.96 * d.se) * u).sort_values("c")
    fig = px.scatter(d, x="c", y="country", template=TPL, color_discrete_sequence=[REDHI], error_x=d.hi - d.c, error_x_minus=d.c - d.lo,
                     labels={"c": W_LABEL[h][1], "country": ""}, custom_data=["country", "n", "firms"])
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Estimate: %{x:.2f}<br>Companies: %{customdata[2]}<br>Company-years: %{customdata[1]}<extra></extra>")
    fig.add_vline(x=0, line_color=MUTED)
    fig.add_vline(x=s["mean"] * u, line_dash="dot", line_color=AMBER)
    fig.add_annotation(x=s["mean"] * u, y=1.0, yref="paper", text=f'pooled {sg(s["mean"] * u, 2)}', showarrow=False, yshift=12, font=dict(color=AMBER, size=11), xanchor="left")
    fig.update_layout(margin=dict(l=0, r=10, t=36, b=60))
    fig.update_xaxes(title_standoff=14)
    fig.update_yaxes(categoryorder="array", categoryarray=list(d.country), tickfont=dict(size=11))
    return fig


def sector_fig():
    d = pd.DataFrame({"sector": list(SEC["mean"]), "premium": list(SEC["mean"].values()), "positive": [SEC["share_positive"][k] for k in SEC["mean"]],
                      "countries": [SEC["countries"][k] for k in SEC["mean"]]}).sort_values("premium")
    fig = px.bar(d, x="premium", y="sector", orientation="h", template=TPL, color_discrete_sequence=[REDHI], custom_data=["positive", "countries"],
                 labels={"premium": "ROA vs country median (points)", "sector": ""})
    fig.update_traces(hovertemplate="<b>%{y}</b><br>ROA vs country median: %{x:.2f} points<br>Above the median in %{customdata[0]:.0%} of %{customdata[1]} countries<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=60))
    fig.update_xaxes(title_standoff=14)
    return fig


def eu_fig():
    d = EUC.assign(group=np.where(EUC.eu, "EU", "non-EU"))
    fig = px.strip(d, x="adj_roa", y="group", color="group", hover_name="name", hover_data={"adj_roa": ":.2f", "group": False}, template=TPL, color_discrete_map={"EU": REDHI, "non-EU": "#4cc9f0"},
                   labels={"adj_roa": "adjusted ROA (points, relative)", "group": ""})
    for g, v in d.groupby("group").adj_roa.mean().items():
        fig.add_shape(type="line", x0=v, x1=v, y0=-0.4 if g == "EU" else 0.6, y1=0.4 if g == "EU" else 1.4, line=dict(color=AMBER, dash="dot"))
    fig.update_traces(marker_size=11)
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=60), showlegend=False)
    fig.update_xaxes(title_standoff=14)
    return fig


layout_company = html.Div([
    html.H5("Inside a country: which companies earn more, and does it repeat?"),
    dbc.Alert([html.B("How this differs from the other tabs. "), "Everywhere else we compare countries, and there are only 34. Here we compare companies inside the same country, "
               "where the economy, tax system and institutions are shared, then run the same comparison in every country and ask whether the pattern repeats. "
               "Each country is a replication."], className="alert-quest"),
    dbc.Row([dbc.Col(xs=6, md=3, children=kpi("Equity share and ROA", f'{sg(W1S["mean"] * 0.1)} pp', "per +0.1 equity ratio")),
             dbc.Col(xs=6, md=3, children=kpi("Repeats in", f'{round(W1S["same_sign_share"] * W1S["k"])} of {W1S["k"]}', "countries, same direction")),
             dbc.Col(xs=6, md=3, children=kpi("Sector ranking", f'{sg(SEC["agreement"], 2)}', f'country vs the rest (if sectors are shuffled: {sg(SEC["placebo"], 2)})')),
             dbc.Col(xs=6, md=3, children=kpi("Company size", "no pattern", f'same direction in {W2S["same_sign_share"]:.0%} of countries'))], className="g-3 mb-4"),
    html.H5(id="w-title"), html.P(id="w-desc", className="text-muted"),
    dbc.RadioItems(id="w-trait", options=[{"label": "Equity ratio (repeats)", "value": "W1"}, {"label": "Company size (does not repeat)", "value": "W2"}], value="W1", inline=True, className="mb-2"),
    graph(id="w-forest", h=700), html.Div(id="w-note", className="text-muted"),
    cap(f"Each dot is one country's own estimate, with its 95% range; the dotted line is the pooled value. Countries with fewer than 30 listed companies are left out ({len(country) - N_CO} of 34). "
        "When the dots all fall on the same side of zero, the pattern repeats; when they scatter on both sides, it does not."),
    stats_toggle(f'Equity ratio: pooled {sg(W1S["mean"] * 0.1, 2)} pp (95% range {sg(W1S["ci_low"] * 0.1, 2)} to {sg(W1S["ci_high"] * 0.1, 2)}), p = {W1S["p"]:.2g}; '
                 f'{W1S["significant_same_sign"]} countries individually significant in the same direction, {W1S["significant_opposite"]} in the opposite one; still {W1S["loo_agreement"]:.0%} same-direction when any one country is dropped.',
                 f'Size: pooled {sg(W2S["mean"], 2)} pp (95% range {sg(W2S["ci_low"], 2)} to {sg(W2S["ci_high"], 2)}), p = {W2S["p"]:.2g}; {W2S["significant_same_sign"]} countries significant in the pooled direction, {W2S["significant_opposite"]} in the opposite one.'),
    html.Small(id="w-caveat", className="text-muted d-block mt-2"),
    html.Hr(),
    html.H5("2 · The sector ranking is the same everywhere"),
    html.P(f'Each country\'s ranking of sectors by ROA matches the average of the other countries (match {sg(SEC["agreement"], 2)} on a −1 to +1 scale, positive in {SEC["agreement_positive"]} of {SEC["agreement_n"]} countries). '
           f'If sector labels are shuffled inside each country the match falls to {sg(SEC["placebo"], 2)}, so the agreement is real.', className="text-muted"),
    graph(sector_fig(), h=420),
    cap("Average ROA of each sector relative to its country's median company. Technology and consumer-defensive companies sit above the median in most countries; financials sit below it almost everywhere (financial ROA is structurally low)."),
    html.Hr(),
    html.H5("3 · EU companies earn less than non-EU companies, but this is a description, not an effect"),
    dbc.Row([dbc.Col([graph(eu_fig(), h=280), cap("Each dot is a country (all 34 here; the charts above use the 28 with at least 30 listed companies); dotted lines are group averages. ROA is adjusted for sector, company size and year. The headline gap compares companies directly, so it differs a little from the gap between these country averages.")], md=7),
             dbc.Col([html.H2(f'{sg(AB1["effect"])} pp', className="big"), html.Small("EU minus non-EU, ROA (points)", className="text-muted d-block mb-2"),
                      html.P(f'The gap holds in every robustness check (same direction in {AB1["robust_share"]:.0%}). About {1 - EUC_CTRL["coef"] / AB1["effect"]:.0%} of it is the EU being more developed; {sg(EUC_CTRL["coef"])} pp remains when we compare countries at the same development level. '
                             'It is concentrated in smaller companies and close to zero for the largest ones.'),
                      html.Small("Countries cannot be randomly assigned to the EU, so the gap also reflects tax hubs, accounting rules and which companies are listed. Do not read it as the effect of membership.", className="text-muted")], md=5)]),
    stats_toggle(f'p = {AB1["p_primary"]:.2g} (wild cluster bootstrap), country permutation p = {AB1["p_permutation"]:.2g}; 95% range {sg(AB1["ci_low"], 2)} to {sg(AB1["ci_high"], 2)}; {AB1["clusters"]} countries, {AB1["n"]:,} company-years. No ESEF replication is possible (EU filers only).'),
    html.Hr(),
    html.Small(f"Listed companies, mainly {WINDOW}, {N_CO} of 34 countries (the other {len(country) - N_CO} have fewer than 30 listed companies). Associations, not causation. Notebooks 2.1 and 3.1 hold the full analysis.", className="text-muted"),
])


# ---------------------------------------------------------------- method
def pipeline():
    steps = [("Sources", "World Bank · IMF · Eurostat · Yahoo Finance · ESEF"), ("PostgreSQL", "staging → core → mart"), ("DuckDB", "one shared fact table"),
             ("Analysis", "17 ideas tested · 15 counted in the correction"), ("Dashboard", "this app · Dash + Plotly")]
    out = []
    for i, (t, sub) in enumerate(steps):
        out.append(html.Div([html.B(t), html.Small(sub, className="text-muted d-block")], className="pipe-step"))
        if i < len(steps) - 1:
            out.append(html.Div("→", className="pipe-arrow"))
    return html.Div(out, className="pipeline")


def hyp_index():
    order = sorted(HYP, key=HNUM)
    rows = [html.Tr([html.Td(hlink(h)), html.Td(hlink(h, HYP[h]["title"], "plain-link")), html.Td(status_badge(h)[0] + (" (exploratory)" if h in EXPLORATORY else ""))]) for h in order]
    return dbc.Table([html.Thead(html.Tr([html.Th(""), html.Th("Question"), html.Th("Result")])), html.Tbody(rows)], size="sm", responsive=True)


layout_about = html.Div([
    html.H5("About this project"),
    html.P(f"I'm {AUTHOR}. I wanted to know whether the economic and social conditions of a country show up in how its listed companies perform, and to answer it "
           "with a pipeline I could defend end to end rather than a one-off chart."),
    html.H6("What I built", className="subhead"),
    html.Ul([html.Li("A data pipeline that pulls 35 country indicators and 10,000+ company-years from five sources, lands them in PostgreSQL and exports a DuckDB analysis database."),
             html.Li("A structured test of 17 ideas (hypotheses), with a wild cluster bootstrap (only 34 countries) and a correction for testing many ideas at once (Benjamini–Hochberg). 15 of the 17 are counted in the correction; the other 2 are robustness and data checks."),
             html.Li("A replication on a second filing source (ESEF) and a leave-one-country-out check."),
             html.Li("This dashboard, which is built to be read by people who don't do statistics.")]),
    html.H6("What I would highlight", className="subhead"),
    html.Ul([html.Li("Honest reporting: exploratory findings are labelled, null results are listed, caveats are on the page."),
             html.Li("Tools: Python, pandas, DuckDB, PostgreSQL, statsmodels, Plotly Dash, Docker.")]),
    html.P([html.A("GitHub profile", href=GITHUB, target="_blank"), " · ", html.A("project repository", href=REPO, target="_blank")]),
    glossary(),
])

layout_method = html.Div([
    html.H6("How the pieces fit"), pipeline(), cap("From raw data to what you see on screen."),
    dbc.Accordion([
        dbc.AccordionItem(html.Ul([
            html.Li("Sources: World Bank (WDI and Worldwide Governance Indicators), IMF DataMapper, Eurostat, Yahoo Finance (an unofficial route to annual accounts and monthly prices), and official EU filings (ESEF)."),
            html.Li("Storage: PostgreSQL (staging / core / mart) exported to DuckDB; one shared company financial fact table."),
            html.Li("Method: country conditions from the year before vs company ROA, with errors grouped by country; a re-sampling method (wild cluster bootstrap) that suits few countries; a correction for testing many ideas (Benjamini–Hochberg); replication on ESEF."),
            html.Li("Countries dropped for lack of company data: Bulgaria, Croatia, Cyprus, Slovenia, Slovakia, and Malta (1 firm).")]), title="What was built"),
        dbc.AccordionItem(html.Ul([
            html.Li("Thin company samples (fewer than 50 firms) for Estonia, Latvia, Lithuania, Czechia and Ireland: markets too small."),
            html.Li("Listed firms only; Yahoo is unofficial and skewed towards some sectors; ESEF has no sector."),
            html.Li("34 countries limit what can be separated: development, institutions and openness move together."),
            html.Li("H1, H2, H3, H5 and H13 were suggested by looking at the same data; treat as exploratory."),
            html.Li("Company-level tab (notebooks 2.1, 3.1): the equity-ratio and sector results are associations between companies inside countries; both traits were checked on the same data before the notebook was finalised, so they sit outside the 15-test correction. Simple comparisons of company averages between country groups are badly over-confident (about 65% of random splits look significant), which is why the comparisons use country-level inference."),
            html.Li("Yahoo and ESEF agree almost perfectly (H15), so ESEF replications are not independent for the companies in both sources."),
            html.Li(f"Company accounts cover {WINDOW} in practice (a few earlier and later filings exist but are too few to use on their own). Eurostat business statistics change definition between 2020 and 2021 (levels jump about 33%), so those tables are kept as separate series and used from 2021 onward only.")]), title="Caveats"),
        dbc.AccordionItem([html.P("Click any question to read its page.", className="text-muted small"), hyp_index()], title="Every hypothesis in detail"),
    ], start_collapsed=False, always_open=True),
    glossary(),
])

# ---------------------------------------------------------------- app shell
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"], title="Country conditions and company performance")
server = app.server
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div([html.H1(["Country conditions and ", html.Span("company performance")]),
              html.P(["34 countries, 10,000+ company-years. ", html.A(f"by {AUTHOR}", href=GITHUB, target="_blank"), " · ", html.A("code on GitHub", href=REPO, target="_blank")],
                     className="text-muted mb-0")], className="hero"),
    html.Div(dbc.Tabs([dbc.Tab(layout_overview, label="Overview", tab_id="overview"), dbc.Tab(layout_explore, label="Explore", tab_id="explore"),
                       dbc.Tab(layout_countries, label="Countries", tab_id="countries"), dbc.Tab(layout_company, label="Company level", tab_id="company"),
                       dbc.Tab(layout_method, label="Method & caveats", tab_id="method"),
                       dbc.Tab(layout_about, label="About", tab_id="about")],
                      active_tab="overview", className="mb-3"), id="main"),
    html.Div(id="hyp-page"),
    html.Footer(f"Built by {AUTHOR}."),
], fluid=True, className="pb-5", style={"maxWidth": "1400px"})


@callback(Output("main", "style"), Output("hyp-page", "children"), Input("url", "pathname"))
def route(path):
    h = (path or "").removeprefix("/h/") if (path or "").startswith("/h/") else None
    if h in HYP:
        return {"display": "none"}, hyp_page(h)
    if (path or "").startswith("/h/"):
        return {"display": "none"}, html.Div([dcc.Link("← Back to all findings", href="/", className="more-link"), html.H4("Finding not found", className="mt-3"), html.P("There is no page for that finding.", className="text-muted")])
    return {}, None


@callback(Output("x", "value"), Output("y", "value"), Input({"type": "preset", "index": ALL}, "n_clicks"), prevent_initial_call=True)
def apply_preset(_):
    t = ctx.triggered_id
    if not t or not any(_):
        return "dev_index", "roa"
    _, x, y = PRESETS[t["index"]]
    return x, y


@callback(Output("scatter", "figure"), Output("scatter-note", "children"), Input("x", "value"), Input("y", "value"), Input("colour", "value"), Input("sector", "value"))
def scatter(x, y, colour, sector):
    d = country_stats(sector).dropna(subset=[x, y])
    d = with_labels(d, x, y, 9).assign(eu_flag=lambda d: np.where(d.is_eu, "EU", "Non-EU"))
    colour = "eu_flag" if colour == "eu_group" else colour
    d = d.assign(_x=np.exp(d[x]) if x in LOGS else d[x])
    fig = px.scatter(d, x="_x", y=y, color=colour, text="lab", color_discrete_map={"EU": REDHI, "Non-EU": "#4cc9f0"}, custom_data=["name", "firms"], size="firms", size_max=18, template=TPL,
                     color_discrete_sequence=DISTINCT)
    if len(d) > 3:
        b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 20)
        fig.add_trace(go.Scatter(x=np.exp(xs) if x in LOGS else xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, dash="dash"), name="fit", showlegend=False, hoverinfo="skip"))
    fig.update_traces(textposition="top center", textfont_color=TEXT, selector=dict(mode="markers+text"))
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>" + nice(x) + ": %{x:,.3~s}<br>" + OUTCOMES.get(y, y) + ": %{y:,.2f}<br>Firms: %{customdata[1]}<extra></extra>", selector=dict(mode="markers+text"))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=nice(x) + (" (log scale)" if x in LOGS else ""), yaxis_title=OUTCOMES.get(y, y), legend=dict(orientation="h", y=-0.2, title=None))
    fig.update_xaxes(title_standoff=14, type="log" if x in LOGS else "linear")
    if x in LOGS:
        tv, tt = log_ticks(d["_x"].min(), d["_x"].max())
        fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)
    fig.update_yaxes(title_standoff=14)
    rho = d[x].corr(d[y], method="spearman")
    return fig, f"Match between the two (Spearman ρ, from −1 to +1) = {sg(rho, 2)} across {len(d)} countries" + ("" if sector in (None, "all") else f" with at least {MIN_SECTOR_FIRMS} {sector} firms") + ". Descriptive only; bubble size = number of firms."


@callback(Output("dl", "data"), Input("dl-btn", "n_clicks"), Input("sector", "value"), prevent_initial_call=True)
def download(n, sector):
    if ctx.triggered_id != "dl-btn":
        return None
    cols = ["name", "iso2", "region", "income_group", "eu_group", "firms", "roa", "margin", "loss", "dev_index", "inst_index"] + FEATS
    return dcc.send_data_frame(country_stats(sector)[cols].round(3).to_csv, f"gbc_country_table_{sector}.csv")


@callback(Output("map", "figure"), Input("map-metric", "value"), Input("map-scope", "value"), Input("sector", "value"))
def draw_map(metric, scope, sector):
    d = country_stats(sector).reset_index()
    label = {**OUTCOMES, **INDICES}.get(metric, nice(metric))
    d = d.dropna(subset=[metric])
    fig = px.choropleth(d, locations="iso3", color=metric, hover_name="name", color_continuous_scale=MAP_SCALE, template=TPL, labels={metric: label})
    geo = dict(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#4a4a52", showcountries=True, countrycolor=BG, showframe=True, framecolor=LINE, framewidth=1, showcoastlines=False, showocean=True, oceancolor=BG)
    if scope == "europe":
        geo.update(projection_type="mercator", lonaxis_range=[-25, 48], lataxis_range=[33, 71])
    else:
        geo.update(projection_type="robinson", lonaxis_range=[-170, 180], lataxis_range=[-56, 80])
    fig.update_geos(**geo)
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>" + label + ": %{z:,.2f}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=dict(text=label, side="right"), thickness=12, len=0.8))
    return fig


@callback(Output("countries", "value"), Input("countries", "value"))
def cap_countries(v):
    return (v or ["Germany"])[:3]


@callback(Output("country-cards", "children"), Output("country-bars", "figure"), Output("country-bars", "style"), Output("country-roa", "figure"),
          Input("countries", "value"), Input("theme", "value"), Input("c-sector", "value"), Input("years", "value"))
def country_view(names, theme, sector, years):
    picked = (names or ["Germany"])[:3]
    cmap = dict(zip(picked, CC))
    tile = {"md": 6} if len(picked) > 1 else {"md": True}
    cs = country_stats(sector, tuple(years))
    names = [n for n in picked if n in set(cs.name)] or picked[:1]
    missing = [n for n in picked if n not in set(cs.name)]
    rows = [(cs if n in set(cs.name) else country)[(cs if n in set(cs.name) else country).name == n].iloc[0] for n in names]
    cards = dbc.Row([dbc.Col([html.H5(r["name"], className="mb-2", style={"borderLeft": f"4px solid {cmap[r['name']]}", "paddingLeft": ".5rem"}), dbc.Row([
        dbc.Col(xs=6, **tile, children=kpi("Firms", f'{int(r.firms)}', "small sample" if r.small_firm_sample else "")), dbc.Col(xs=6, **tile, children=kpi("Mean ROA", f"{fm(r.roa)}%")),
        dbc.Col(xs=6, **tile, children=kpi("Development", f"{sg(r.dev_index, 2)} SD")), dbc.Col(xs=6, **tile, children=kpi("Institutions", f"{sg(r.inst_index, 2)} SD")),
        dbc.Col(xs=6, **tile, children=kpi("Loss-making", f"{r.loss:.0f}%"))],
        className="g-2")], md=12 // len(names)) for r in rows], className="g-3 mb-3")
    if missing:
        cards = html.Div([html.Div(f'Not shown: {", ".join(missing)} (fewer than {MIN_SECTOR_FIRMS} firms for this sector and years).', className="text-muted small mb-2"), cards])
    cols = [f for f in FEATS if theme == "all" or X["group"][f] == theme]
    iso = [r.name for r in rows]
    z = Z.loc[iso, cols].T.dropna(how="all")
    z.columns = names
    z = z.sort_values(names[0])
    bars = px.bar(z.reset_index().melt(id_vars="index", var_name="country", value_name="sd"), x="sd", y="index", color="country", orientation="h", barmode="group", template=TPL,
                  labels={"sd": "SD from the 34-country average", "index": ""}, color_discrete_map=cmap, category_orders={"country": names})
    h = max(380, 22 * len(z) * len(names)) + 90
    bars.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=90), yaxis_title=None, legend=dict(orientation="h", y=-0.1 if h > 600 else -0.2, title=None))
    bars.update_xaxes(title_standoff=14)
    bars.update_yaxes(tickvals=list(z.index), ticktext=[nice(i) for i in z.index])
    f = firm_slice(sector, tuple(years))
    f = f[f.iso3.isin(iso)].groupby(["iso3", "fiscal_year"]).roa_pp.agg(["median", "count"]).reset_index()
    f["country"] = f.iso3.map(dict(zip(iso, names)))
    roa = px.line(f, x="fiscal_year", y="median", color="country", markers=True, hover_data=["count"], template=TPL, labels={"median": "median ROA (%)", "fiscal_year": ""},
                  color_discrete_map=cmap, category_orders={"country": names})
    roa.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(title=None))
    roa.update_xaxes(dtick=1)
    roa.update_yaxes(title_standoff=14)
    return cards, bars, {"height": f"{h}px"}, roa


W_CAVEAT = {"W1": "An equity ratio can also be high because the company has been profitable (retained earnings), so this is an association, not proof that equity financing raises returns.",
            "W2": "Company size is a rank inside each country. No repeating pattern means other things (sector, age, ownership) matter more than size, not that size never matters."}


@callback(Output("w-title", "children"), Output("w-desc", "children"), Output("w-forest", "figure"), Output("w-note", "children"), Output("w-caveat", "children"), Input("w-trait", "value"))
def w_view(h):
    s = CO["summary"][h]
    k = round(s["same_sign_share"] * s["k"])
    return W_TEXT[h][0], W_TEXT[h][1], forest_fig(h), f'{k} of {s["k"]} countries have the same direction as the pooled estimate ({s["same_sign_share"]:.0%}).', W_CAVEAT[h]


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8050)), debug=False)
