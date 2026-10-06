"""Preview dashboard (Plotly Dash): what the project has learnt so far.

Run:  uv run --group dash python app/dash_app.py   ->  http://127.0.0.1:8050
Reads data/gbc.duckdb through src/analysis/prep.py and results/H*.json.
"""
import json
import pathlib
import sys

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis import prep  # noqa: E402

# ---------------------------------------------------------------- data
con = prep.connect()
C = prep.load_country(con)
F = prep.load_firms(con, C)
sys.path.insert(0, str(ROOT / "src" / "analysis"))
import utils  # noqa: E402

HYP = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("H*.json"), key=lambda p: int(p.stem[1:]))}
NOT_A_TEST = {"H11", "H15"}   # robustness / data-quality entries, outside the multiple-testing family
_fam = pd.Series({h: r["p_primary"] for h, r in HYP.items() if h not in NOT_A_TEST})
Q = utils.bh(_fam)
N_PASS = int((Q < 0.10).sum())

VERDICT_COLOR = {"supported": "#2a9d8f", "contradicted": "#e63946", "not supported": "#8d99ae"}
EXPLORATORY = {"H1", "H2", "H3", "H5", "H13"}   # suggested by exploratory scans of the same data

dim = C["dim"]
prof = C["profile"]
firms_by_country = F.groupby("iso3").agg(firms=("company_id", "nunique"), roa=("roa_pp", "mean"),
                                         margin=("net_margin_w", "mean"), loss=("loss_pp", "mean"))
country = dim.join(firms_by_country).join(prof)
country["dev_index"] = C["dev_idx"]
country["inst_index"] = C["inst_idx"]
country = country[country.firms >= prep.MIN_FIRMS].copy()
country["label"] = country.name

FEATURE_OPTIONS = [{"label": f.replace("_", " "), "value": f} for f in C["feat_cols"]] + [
    {"label": "development index", "value": "dev_index"}, {"label": "institutions index", "value": "inst_index"}]
OUTCOME_OPTIONS = [{"label": "ROA (pp)", "value": "roa"}, {"label": "net margin", "value": "margin"},
                   {"label": "loss-making firm-years (%)", "value": "loss"}]


def ci(r):
    return "n/a" if r["ci_low"] is None or r["ci_high"] is None else f'{r["ci_low"]:.2f} to {r["ci_high"]:.2f}'


def kpi(title, value, sub=""):
    return dbc.Card(dbc.CardBody([html.Small(title, className="text-muted"), html.H3(value), html.Small(sub, className="text-muted")]))


# ---------------------------------------------------------------- overview: lead with what holds
# plain-English headline and the number to show, for the results that hold (shown first, in this order)
FINDINGS = [
    ("H0", "Firms in more developed countries earn slightly less on their assets", lambda r: f'{r["effect"]:+.1f} pp ROA', "per +1 SD of development"),
    ("H13", "The effect sits in small firms; the largest third of firms show almost none", lambda r: f'{r["effect"]:+.1f} pp', "development × size interaction"),
    ("H11", "The main finding survives dropping any single country", lambda r: f'{r["effect"]:+.1f} pp', "weakest of 34 leave-one-out runs"),
    ("H5", "Countries with deeper bank credit have deeper stock markets", lambda r: f'ρ = {r["effect"]:+.2f}', "credit vs market value, 20 countries"),
    ("H6", "In the EU, higher labour cost goes with higher productivity", lambda r: f'ρ = {r["effect"]:+.2f}', "hourly labour cost vs output per worker"),
    ("H1", "Surprise: stronger institutions go with MORE loss-making firms (small open economies explain it)", lambda r: f'{r["effect"]:+.1f} pp', "loss probability per +1 SD institutions"),
    ("H15", "The two data sources agree, so the numbers can be trusted", lambda r: f'ρ = {r["effect"]:+.2f}', "Yahoo vs ESEF ROA, same company-year"),
]
HOLDS = [f[0] for f in FINDINGS if f[0] in HYP]
REST = [h for h in HYP if h not in HOLDS]


def finding_card(h, headline, big, unit):
    r = HYP[h]
    tags = []
    if h in NOT_A_TEST:
        tags.append(dbc.Badge("check, not a new claim", color="secondary", className="me-1"))
    else:
        tags.append(dbc.Badge("q < 0.001" if Q[h] < 0.001 else f"q = {Q[h]:.3f}", color="success", className="me-1"))
    if h in EXPLORATORY:
        tags.append(dbc.Badge("exploratory", color="warning", text_color="dark", className="me-1"))
    rep = r.get("replication_coef")
    if rep is not None and not pd.isna(rep) and r.get("replication_same_sign"):
        tags.append(dbc.Badge("holds on ESEF", color="light", text_color="dark", className="me-1"))
    color = "#e63946" if r["verdict"] == "contradicted" else "#2a9d8f"
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Small(f'{h} · {r["title"]}', className="text-muted"),
        html.H5(headline, className="mt-1"),
        html.H2(big, style={"color": color}), html.Small(unit, className="text-muted d-block mb-2"), html.Div(tags)]),
        className="h-100 shadow-sm", style={"borderTop": f"4px solid {color}"}), md=6, lg=4, className="mb-3")


def evidence_fig():
    ids = [h for h in HOLDS if h not in NOT_A_TEST]
    d = pd.DataFrame({"h": ids, "title": [HYP[h]["title"] for h in ids], "strength": [-np.log10(max(HYP[h]["p_primary"], 1e-9)) for h in ids],
                      "verdict": [HYP[h]["verdict"] for h in ids]})
    fig = px.bar(d, x="strength", y=d.h + " " + d.title, orientation="h", color="verdict", color_discrete_map=VERDICT_COLOR, template="plotly_white",
                 labels={"strength": "evidence strength, −log10(p)", "y": ""})
    fig.add_vline(x=-np.log10(0.05), line_dash="dot", annotation_text="p = 0.05")
    fig.update_layout(height=60 + 42 * len(d), margin=dict(l=0, r=10, t=10, b=40), showlegend=False, yaxis_autorange="reversed")
    return fig


def rest_table():
    head = html.Thead(html.Tr([html.Th(x) for x in ["", "Question", "Effect", "p", "q (BH)"]]))
    rows = [html.Tr([html.Td(h), html.Td(HYP[h]["title"]), html.Td(f'{HYP[h]["effect"]:.2f}', title=HYP[h]["effect_unit"]), html.Td(f'{HYP[h]["p_primary"]:.2g}'),
                     html.Td(f"{Q[h]:.2f}" if h in Q.index else "–")]) for h in REST]
    return dbc.Table([head, html.Tbody(rows)], size="sm", responsive=True, className="small text-muted")


layout_overview = html.Div([
    dbc.Row([dbc.Col(kpi("Countries", f"{len(country)}", "21 EU + 13 non-EU")),
             dbc.Col(kpi("Companies", f'{F.company_id.nunique():,}', "listed, Yahoo Finance")),
             dbc.Col(kpi("Firm-years", f"{len(F):,}", f"fiscal {F.fiscal_year.min()}–{F.fiscal_year.max()}")),
             dbc.Col(kpi("Country indicators", f'{len(C["feat_cols"])}', "World Bank, WGI, IMF, Eurostat")),
             dbc.Col(kpi("Findings that hold", f"{N_PASS}", f"of {len(Q)} tests, after multiple-testing correction"))],
            className="g-3 mb-4"),
    dbc.Alert([html.B("Question: "), "how do the social and economic conditions of a country relate to the performance of the companies that operate there?"], color="light"),
    html.H4("What we found"),
    dbc.Row([finding_card(h, hl, fn(HYP[h]), u) for h, hl, fn, u in FINDINGS if h in HYP], className="g-3"),
    html.H6("How strong is the evidence?", className="mt-3"), dcc.Graph(figure=evidence_fig(), config={"displayModeBar": False}),
    dbc.Accordion([dbc.AccordionItem([html.P(f"{len(REST)} further questions were tested; the data show no clear effect. They are listed for completeness: "
                                             "a clear answer of \"no\" is still an answer.", className="small text-muted"), rest_table()],
                                     title=f"Also tested: no clear evidence ({len(REST)})")], start_collapsed=True, className="mt-3"),
    html.Small("q = Benjamini–Hochberg corrected p-value across the 15 primary tests. exploratory = suggested by looking at the same data. "
               "Only 34 countries are independent observations: inference uses a wild cluster bootstrap; results are associations, not causation.",
               className="text-muted d-block mt-3"),
])

layout_main = html.Div([
    html.H5("Main finding (H0): richer, better-governed countries have slightly LOWER firm returns"),
    html.P("Pick any country condition and any firm outcome. Each dot is a country (mean over 2019–2023); the line is a simple fit across countries.", className="text-muted"),
    dbc.Row([dbc.Col([dbc.Label("Country condition (x)"), dcc.Dropdown(FEATURE_OPTIONS, "dev_index", id="x", clearable=False)], md=4),
             dbc.Col([dbc.Label("Firm outcome (y)"), dcc.Dropdown(OUTCOME_OPTIONS, "roa", id="y", clearable=False)], md=4),
             dbc.Col([dbc.Label("Colour by"), dcc.Dropdown([{"label": "region", "value": "region"}, {"label": "income group", "value": "income_group"},
                                                           {"label": "EU group", "value": "eu_group"}], "region", id="colour", clearable=False)], md=4)],
            className="mb-3"),
    dcc.Graph(id="scatter"), html.Div(id="scatter-note", className="text-muted"),
    html.Hr(),
    html.H6("Primary test"), html.P(HYP["H0"]["notes"][0]), html.P(HYP["H0"]["notes"][1]),
])

_d = F.dropna(subset=["roa_pp", "size_rank", "dev_z"]).copy()
_d["size_group"] = pd.cut(_d.size_rank, [0, 1 / 3, 2 / 3, 1.0001], labels=["small", "medium", "large"], include_lowest=True)
_d["dev_group"] = pd.qcut(_d.iso3.map(C["dev_idx"]), 3, labels=["least developed", "middle", "most developed"])
_grid = _d.pivot_table(index="size_group", columns="dev_group", values="roa_pp", aggfunc="median", observed=True)
_by = []
for g, gd in _d.groupby("size_group", observed=True):
    m = utils.fit_cluster("roa_pp ~ dev_z + C(sector) + C(fiscal_year)", gd)
    _by.append({"size": g, "coef": m.params["dev_z"], "lo": m.params["dev_z"] - 1.96 * m.bse["dev_z"], "hi": m.params["dev_z"] + 1.96 * m.bse["dev_z"]})
_by = pd.DataFrame(_by)
_size_fig = px.bar(_by, x="size", y="coef", error_y=_by.hi - _by.coef, error_y_minus=_by.coef - _by.lo, template="plotly_white",
                   labels={"coef": "pp ROA per +1 SD development", "size": "company size within its country"})
_size_fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
_heat = px.imshow(_grid, text_auto=".1f", color_continuous_scale="RdBu", color_continuous_midpoint=float(_grid.values.mean()), aspect="auto", template="plotly_white")
_heat.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
layout_size = html.Div([
    html.H5("Who carries the country effect? Small firms (H13)"),
    html.P(HYP["H13"]["notes"][0]),
    dbc.Row([dbc.Col([html.H6("Development → ROA by company size"), dcc.Graph(figure=_size_fig, config={"displayModeBar": False})], md=6),
             dbc.Col([html.H6("Median ROA (pp): size × country development"), dcc.Graph(figure=_heat, config={"displayModeBar": False})], md=6)]),
    html.Small(HYP["H13"]["notes"][2] + " Exploratory: this refinement grew out of looking at H0.", className="text-muted"),
])

layout_countries = html.Div([
    html.H5("Country explorer"),
    dbc.Row([dbc.Col(dcc.Dropdown(sorted(country.name), "Germany", id="country", clearable=False), md=4)], className="mb-3"),
    html.Div(id="country-cards"), dcc.Graph(id="country-bars"),
    html.H6("Firm return (ROA) by year"), dcc.Graph(id="country-roa"),
])

layout_hyp = html.Div([
    html.H5("Hypothesis detail"),
    dcc.Dropdown([{"label": f'{h}: {HYP[h]["title"]}' + ("" if h in HOLDS else "  (no clear evidence)"), "value": h} for h in HOLDS + REST], "H13", id="hyp", clearable=False, className="mb-3"),
    html.Div(id="hyp-detail"),
])

layout_about = html.Div([
    html.H5("What was built"),
    html.Ul([html.Li("Sources: World Bank WDI and WGI, IMF DataMapper, Eurostat, Yahoo Finance (yfinance, unofficial; annual accounts and monthly prices), ESEF/xBRL filings (EU), 13 extra World Bank indicators."),
             html.Li("Storage: PostgreSQL (staging / core / mart) exported to DuckDB; one shared company financial fact table."),
             html.Li("Method: lagged (t-1) country features vs firm ROA, clustered by country; wild cluster bootstrap; Benjamini–Hochberg across the primary tests; ESEF replication."),
             html.Li("Dropped: BGR, HRV, CYP, SVN, SVK (no firm data) and Malta (1 firm).")]),
    html.H5("Caveats"),
    html.Ul([html.Li("Small samples for EST, LVA, LTU, CZE, IRL (markets too small)."),
             html.Li("Listed firms only; yfinance is unofficial and sector-skewed; ESEF has no sector."),
             html.Li("34 countries limit what can be separated: development, institutions and openness move together."),
             html.Li("H1, H2, H3, H5 and H13 were suggested by looking at the same data; treat as exploratory."),
             html.Li("Yahoo and ESEF agree almost perfectly (H15), so ESEF replications are not independent for the companies in both sources."),
             html.Li("Series breaks (e.g. Eurostat 2020/21) are not yet handled.")]),
])

# ---------------------------------------------------------------- app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Country conditions vs firm performance")
app.layout = dbc.Container([
    html.H2("Country conditions and company performance", className="mt-3 mb-0"),
    html.P("Preview dashboard: what we have learnt so far", className="text-muted"),
    dbc.Tabs([dbc.Tab(layout_overview, label="Overview", tab_id="overview"), dbc.Tab(layout_main, label="Main finding", tab_id="main"),
              dbc.Tab(layout_size, label="Who it affects", tab_id="size"), dbc.Tab(layout_countries, label="Countries", tab_id="countries"), dbc.Tab(layout_hyp, label="Hypotheses", tab_id="hyp"),
              dbc.Tab(layout_about, label="Method & caveats", tab_id="about")], active_tab="overview", className="mb-3"),
], fluid=True, className="pb-5")


@callback(Output("scatter", "figure"), Output("scatter-note", "children"),
          Input("x", "value"), Input("y", "value"), Input("colour", "value"))
def scatter(x, y, colour):
    d = country.dropna(subset=[x, y])
    fig = px.scatter(d, x=x, y=y, color=colour, text="iso2", hover_name="name", size="firms", size_max=28, template="plotly_white",
                     hover_data={"firms": True, "iso2": False})
    if len(d) > 3:
        b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color="#555", dash="dash"), name="fit", showlegend=False))
    fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    rho = d[x].corr(d[y], method="spearman")
    return fig, f"Spearman ρ = {rho:+.2f} across {len(d)} countries (descriptive; bubble size = number of firms)."


@callback(Output("country-cards", "children"), Output("country-bars", "figure"), Output("country-roa", "figure"), Input("country", "value"))
def country_view(name):
    row = country[country.name == name].iloc[0]
    iso = row.name
    cards = dbc.Row([dbc.Col(kpi("Firms", f"{int(row.firms)}", "small sample" if row.small_firm_sample else "")),
                     dbc.Col(kpi("Mean ROA", f"{row.roa:.1f} pp")), dbc.Col(kpi("Development index", f"{row.dev_index:+.2f} SD")),
                     dbc.Col(kpi("Institutions index", f"{row.inst_index:+.2f} SD")),
                     dbc.Col(kpi("Loss-making firm-years", f"{row.loss:.0f}%"))], className="g-3 mb-3")
    z = ((prof - prof.mean()) / prof.std()).loc[iso].dropna().sort_values()
    bars = px.bar(z.rename("SD from the 34-country mean").reset_index(), x="SD from the 34-country mean", y="short" if "short" in z.reset_index().columns else "index",
                  orientation="h", template="plotly_white")
    bars.update_layout(height=max(400, 18 * len(z)), margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
    f = F[F.iso3 == iso].groupby("fiscal_year").roa_pp.agg(["median", "count"]).reset_index()
    roa = px.bar(f, x="fiscal_year", y="median", hover_data=["count"], template="plotly_white", labels={"median": "median ROA (pp)"})
    roa.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    return cards, bars, roa


@callback(Output("hyp-detail", "children"), Input("hyp", "value"))
def hyp_detail(h):
    r = HYP[h]
    rep = ("no replication" if r.get("replication_coef") is None or pd.isna(r.get("replication_coef"))
           else f'ESEF: {r["replication_coef"]:.2f} (p = {r["replication_p"]:.3g}, {"same" if r["replication_same_sign"] else "opposite"} sign)')
    return html.Div([
        html.H4([r["title"], " ", dbc.Badge(r["verdict"], style={"backgroundColor": VERDICT_COLOR.get(r["verdict"], "#999")})]),
        html.P(r["statement"], className="lead"),
        html.P(f'Effect: {r["effect"]:.3f} {r["effect_unit"]}  (95% CI {ci(r)}); p = {r["p_primary"]:.3g} ({r["p_method"]}); '
               f'n = {r["n"]:,}, {r["clusters"]} countries. Robust in {r["robust_share"]:.0%} of variants. {rep}.'),
        html.Ul([html.Li(n) for n in r["notes"]]),
        html.Small(f'Details in notebook {r["notebook"]}' + ("  ·  exploratory (suggested by scans of the same data)" if h in EXPLORATORY else ""), className="text-muted"),
    ])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
