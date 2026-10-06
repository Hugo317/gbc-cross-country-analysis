"""Dashboard (Plotly Dash): how country conditions relate to company performance.

Run:   uv run --group dash python app/dash_app.py   ->  http://127.0.0.1:8050
Reads only app/data/ (built by app/build_extract.py), so it can be hosted without data/gbc.duckdb.
"""
import json
import os
import pathlib

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
RED_SCALE = [[0, "#1a1114"], [0.5, BLOOD], [1, "#ff4d50"]]
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
N_PASS = int((Q < 0.10).sum())
EXPLORATORY = {"H1", "H2", "H3", "H5", "H13"}    # suggested by exploratory scans of the same data
VERDICT_COLOR = {"supported": REDHI, "contradicted": AMBER, "not supported": "#6b6b74"}
FEATS = X["feat_cols"]
pretty = lambda s: s.replace("_", " ")
INDICES = {"dev_index": "development index", "inst_index": "institutions index"}
OUTCOMES = {"roa": "ROA (pp)", "margin": "net margin", "loss": "loss-making firm-years (%)"}
FEATURE_OPTIONS = [{"label": pretty(f), "value": f} for f in FEATS] + [{"label": v, "value": k} for k, v in INDICES.items()]
OUTCOME_OPTIONS = [{"label": v, "value": k} for k, v in OUTCOMES.items()]
MAP_OPTIONS = OUTCOME_OPTIONS + [{"label": v, "value": k} for k, v in INDICES.items()] + [{"label": pretty(f), "value": f} for f in FEATS]
Z = ((country[FEATS] - country[FEATS].mean()) / country[FEATS].std())
GROUPS = sorted(set(X["group"].values()))
PRESETS = [("Richer countries, lower returns", "dev_index", "roa"), ("Institutions vs loss-making firms", "inst_index", "loss"),
           ("Credit depth vs net margin", "private_credit_gdp", "margin"), ("Trade openness vs ROA", "trade_gdp", "roa"),
           ("Life expectancy vs ROA", "life_expectancy", "roa")]


SECTORS = sorted(s for s in F.sector.dropna().unique() if s != "Unknown")
SECTOR_OPTIONS = [{"label": "All sectors", "value": "all"}] + [{"label": s, "value": s} for s in SECTORS]
YEARS = (int(F.fiscal_year.clip(2022, 2025).min()), int(F.fiscal_year.clip(2022, 2025).max()))
MIN_FIRMS = 10


def firm_slice(sector="all", years=None):
    d = F if sector in (None, "all") else F[F.sector == sector]
    return d if years is None else d[d.fiscal_year.between(*years)]


def country_stats(sector="all", years=None):
    """Country table whose firm outcomes are recomputed for a sector and/or fiscal-year window (thin countries drop out)."""
    filtered = sector not in (None, "all") or years is not None
    a = firm_slice(sector, years).groupby("iso3").agg(firms=("company_id", "nunique"), roa=("roa_pp", "mean"), margin=("net_margin_w", "mean"), loss=("loss_pp", "mean"))
    a = a[a.firms >= (3 if filtered else MIN_FIRMS)]
    return country.drop(columns=["firms", "roa", "margin", "loss"]).join(a, how="inner")


def cap(text):
    return html.Small(text, className="text-muted d-block mb-3 caption")


def ci(r):
    return "n/a" if r["ci_low"] is None or r["ci_high"] is None else f'{r["ci_low"]:.2f} to {r["ci_high"]:.2f}'


def kpi(title, value, sub=""):
    return dbc.Card(dbc.CardBody([html.Small(title, className="text-muted"), html.H3(value), html.Small(sub, className="text-muted")]), className="kpi h-100")


def stats_toggle(*lines):
    return html.Details([html.Summary("Stats detail"), *[html.Div(l) for l in lines]], className="stats")


# ---------------------------------------------------------------- overview
# plain-English headline and the number to show; (id, headline, number, unit, plain reading)
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
CONFIRMED = [h for h in HOLDS if h not in EXPLORATORY and h != "H0"]
EXPLO = [h for h in HOLDS if h in EXPLORATORY]
FDICT = {f[0]: f for f in FINDINGS}
REST = [h for h in HYP if h not in HOLDS]


def finding_card(h, headline, big, unit, hero=False):
    r = HYP[h]
    warn = r["verdict"] == "contradicted"
    tags = [dbc.Badge("check, not a new claim", color="secondary", className="me-1")] if h in NOT_A_TEST else \
           [dbc.Badge("highly robust" if Q[h] < 0.001 else "holds up", className="me-1 bg-red")]
    if h in EXPLORATORY:
        tags.append(dbc.Badge("exploratory", color="warning", text_color="dark", className="me-1"))
    rep = r.get("replication_coef")
    if rep is not None and not pd.isna(rep) and r.get("replication_same_sign"):
        tags.append(dbc.Badge("confirmed on 2nd data source", className="me-1 bg-ash"))
    qtxt = "" if h in NOT_A_TEST else f'q {"< 0.001" if Q[h] < 0.001 else f"= {Q[h]:.3f}"} (Benjamini–Hochberg corrected across 15 tests). '
    stats = stats_toggle(f'{qtxt}p = {r["p_primary"]:.2g} ({r["p_method"]}); 95% CI {ci(r)}; n = {r["n"]:,} firm-years in {r["clusters"]} countries.')
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Small(f'{h} · {r["title"]}', className="text-muted"), html.H5(headline, className="mt-1"),
        html.H2(big, className="big warn" if warn else "big"), html.Small(unit, className="text-muted d-block mb-2"), html.Div(tags), stats]),
        className="finding h-100 shadow-sm" + (" warn" if warn else "")), md=12 if hero else 6, lg=12 if hero else 4, className="mb-3")


def evidence_fig():
    ids = [h for h in HOLDS if h not in NOT_A_TEST]
    d = pd.DataFrame({"h": ids, "title": [HYP[h]["title"] for h in ids], "strength": [-np.log10(max(HYP[h]["p_primary"], 1e-9)) for h in ids],
                      "verdict": [HYP[h]["verdict"] for h in ids]})
    fig = px.bar(d, x="strength", y=d.h + " " + d.title.str.slice(0, 32), orientation="h", color="verdict", color_discrete_map=VERDICT_COLOR, template=TPL,
                 labels={"strength": "evidence strength (higher = harder to explain by chance)", "y": ""})
    fig.add_vline(x=-np.log10(0.05), line_dash="dot", line_color=MUTED, annotation_text="conventional threshold", annotation_font_color=MUTED)
    fig.update_layout(height=60 + 42 * len(d), margin=dict(l=10, r=10, t=10, b=40), showlegend=False, yaxis_autorange="reversed")
    fig.update_yaxes(automargin=True)
    return fig


def rest_table():
    head = html.Thead(html.Tr([html.Th(x) for x in ["", "Question", "Effect", "p", "q (BH)"]]))
    rows = [html.Tr([html.Td(h), html.Td(HYP[h]["title"]), html.Td(f'{HYP[h]["effect"]:.2f}', title=HYP[h]["effect_unit"]), html.Td(f'{HYP[h]["p_primary"]:.2g}'),
                     html.Td(f"{Q[h]:.2f}" if h in Q.index else "–")]) for h in REST]
    return dbc.Table([head, html.Tbody(rows)], size="sm", responsive=True, className="small text-muted")


def hero_card():
    h, hl, fn, unit = FDICT["H0"]
    r = HYP[h]
    d = country
    fig = px.scatter(d.reset_index(), x="dev_index", y="roa", text="iso2", hover_name="name", template=TPL, color_discrete_sequence=[REDHI],
                     labels={"dev_index": "development index (SD)", "roa": "mean ROA (pp)"})
    b = np.polyfit(d.dev_index, d.roa, 1)
    xs = np.linspace(d.dev_index.min(), d.dev_index.max(), 20)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, dash="dash"), showlegend=False))
    fig.update_traces(textposition="top center", textfont=dict(size=9, color=MUTED), marker=dict(size=9, opacity=.9), selector=dict(mode="markers+text"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    return dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([html.Small(f'{h} · {r["title"]}', className="text-muted"), html.H3(hl, className="mt-1"), html.H1(fn(r), className="big"),
                 html.Small(unit, className="text-muted d-block mb-2"), dbc.Badge("highly robust", className="me-1 bg-red"), dbc.Badge("confirmed on 2nd data source", className="bg-ash"),
                 stats_toggle(f'p = {r["p_primary"]:.2g} ({r["p_method"]}); 95% CI {ci(r)}; n = {r["n"]:,} firm-years in {r["clusters"]} countries.')], md=5),
        dbc.Col([dcc.Graph(figure=fig, config=CFG), cap("Each dot is a country. Further right = more developed; higher up = firms earn more on their assets. The dashed line slopes down.")], md=7)],
        align="center")), className="finding hero-card shadow-sm")


layout_overview = html.Div([
    dbc.Row([dbc.Col(xs=6, md=True, children=kpi("Countries", f"{len(country)}", "21 EU + 13 non-EU")),
             dbc.Col(xs=6, md=True, children=kpi("Companies", f'{X["n_companies"]:,}', "listed, Yahoo Finance")),
             dbc.Col(xs=6, md=True, children=kpi("Firm-years", f'{X["n_firm_years"]:,}', f'fiscal {X["years"][0]}–{X["years"][1]}')),
             dbc.Col(xs=6, md=True, children=kpi("Country indicators", f"{len(FEATS)}", "World Bank, WGI, IMF, Eurostat")),
             dbc.Col(xs=6, md=True, children=kpi("Findings that hold", f"{N_PASS}", f"of {len(Q)} tests, after multiple-testing correction"))], className="g-3 mb-4"),
    dbc.Alert([html.B("Question: "), "how do the social and economic conditions of a country relate to the performance of the companies that operate there?"], className="alert-quest"),
    html.H4("The headline result"),
    hero_card(),
    html.H4("Also holds up", className="mt-4"), html.P("Checks that back the headline up, and findings that were planned in advance.", className="text-muted small"),
    dbc.Row([finding_card(*FDICT[h][:2], FDICT[h][2](HYP[h]), FDICT[h][3]) for h in CONFIRMED], className="g-3"),
    html.H4("Exploratory findings", className="mt-4"),
    html.P("These came from looking at the same data, so treat them as leads rather than proof.", className="text-muted small"),
    dbc.Row([finding_card(*FDICT[h][:2], FDICT[h][2](HYP[h]), FDICT[h][3]) for h in EXPLO], className="g-3"),
    html.H6("How strong is the evidence?", className="mt-3"), dcc.Graph(figure=evidence_fig(), config=CFG),
    cap("Each bar is one finding; the longer the bar, the harder the result is to explain by chance. Bars past the dotted line pass the conventional threshold."),
    dbc.Accordion([dbc.AccordionItem([html.P(f"{len(REST)} further questions were tested; the data show no clear effect. They are listed for completeness: "
                                             "a clear answer of \"no\" is still an answer.", className="small text-muted"), rest_table()],
                                     title=f"Also tested: no clear evidence ({len(REST)})")], start_collapsed=True, className="mt-3"),
    html.Small("Results are associations, not causation. Only 34 countries are independent observations, so inference uses a method built for few groups (wild cluster bootstrap).",
               className="text-muted d-block mt-3"),
])

# ---------------------------------------------------------------- explore: scatter + map + size
_se = pd.DataFrame(X["size_effects"])
_size_fig = px.bar(_se, x="size", y="coef", error_y=_se.hi - _se.coef, error_y_minus=_se.coef - _se.lo, template=TPL, color_discrete_sequence=[REDHI],
                   labels={"coef": "pp ROA per +1 SD development", "size": "company size within its country"})
_size_fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
_g = X["grid"]
_heat = px.imshow(pd.DataFrame(_g["data"], index=_g["index"], columns=_g["columns"]), text_auto=".1f", color_continuous_scale=RED_SCALE, aspect="auto", template=TPL)
_heat.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)

layout_explore = html.Div([
    html.H5("Explore: pick any country condition and any firm outcome"),
    html.P("Each dot is a country (average over 2019–2023). Start from a preset or choose your own.", className="text-muted"),
    html.Div([dbc.Button(lbl, id={"type": "preset", "index": i}, className="btn-preset me-2 mb-2", n_clicks=0) for i, (lbl, _, _) in enumerate(PRESETS)]),
    dbc.Row([dbc.Col([dbc.Label("Country condition (x)"), dcc.Dropdown(FEATURE_OPTIONS, "dev_index", id="x", clearable=False)], md=4),
             dbc.Col([dbc.Label("Firm outcome (y)"), dcc.Dropdown(OUTCOME_OPTIONS, "roa", id="y", clearable=False)], md=4),
             dbc.Col([dbc.Label("Colour by"), dcc.Dropdown([{"label": "region", "value": "region"}, {"label": "income group", "value": "income_group"},
                                                           {"label": "EU group", "value": "eu_group"}], "region", id="colour", clearable=False)], md=4)], className="mb-3"),
    dbc.Row([dbc.Col([dbc.Label("Sector (recomputes firm outcomes)"), dcc.Dropdown(SECTOR_OPTIONS, "all", id="sector", clearable=False)], md=4),
             dbc.Col([dbc.Button("Download country table (CSV)", id="dl-btn", className="btn-preset mt-4"), dcc.Download(id="dl")], md=4)], className="mb-3"),
    dcc.Graph(id="scatter", config=CFG), html.Div(id="scatter-note", className="text-muted"),
    cap("Each bubble is a country; bigger bubbles have more firms. A slope means the condition on the x-axis goes with the firm outcome on the y-axis. It does not prove cause. Use the camera icon on a chart to save a PNG."),
    html.Hr(),
    html.H5("Map"),
    dbc.Row([dbc.Col([dbc.Label("Show"), dcc.Dropdown(MAP_OPTIONS, "roa", id="map-metric", clearable=False)], md=6),
             dbc.Col([dbc.Label("Area"), dbc.RadioItems(options=[{"label": "Europe", "value": "europe"}, {"label": "World", "value": "world"}], value="europe", id="map-scope", inline=True)], md=6)],
            className="mb-2"),
    dcc.Graph(id="map", config=CFG), cap("Darker = lower, brighter red = higher. Grey countries have no data in this study."),
    html.Hr(),
    html.H5("Who carries the effect? Small firms"),
    html.P(HYP["H13"]["notes"][0]),
    dbc.Row([dbc.Col([html.H6("Development → ROA by company size"), dcc.Graph(figure=_size_fig, config=CFG),
                      cap("Bars show how much ROA changes per +1 SD of development, for small, medium and large firms. Whiskers are the 95% range. The effect shrinks as firms get bigger.")], md=6),
             dbc.Col([html.H6("Median ROA (pp): size × country development"), dcc.Graph(figure=_heat, config=CFG),
                      cap("Median ROA by firm size (rows) and country development (columns). Brighter = higher return.")], md=6)]),
    html.Small(HYP["H13"]["notes"][2] + " Exploratory: this refinement grew out of looking at H0.", className="text-muted"),
    html.Hr(),
    html.H6("The primary test (H0)"), html.P(HYP["H0"]["notes"][0]), stats_toggle(HYP["H0"]["notes"][1]),
])

# ---------------------------------------------------------------- countries (single view or side-by-side)
layout_countries = html.Div([
    html.H5("Countries: look at one, or compare up to three"),
    dbc.Row([dbc.Col([dbc.Label("Countries (max 3)"), dcc.Dropdown(sorted(country.name), ["Germany", "Poland"], id="countries", multi=True)], md=6),
             dbc.Col([dbc.Label("Indicator theme"), dcc.Dropdown(["all"] + GROUPS, "governance", id="theme", clearable=False)], md=3),
             dbc.Col([dbc.Label("Sector"), dcc.Dropdown(SECTOR_OPTIONS, "all", id="c-sector", clearable=False)], md=3)], className="mb-3"),
    dbc.Label("Fiscal years"), dcc.RangeSlider(YEARS[0], YEARS[1], 1, value=list(YEARS), marks={y: str(y) for y in range(YEARS[0], YEARS[1] + 1)}, id="years", className="mb-4"),
    html.Div(id="country-cards"), dcc.Graph(id="country-bars", config=CFG),
    cap("Bars show how far each indicator sits from the average of the 34 countries, in standard deviations. Right = above average, left = below."),
    html.H6("Median firm return (ROA) by year"), dcc.Graph(id="country-roa", config=CFG),
    cap("Median return on assets of listed firms in each country and fiscal year."),
])

# ---------------------------------------------------------------- method
def pipeline():
    steps = [("Sources", "World Bank · IMF · Eurostat · Yahoo Finance · ESEF"), ("PostgreSQL", "staging → core → mart"), ("DuckDB", "one shared fact table"),
             ("Analysis", "17 hypotheses · wild cluster bootstrap · BH correction"), ("Dashboard", "this app · Dash + Plotly")]
    out = []
    for i, (t, sub) in enumerate(steps):
        out.append(html.Div([html.B(t), html.Small(sub, className="text-muted d-block")], className="pipe-step"))
        if i < len(steps) - 1:
            out.append(html.Div("→", className="pipe-arrow"))
    return html.Div(out, className="pipeline")


layout_about = html.Div([
    html.H5("About this project"),
    html.P(f"I'm {AUTHOR}. I wanted to know whether the economic and social conditions of a country show up in how its listed companies perform, and to answer it "
           "with a pipeline I could defend end to end rather than a one-off chart."),
    html.H6("What I built"),
    html.Ul([html.Li("A data pipeline that pulls 35 country indicators and 10,000+ company-years from five sources, lands them in PostgreSQL and exports a DuckDB analysis database."),
             html.Li("A structured test of 17 hypotheses, with a wild cluster bootstrap (only 34 countries) and Benjamini–Hochberg correction for multiple testing."),
             html.Li("A replication on a second, independent filing source (ESEF) and a leave-one-country-out check."),
             html.Li("This dashboard, which is built to be read by people who don't do statistics.")]),
    html.H6("What I would highlight"),
    html.Ul([html.Li("Honest reporting: exploratory findings are labelled, null results are listed, caveats are on the page."),
             html.Li("Tools: Python, pandas, DuckDB, PostgreSQL, statsmodels, Plotly Dash, Docker.")]),
    html.P([html.A("GitHub profile", href=GITHUB, target="_blank"), " · ", html.A("project repository", href=REPO, target="_blank")]),
])

layout_method = html.Div([
    html.H6("How the pieces fit"), pipeline(), cap("From raw data to what you see on screen."),
    dbc.Accordion([
        dbc.AccordionItem(html.Ul([
            html.Li("Sources: World Bank WDI and WGI, IMF DataMapper, Eurostat, Yahoo Finance (yfinance, unofficial; annual accounts and monthly prices), ESEF/xBRL filings (EU)."),
            html.Li("Storage: PostgreSQL (staging / core / mart) exported to DuckDB; one shared company financial fact table."),
            html.Li("Method: lagged (t-1) country features vs firm ROA, clustered by country; wild cluster bootstrap; Benjamini–Hochberg across the primary tests; ESEF replication."),
            html.Li("Dropped: BGR, HRV, CYP, SVN, SVK (no firm data) and Malta (1 firm).")]), title="What was built"),
        dbc.AccordionItem(html.Ul([
            html.Li("Small samples for EST, LVA, LTU, CZE, IRL (markets too small)."),
            html.Li("Listed firms only; yfinance is unofficial and sector-skewed; ESEF has no sector."),
            html.Li("34 countries limit what can be separated: development, institutions and openness move together."),
            html.Li("H1, H2, H3, H5 and H13 were suggested by looking at the same data; treat as exploratory."),
            html.Li("Yahoo and ESEF agree almost perfectly (H15), so ESEF replications are not independent for the companies in both sources."),
            html.Li("Eurostat business statistics change definition between 2020 and 2021 (levels jump about 33%). The two tables are stored as separate series and every analysis uses only 2021 onward, so no result spans the break.")]), title="Caveats"),
        dbc.AccordionItem([
            dcc.Dropdown([{"label": f'{h}: {HYP[h]["title"]}' + ("" if h in HOLDS else "  (no clear evidence)"), "value": h} for h in HOLDS + REST], "H0", id="hyp", clearable=False, className="mb-3"),
            html.Div(id="hyp-detail")], title="Every hypothesis in detail"),
    ], start_collapsed=False, always_open=True),
])

# ---------------------------------------------------------------- app shell
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap"], title="Country conditions and company performance")
server = app.server
app.layout = dbc.Container([
    html.Div([html.H1(["Country conditions and ", html.Span("company performance")]),
              html.P(["34 countries, 10,000+ firm-years. ", html.A(f"by {AUTHOR}", href=GITHUB, target="_blank"), " · ", html.A("code on GitHub", href=REPO, target="_blank")],
                     className="text-muted mb-0")], className="hero"),
    dbc.Tabs([dbc.Tab(layout_overview, label="Overview", tab_id="overview"), dbc.Tab(layout_explore, label="Explore", tab_id="explore"),
              dbc.Tab(layout_countries, label="Countries", tab_id="countries"), dbc.Tab(layout_method, label="Method & caveats", tab_id="method"),
              dbc.Tab(layout_about, label="About", tab_id="about")],
             active_tab="overview", className="mb-3"),
    html.Footer([f"Built by {AUTHOR} · ", html.A("GitHub", href=GITHUB, target="_blank"), " · ", html.A("source & data pipeline", href=REPO, target="_blank")]),
], fluid=True, className="pb-5", style={"maxWidth": "1400px"})


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
    fig = px.scatter(d, x=x, y=y, color=colour, text="iso2", hover_name="name", size="firms", size_max=28, template=TPL, hover_data={"firms": True, "iso2": False})
    if len(d) > 3:
        b = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=MUTED, dash="dash"), name="fit", showlegend=False))
    fig.update_traces(textposition="top center", textfont_color=TEXT, selector=dict(mode="markers+text"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=pretty(INDICES.get(x, x)), yaxis_title=OUTCOMES.get(y, y),
                      legend=dict(orientation="h", y=-0.18, title=None))
    rho = d[x].corr(d[y], method="spearman")
    return fig, f"Spearman ρ = {rho:+.2f} across {len(d)} countries (descriptive; bubble size = number of firms)."


@callback(Output("dl", "data"), Input("dl-btn", "n_clicks"), Input("sector", "value"), prevent_initial_call=True)
def download(n, sector):
    if ctx.triggered_id != "dl-btn":
        return None
    cols = ["name", "iso2", "region", "income_group", "eu_group", "firms", "roa", "margin", "loss", "dev_index", "inst_index"] + FEATS
    return dcc.send_data_frame(country_stats(sector)[cols].round(3).to_csv, f"gbc_country_table_{sector}.csv")


@callback(Output("map", "figure"), Input("map-metric", "value"), Input("map-scope", "value"))
def draw_map(metric, scope):
    d = country.reset_index()
    label = {**OUTCOMES, **INDICES}.get(metric, pretty(metric))
    fig = px.choropleth(d, locations="iso3", color=metric, hover_name="name", color_continuous_scale=RED_SCALE, template=TPL, scope=scope, labels={metric: label})
    fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#1a1a1e", showcountries=True, countrycolor=LINE, showframe=False, showcoastlines=False, showocean=True, oceancolor=BG,
                    **({"lonaxis_range": [-25, 45], "lataxis_range": [33, 72]} if scope == "europe" else {}))
    fig.update_layout(height=520 if scope == "europe" else 440, margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title=label, thickness=12))
    return fig


@callback(Output("countries", "value"), Input("countries", "value"))
def cap_countries(v):
    return (v or ["Germany"])[:3]


@callback(Output("country-cards", "children"), Output("country-bars", "figure"), Output("country-roa", "figure"), Input("countries", "value"), Input("theme", "value"),
          Input("c-sector", "value"), Input("years", "value"))
def country_view(names, theme, sector, years):
    names = (names or ["Germany"])[:3]
    cs = country_stats(sector, tuple(years))
    names = [n for n in names if n in set(cs.name)] or names[:1]
    rows = [(cs if n in set(cs.name) else country)[(cs if n in set(cs.name) else country).name == n].iloc[0] for n in names]
    cards = dbc.Row([dbc.Col([html.H5(r["name"], className="mb-2"), dbc.Row([
        dbc.Col(xs=6, md=True, children=kpi("Firms", f'{int(r.firms)}', "small sample" if r.small_firm_sample else "")), dbc.Col(xs=6, md=True, children=kpi("Mean ROA", f"{r.roa:.1f} pp")),
        dbc.Col(xs=6, md=True, children=kpi("Development", f"{r.dev_index:+.2f} SD")), dbc.Col(xs=6, md=True, children=kpi("Institutions", f"{r.inst_index:+.2f} SD")), dbc.Col(xs=6, md=True, children=kpi("Loss-making", f"{r.loss:.0f}%"))],
        className="g-2")], md=12 // len(names)) for r in rows], className="g-3 mb-3")
    cols = [f for f in FEATS if theme == "all" or X["group"][f] == theme]
    iso = [r.name for r in rows]
    z = Z.loc[iso, cols].T.dropna(how="all")
    z.columns = names
    z = z.sort_values(names[0])
    bars = px.bar(z.reset_index().melt(id_vars="index", var_name="country", value_name="sd"), x="sd", y="index", color="country", orientation="h", barmode="group", template=TPL,
                  labels={"sd": "standard deviations from the 34-country average", "index": ""}, color_discrete_sequence=[REDHI, "#d9d4cf", AMBER])
    bars.update_layout(height=max(380, 22 * len(z) * len(names)) + 40, margin=dict(l=0, r=0, t=10, b=50), yaxis_title=None)
    bars.update_yaxes(tickvals=list(z.index), ticktext=[pretty(i) for i in z.index])
    f = firm_slice(sector, tuple(years))
    f = f[f.iso3.isin(iso)].groupby(["iso3", "fiscal_year"]).roa_pp.agg(["median", "count"]).reset_index()
    f["country"] = f.iso3.map(dict(zip(iso, names)))
    roa = px.line(f, x="fiscal_year", y="median", color="country", markers=True, hover_data=["count"], template=TPL, labels={"median": "median ROA (pp)", "fiscal_year": ""},
                  color_discrete_sequence=[REDHI, "#d9d4cf", AMBER])
    roa.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    roa.update_xaxes(dtick=1)
    bars.update_layout(legend=dict(orientation="h", y=-0.08, title=None))
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
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8050)), debug=False)
