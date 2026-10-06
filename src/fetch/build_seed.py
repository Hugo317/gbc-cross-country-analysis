"""Build config/companies_seed.csv: up to 20 listed companies per country, spread across sectors.

1. CANDIDATES: hand-written Yahoo tickers per country (more than needed, from memory).
2. Each is checked against yfinance .info (resolves? headquarters country matches? sector?).
   Results are cached in data/raw/yfinance_info/ so reruns are cheap.
3. Per country, keep matching tickers and pick up to PER_COUNTRY by round-robin over
   sectors (largest market cap first within each sector).

Outputs: data/profile/seed_validation.csv (every candidate + status) and config/companies_seed.csv.
"""
import json
import pathlib
import time

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parents[2]
INFO_DIR = ROOT / "data" / "raw" / "yfinance_info"
PER_COUNTRY = 100
SECTORS = ['Energy', 'Utilities', 'Consumer Cyclical', 'Consumer Defensive', 'Technology', 'Basic Materials',
           'Communication Services', 'Financial Services', 'Industrials', 'Real Estate', 'Healthcare']
# venues full of foreign cross-listings: look deeper so enough local firms survive validation
DEEP = {'DEU', 'AUT', 'ITA', 'FRA', 'NLD', 'ESP', 'BEL', 'LUX', 'IRL', 'PRT', 'GRC', 'POL', 'CZE', 'HUN'}

_C = {
    "AUT": "EBS.VI VER.VI OMV.VI VOE.VI ANDR.VI RBI.VI WIE.VI BG.VI UQA.VI VIG.VI TKA.VI LNZ.VI AMS.VI CAI.VI DOC.VI EVN.VI MMK.VI POST.VI AGR.VI FLU.VI PAL.VI SBO.VI IIA.VI SPI.VI",
    "BEL": "ABI.BR UCB.BR KBC.BR SYENS.BR GBLB.BR AGS.BR UMI.BR PROX.BR COLR.BR ELI.BR WDP.BR BEKB.BR MELE.BR ARGX.BR DIE.BR SOF.BR BPOST.BR AED.BR ONTEX.BR GLPG.BR COFB.BR ACKB.BR SOLB.BR TESB.BR EVS.BR BAR.BR FAGR.BR KIN.BR",
    "BGR": "SFA.SO EUBG.SO ALB.SO CHIM.SO",
    "HRV": "HT.ZA ZABA.ZA PODR.ZA ATGR.ZA ADPL.ZA KOEI.ZA ERNT.ZA RIVP.ZA ARNT.ZA LKRI.ZA KRAS.ZA DLKV.ZA",
    "CYP": "BOCH.AT BOCY.L",
    "CZE": "CEZ.PR KOMB.PR MONET.PR COLT.PR KOFOL.PR TABAK.PR PMCR.PR",
    "DNK": "NOVO-B.CO DSV.CO MAERSK-B.CO VWS.CO ORSTED.CO CARL-B.CO DANSKE.CO COLO-B.CO GMAB.CO PNDORA.CO NZYM-B.CO TRYG.CO ISS.CO DEMANT.CO GN.CO FLS.CO ROCK-B.CO AMBU-B.CO RBREW.CO JYSK.CO NKT.CO BAVA.CO NETC.CO ZEAL.CO SYDB.CO ALMB.CO",
    "EST": "TKM1T.TL TAL1T.TL MRK1T.TL CPA1T.TL LHV1T.TL EEE1T.TL HAE1T.TL SFG1T.TL TVEAT.TL NCN1T.TL ARC1T.TL BLT1T.TL PKG1T.TL INF1T.TL EGR1T.TL",
    "FIN": "NOKIA.HE KNEBV.HE NESTE.HE SAMPO.HE FORTUM.HE UPM.HE STERV.HE KESKOB.HE ELISA.HE WRT1V.HE METSO.HE KCR.HE ORNBV.HE TIE1V.HE KEMIRA.HE VALMT.HE HUH1V.HE FSKRS.HE NDA-FI.HE OUT1V.HE QTCOM.HE SSABB.HE CGCBV.HE YIT.HE FIA1S.HE SANOMA.HE TELIA1.HE KOJAMO.HE",
    "FRA": "MC.PA OR.PA RMS.PA TTE.PA SAN.PA AIR.PA SU.PA AI.PA BNP.PA CS.PA DG.PA EL.PA SAF.PA KER.PA CAP.PA ORA.PA RI.PA ENGI.PA VIE.PA GLE.PA ACA.PA STMPA.PA DSY.PA PUB.PA EN.PA ML.PA CA.PA BN.PA RNO.PA HO.PA SGO.PA LR.PA VIV.PA TEP.PA URW.PA SW.PA ALO.PA AC.PA FR.PA",
    "DEU": "SAP.DE SIE.DE ALV.DE MUV2.DE DTE.DE BAS.DE BAYN.DE BMW.DE MBG.DE VOW3.DE ADS.DE IFX.DE DB1.DE DBK.DE EOAN.DE RWE.DE HEI.DE HEN3.DE FRE.DE FME.DE MRK.DE SHL.DE CON.DE DHL.DE LHA.DE ZAL.DE BEI.DE SY1.DE 1COV.DE RHM.DE AIR.DE PUM.DE QIA.DE VNA.DE SRT3.DE HNR1.DE",
    "GRC": "OPAP.AT EUROB.AT ETE.AT ALPHA.AT TPEIR.AT PPC.AT MYTIL.AT TITC.AT HTO.AT MOH.AT ELPE.AT GEKTERNA.AT JUMBO.AT AEGN.AT LAMDA.AT BELA.AT EYDAP.AT ADMIE.AT INTRK.AT ATTICA.AT",
    "HUN": "OTP.BD MOL.BD RICHTER.BD MTELEKOM.BD ANY.BD AUTOWALLIS.BD APPENINN.BD MASTERPLAST.BD RABA.BD PANNERGY.BD",
    "IRL": "RYA.IR KRZ.IR GLB.L DCC.L GNC.L GFTU.L EXPN.L KWS.L HSW.L FLTR.L UPR.IR GVR.IR MCON.IR IR5B.IR PTSB.IR JCI CMPR PRTA AMRN HZNP ENDP ADNT AVDL TRIB NBRV HTOO ITRM AER ARGX ALKS RYAAY AIBRY IRBT NVT PNR STE AMBP SPOT KRX.IR A5G.IR BIRG.IR GLB.IR DLG.IR IRES.IR CRN.IR OIZ.IR FBH.IR CRH FLUT MDT ACN ETN LIN TT AON STX JAZZ PRGO ICLR ALLE STE",
    "ITA": "ENEL.MI ISP.MI UCG.MI ENI.MI G.MI RACE.MI STMMI.MI TIT.MI LDO.MI PRY.MI MONC.MI CPR.MI BAMI.MI BPE.MI MB.MI PST.MI SRG.MI TRN.MI AMP.MI REC.MI DIA.MI NEXI.MI BMED.MI FBK.MI HER.MI IP.MI IG.MI A2A.MI ERG.MI",
    "LVA": "ELEVR.RG IDX1R.RG DGR1R.RG VIRSI.RG",
    "LTU": "",
    "LUX": "MT.AS TEN.MI APAM.AS SESG.PA RRTL.DE ERF.PA SPOT GLOB TIGO TX SUBC.OL",
    "MLT": "KIND-SDB.ST CTM.ST",
    "NLD": "ASML.AS SHELL.AS PRX.AS HEIA.AS INGA.AS ADYEN.AS WKL.AS AD.AS PHIA.AS AKZA.AS NN.AS ASM.AS BESI.AS IMCD.AS DSFIR.AS RAND.AS ABN.AS FUR.AS ASRNL.AS LIGHT.AS AALB.AS SBMO.AS KPN.AS FLOW.AS UMG.AS EXO.AS",
    "POL": "PKN.WA PKO.WA PZU.WA PEO.WA KGH.WA DNP.WA LPP.WA CDR.WA ALE.WA SPL.WA MBK.WA PGE.WA OPL.WA CPS.WA TPE.WA JSW.WA ALR.WA KRU.WA ACP.WA PCO.WA ENA.WA 11B.WA XTB.WA BDX.WA KTY.WA BFT.WA DOM.WA",
    "PRT": "EDP.LS EDPR.LS GALP.LS JMT.LS BCP.LS NOS.LS SON.LS NVG.LS RENE.LS ALTR.LS CTT.LS SEM.LS COR.LS EGL.LS IBS.LS NBA.LS",
    "ROU": "TLV.RO SNP.RO H2O.RO SNG.RO BRD.RO TGN.RO SNN.RO EL.RO DIGI.RO M.RO TEL.RO ONE.RO WINE.RO COTE.RO FP.RO TRP.RO SFG.RO BVB.RO AQ.RO EVER.RO CMF.RO ALR.RO PE.RO IMP.RO",
    "SVK": "",
    "SVN": "KRKG.LJ PETG.LJ ZVTG.LJ NLBR.LJ TLSG.LJ LKPG.LJ SAVA.LJ POSR.LJ",
    "ESP": "ITX.MC SAN.MC BBVA.MC IBE.MC TEF.MC REP.MC AMS.MC FER.MC CABK.MC AENA.MC ELE.MC ENG.MC NTGY.MC GRF.MC ACS.MC SAB.MC BKT.MC MAP.MC RED.MC CLNX.MC IAG.MC COL.MC MRL.MC ANA.MC ACX.MC MTS.MC FDR.MC LOG.MC PHM.MC SCYR.MC IDR.MC ROVI.MC",
    "SWE": "VOLV-B.ST ERIC-B.ST ATCO-A.ST INVE-B.ST HEXA-B.ST SEB-A.ST SHB-A.ST SWED-A.ST ESSITY-B.ST SAND.ST ALFA.ST ASSA-B.ST HM-B.ST SKF-B.ST TELIA.ST EVO.ST NIBE-B.ST BOL.ST EQT.ST SINCH.ST GETI-B.ST KINV-B.ST SCA-B.ST ELUX-B.ST TEL2-B.ST LIFCO-B.ST BALD-B.ST SSAB-A.ST SAAB-B.ST SECU-B.ST HUSQ-B.ST NDA-SE.ST LATO-B.ST EPI-A.ST INDT.ST CAST.ST TREL-B.ST SBB-B.ST",
    "USA": "AAPL MSFT NVDA JPM XOM JNJ UNH PG KO HD CAT NEE AMT DIS T NFLX WMT GE PFE CVX V MA COST MRK BA GS AMZN LLY DUK SPG FCX NKE UPS VZ",
    "CAN": "RY.TO TD.TO ENB.TO SHOP.TO CNQ.TO CP.TO CNR.TO BMO.TO BNS.TO SU.TO NTR.TO B.TO TRP.TO MFC.TO SLF.TO ATD.TO CSU.TO BCE.TO T.TO L.TO WCN.TO FTS.TO AEM.TO TRI.TO GIB-A.TO DOL.TO QSR.TO MG.TO WPM.TO CCO.TO IFC.TO CVE.TO TECK-B.TO NA.TO CM.TO FNV.TO K.TO WSP.TO OTEX.TO GFL.TO",
    "MEX": "AMXB.MX WALMEX.MX FEMSAUBD.MX GFNORTEO.MX GMEXICOB.MX CEMEXCPO.MX BIMBOA.MX GAPB.MX ASURB.MX OMAB.MX KOFUBL.MX ELEKTRA.MX ALFAA.MX GRUMAB.MX AC.MX ORBIA.MX PINFRA.MX KIMBERA.MX LABB.MX MEGACPO.MX BBAJIOO.MX CUERVO.MX TLEVISACPO.MX GCARSOA1.MX RA.MX BOLSAA.MX CHDRAUIB.MX VESTA.MX GCC.MX",
    "BRA": "VALE3.SA PETR4.SA ITUB4.SA BBDC4.SA ABEV3.SA B3SA3.SA WEGE3.SA RENT3.SA SUZB3.SA BBAS3.SA ELET3.SA JBSS3.SA RADL3.SA RAIL3.SA EQTL3.SA SBSP3.SA VIVT3.SA LREN3.SA GGBR4.SA CSNA3.SA HAPV3.SA PRIO3.SA BPAC11.SA KLBN11.SA EMBR3.SA CMIG4.SA TOTS3.SA CCRO3.SA MGLU3.SA BRFS3.SA UGPA3.SA ASAI3.SA RDOR3.SA TIMS3.SA CPLE6.SA",
    "CHL": "SQM-B.SN FALABELLA.SN COPEC.SN CENCOSUD.SN BCI.SN CHILE.SN ENELAM.SN ENELCHILE.SN COLBUN.SN CMPC.SN LTM.SN CCU.SN CONCHATORO.SN ANDINA-B.SN ECL.SN AGUAS-A.SN SONDA.SN PARAUCO.SN MALLPLAZA.SN ITAUCL.SN SMU.SN RIPLEY.SN VAPORES.SN QUINENCO.SN CAP.SN ENTEL.SN IAM.SN SALFACORP.SN",
    "JPN": "7203.T 6758.T 9984.T 8306.T 6861.T 9432.T 8035.T 6098.T 4063.T 7974.T 9983.T 8058.T 8001.T 8031.T 4502.T 4568.T 6501.T 6367.T 7267.T 2914.T 4519.T 8316.T 9433.T 7741.T 6954.T 5401.T 8801.T 9020.T 9501.T 2802.T 1925.T 4901.T 5108.T 9101.T 7751.T 6902.T 3382.T 5020.T",
    "KOR": "005930.KS 000660.KS 373220.KS 207940.KS 005380.KS 000270.KS 005490.KS 035420.KS 051910.KS 105560.KS 055550.KS 012330.KS 028260.KS 068270.KS 066570.KS 096770.KS 015760.KS 017670.KS 030200.KS 009150.KS 003550.KS 034730.KS 086790.KS 033780.KS 051900.KS 090430.KS 011200.KS 003670.KS 006400.KS 035720.KS",
    "CHN": "600519.SS 601318.SS 601398.SS 600036.SS 601288.SS 601857.SS 600900.SS 601088.SS 600028.SS 600276.SS 601899.SS 600887.SS 600030.SS 601012.SS 600309.SS 000858.SZ 300750.SZ 000333.SZ 002594.SZ 000651.SZ 002415.SZ 000001.SZ 300059.SZ 002714.SZ 000725.SZ 601166.SS 600104.SS 601668.SS 601888.SS 600690.SS",
    "IND": "RELIANCE.NS TCS.NS HDFCBANK.NS ICICIBANK.NS INFY.NS HINDUNILVR.NS ITC.NS SBIN.NS BHARTIARTL.NS LT.NS KOTAKBANK.NS AXISBANK.NS ASIANPAINT.NS MARUTI.NS SUNPHARMA.NS TITAN.NS NTPC.NS ONGC.NS POWERGRID.NS ULTRACEMCO.NS TATASTEEL.NS M&M.NS BAJFINANCE.NS WIPRO.NS NESTLEIND.NS COALINDIA.NS ADANIPORTS.NS DRREDDY.NS DLF.NS HCLTECH.NS JSWSTEEL.NS HINDALCO.NS CIPLA.NS BPCL.NS",
    "SGP": "D05.SI O39.SI U11.SI Z74.SI C6L.SI C07.SI F34.SI BN4.SI A17U.SI C38U.SI G13.SI S68.SI U96.SI S63.SI V03.SI Y92.SI BS6.SI H78.SI N2IU.SI ME8U.SI C09.SI U14.SI 9CI.SI S58.SI CC3.SI",
    "IDN": "BBCA.JK BBRI.JK BMRI.JK BBNI.JK TLKM.JK ASII.JK UNVR.JK ICBP.JK INDF.JK GGRM.JK HMSP.JK ADRO.JK PTBA.JK ANTM.JK TPIA.JK BRPT.JK AMRT.JK CPIN.JK KLBF.JK MDKA.JK SMGR.JK INTP.JK UNTR.JK PGAS.JK EXCL.JK ISAT.JK MAPI.JK ACES.JK BSDE.JK CTRA.JK",
    "AUS": "BHP.AX CBA.AX CSL.AX NAB.AX WBC.AX ANZ.AX MQG.AX WES.AX WOW.AX FMG.AX TLS.AX RIO.AX GMG.AX TCL.AX WDS.AX STO.AX ALL.AX COL.AX QBE.AX SUN.AX IAG.AX ORG.AX AGL.AX REA.AX XRO.AX RMD.AX SHL.AX COH.AX S32.AX NST.AX EVN.AX BXB.AX QAN.AX MIN.AX APA.AX",
    "ZAF": "NPN.JO FSR.JO SBK.JO SOL.JO AGL.JO MTN.JO BTI.JO GFI.JO ANG.JO SHP.JO BID.JO VOD.JO NED.JO ABG.JO CPI.JO DSY.JO MNP.JO SLM.JO REM.JO IMP.JO SSW.JO BVT.JO WHL.JO PIK.JO TBS.JO CLS.JO APN.JO OMU.JO NTC.JO GRT.JO EXX.JO",
}
CANDIDATES = {iso: t.split() for iso, t in _C.items()}

# Yahoo's `country` value -> our country name, where they differ
ALIASES = {"Czech Republic": "Czechia", "Korea (South)": "South Korea", "Republic of Korea": "South Korea"}


# deeper screener pass where foreign cross-listings crowd out local firms
DEEPER = {"USA": 100, "CAN": 100, "MEX": 100, "BRA": 100, "DEU": 100, "ITA": 100, "AUT": 100,
          "IRL": 100, "LUX": 100, "CHL": 60, "NLD": 80, "PRT": 80, "CZE": 80, "HUN": 80}


def is_depositary_receipt(sym):
    """B3 lists foreign companies as BDRs with two-digit codes like AAPL34.SA (real stocks end 3-8 or 11)."""
    import re
    return bool(re.search(r"\d{2}\.SA$", sym)) and not sym.endswith("11.SA")


def screener_extras(countries):
    """Yahoo screener candidates per country: top firms by market cap in each sector.

    The screener filters by listing region, not headquarters, so foreign cross-listings
    leak in; get_info() + the country check below remove them.
    """
    from yfinance import EquityQuery
    iso2 = pd.read_csv(ROOT / "config" / "countries.csv").set_index("iso3")["iso2"].str.lower().to_dict()
    extra = {}
    for iso3 in countries:
        size = DEEPER.get(iso3, 40 if iso3 in DEEP else 15)
        found = []
        for sector in SECTORS:
            try:
                q = EquityQuery("and", [EquityQuery("eq", ["region", iso2[iso3]]),
                                        EquityQuery("eq", ["sector", sector])])
                r = yf.screen(q, size=size, sortField="intradaymarketcap", sortAsc=False)
                found += [x["symbol"] for x in r.get("quotes", []) if not is_depositary_receipt(x["symbol"])]
            except Exception:
                break   # region unsupported by the screener
            time.sleep(0.3)
        extra[iso3] = found
        print(f"screener {iso3}: {len(found)} candidates", flush=True)
    return extra


def get_info(ticker):
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    cache = INFO_DIR / f"{ticker.replace('/', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    keep = ["longName", "shortName", "country", "sector", "industry", "currency",
            "marketCap", "exchange", "quoteType"]
    out = {"ticker": ticker, "ok": False}
    for attempt in range(2):
        try:
            info = yf.Ticker(ticker).info
            if info and info.get("quoteType") == "EQUITY":
                out.update({k: info.get(k) for k in keep}, ok=True)
            out.pop("error", None)
            break
        except Exception as exc:
            out["error"] = str(exc)[:100]
            time.sleep(3)
    if "error" not in out:
        cache.write_text(json.dumps(out))
    time.sleep(0.4)
    return out


def main():
    countries = pd.read_csv(ROOT / "config" / "countries.csv").set_index("iso3")["name"].to_dict()
    cands = {k: list(v) for k, v in CANDIDATES.items() if k in countries}
    for iso3, syms in screener_extras(list(cands)).items():
        cands[iso3] = list(dict.fromkeys(cands.get(iso3, []) + syms))

    rows = []
    for iso3, tickers in cands.items():
        for t in tickers:
            info = get_info(t)
            ctry = ALIASES.get(info.get("country"), info.get("country"))
            status = ("no_data" if not info["ok"]
                      else "ok" if ctry == countries[iso3] else "wrong_country")
            rows.append({"iso3": iso3, "ticker": t, "status": status, "name": info.get("longName") or info.get("shortName"),
                         "yahoo_country": info.get("country"), "sector": info.get("sector"),
                         "industry": info.get("industry"), "currency": info.get("currency"),
                         "market_cap": info.get("marketCap")})
        n_ok = sum(r["status"] == "ok" for r in rows if r["iso3"] == iso3)
        print(f"{iso3}: {len(tickers)} candidates, {n_ok} valid", flush=True)

    val = pd.DataFrame(rows)
    out = ROOT / "data" / "profile" / "seed_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(out, index=False)

    picked = []
    is_fund = val.name.fillna("").str.contains(r"\b(?:Fund|Trust|ETF)\b", case=False, regex=True)
    for iso3, grp in val[(val.status == "ok") & ~is_fund].groupby("iso3"):
        grp = grp.assign(sector=grp.sector.fillna("Unknown")).sort_values("market_cap", ascending=False)
        queues = [g.to_dict("records") for _, g in grp.groupby("sector")]
        queues.sort(key=lambda q: -(q[0]["market_cap"] or 0))   # biggest sectors first
        chosen = []
        while len(chosen) < PER_COUNTRY and any(queues):
            for q in queues:
                if q and len(chosen) < PER_COUNTRY:
                    chosen.append(q.pop(0))
        picked += chosen
    seed = pd.DataFrame(picked)[["ticker", "iso3", "name", "sector", "industry", "currency", "market_cap"]]
    seed.to_csv(ROOT / "config" / "companies_seed.csv", index=False)
    print("\nPer country (valid candidates -> seeded):")
    summary = (val.groupby("iso3").apply(lambda d: (d.status == "ok").sum()).rename("valid").to_frame()
               .join(seed.groupby("iso3").agg(seeded=("ticker", "size"), sectors=("sector", "nunique")))
               .fillna(0).astype(int))
    print(summary.to_string())


if __name__ == "__main__":
    main()
