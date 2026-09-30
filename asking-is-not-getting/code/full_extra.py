"""Full-length journal paper: neighborhood heterogeneity and balance (zip demographics), reweighted estimates, housing outcomes a year on, equity and occupancy splits.
Only loan id, zip and county codes are read from the geography file; no names or addresses."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "juefull"); os.makedirs(TEX, exist_ok=True)
B = r"<DATA>/geo"
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); D.index = D.index.astype(str); L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); L.index = L.index.astype(str); D = D.join(L[["occ"]])
import re as _re; cl_ = lambda c: _re.sub(r"^[^A-Za-z]+", "", c); g = pd.read_csv(os.path.join(B, "Geography.csv"), encoding="latin1", dtype=str, usecols=lambda c: cl_(c) in ("LoanID", "Zip", "CountyFIPS")); g.columns = [cl_(c) for c in g.columns]
g["zip5"] = g.Zip.str.extract(r"(\d{5})")[0]; g = g.dropna(subset=["LoanID"]).drop_duplicates("LoanID").set_index("LoanID")
z = pd.read_csv(os.path.join(B, "Zip_Demo_Mod.csv"), encoding="latin1", dtype=str, usecols=lambda c: c in ("ZIP5", "POPCY", "AREALAND", "HHSCY", "White_pc", "Black_pc", "Hisp_pc", "AG_HINC_C", "Blue Collar_pct", "Unemp_pc"))
z["zip5"] = z.ZIP5.str.zfill(5); num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[,%$]", "", regex=True), errors="coerce")
for c in ("POPCY", "AREALAND", "HHSCY", "White_pc", "Black_pc", "Hisp_pc", "AG_HINC_C", "Blue Collar_pct", "Unemp_pc"): z[c] = num(z[c])
z = z.drop_duplicates("zip5").set_index("zip5"); sc = lambda v: v * 100 if v.median() <= 1.5 else v
z["nonwhite"] = 100 - sc(z.White_pc); z["blue"] = sc(z["Blue Collar_pct"]); z["hinc"] = z.AG_HINC_C / z.HHSCY.where(z.HHSCY > 0); z["hinc"] = z.hinc / (1000 if z.hinc.median() > 1000 else 1); z["dens"] = z.POPCY / z.AREALAND.where(z.AREALAND > 0)
D = D.join(g[["zip5", "CountyFIPS"]]); D = D.join(z[["nonwhite", "hinc", "dens", "blue"]], on="zip5"); D["ldens"] = np.log(D.dens.where(D.dens > 0))
print("geography matched:", D.zip5.notna().mean().round(3), "| zip demographics matched:", D.nonwhite.notna().mean().round(3)); print(D[["nonwhite", "hinc", "blue"]].describe().round(1).T)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""; cl = lambda x: {"groups": pd.factorize(x.r)[0]}
def ll(x, y, h=14, wt=None):
    x = x[x.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = (1 - x.r.abs() / (h + 1)) * (x[wt] if wt else 1); m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds=cl(x)); return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], len(x)
def dm(x, y, h=14, wt=None):
    x = x[x.r.abs() <= h].dropna(subset=[y]).copy(); m = (smf.wls(f"{y} ~ post", x, weights=x[wt]) if wt else smf.ols(f"{y} ~ post", x)).fit(cov_type="cluster", cov_kwds=cl(x)); return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], len(x)
def inter(x, y, gcol, h=14):
    x = x[x.r.abs() <= h].dropna(subset=[y, gcol]).copy(); m = smf.ols(f"{y} ~ post*{gcol}", x).fit(cov_type="cluster", cov_kwds=cl(x)); return m.pvalues[f"post:{gcol}"]
f = lambda e: f"{e[0]:.1f}{star(e[2])}"; s_ = lambda e: f"({e[1]:.1f})"
def tab(name, head, rows): open(os.path.join(TEX, name), "w", encoding="utf-8").write("\n".join(["\\begin{tabular}{l" + "c" * (sum(2 if "multicolumn{2}" in h else 1 for h in head) - 1) + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"] + ["\\midrule" if r == "MID" else " & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]))
CV = D[D.Gov == 0].copy(); W = CV[CV.r.abs() <= 14]; M = {}
# ---- 1. neighborhoods: splits at the window median ----
OUTC = [("fb", "ll"), ("dq_sep20", "dm"), ("mod_post", "dm"), ("performing", "dm"), ("fc_path", "dm")]; rows = []
for col, lab_hi, lab_lo, nm in [("nonwhite", "Zip non-white share above median", "Zip non-white share below median", "Nw"), ("hinc", "Zip household income above median", "Zip household income below median", "Inc"), ("dens", "Zip population density above median", "Zip population density below median", "Den"), ("blue", "Zip blue-collar share above median", "Zip blue-collar share below median", "Blue")]:
    med = W[col].median(); CV["hi"] = (CV[col] > med).astype(float).where(CV[col].notna()); M["med" + nm] = f"{med:.0f}" if col != "dens" else f"{med:.4g}"
    for k, lab in [(1, lab_hi), (0, lab_lo)]:
        x = CV[CV.hi == k]; es = [(ll if t == "ll" else dm)(x, y) for y, t in OUTC]; rows.append([lab] + [v for e in es for v in (f(e), s_(e))] + [f"{es[0][3]:,}"])
        M[f"fs{nm}{'Hi' if k else 'Lo'}"] = f"{es[0][0]:.1f}"; M[f"fc{nm}{'Hi' if k else 'Lo'}"] = f"{es[4][0]:.1f}"; M[f"dq{nm}{'Hi' if k else 'Lo'}"] = f"{es[1][0]:.1f}"
    ps = [inter(CV, y, "hi") for y, _ in OUTC]; rows.append(["\\quad $p$-value, difference"] + [v for p in ps for v in (f"{p:.2f}", "")] + [""]); rows.append("MID"); M["p" + nm + "Fc"] = f"{ps[4]:.2f}"; M["p" + nm + "Fs"] = f"{ps[0]:.2f}"
rows.pop(); tab("tN_neigh.tex", ["", "\\multicolumn{2}{c}{Forbearance}", "\\multicolumn{2}{c}{Delinq.\\ Sep 2020}", "\\multicolumn{2}{c}{Modified}", "\\multicolumn{2}{c}{Performing Apr 2021}", "\\multicolumn{2}{c}{Foreclosure path}", "Loans"], rows)
# ---- 2. neighborhood balance at the cutoff ----
rows = []
for y, lab, sc in [("nonwhite", "Zip non-white share (\\%)", .01), ("hinc", "Zip mean household income (\\$000)", .01), ("ldens", "Log zip population density", .01), ("blue", "Zip blue-collar share (\\%)", .01)]:
    a = ll(CV, y); b = ll(D[D.Gov == 1], y); rows.append([lab, f"{W[W.post == 0][y].mean():.1f}" if y != "ldens" else f"{W[W.post == 0][y].mean():.2f}", f"{sc*a[0]:.2f}{star(a[2])}", f"({sc*a[1]:.2f})", f"{sc*b[0]:.2f}{star(b[2])}", f"({sc*b[1]:.2f})", f"{a[3]:,}"]); M["bal" + y.capitalize()[:4]] = f"{sc*a[0]:.1f}"; M["bal" + y.capitalize()[:4] + "SE"] = f"{sc*a[1]:.1f}"
tab("tN_balance.tex", ["Neighborhood characteristic", "Mean before", "Conv.\\ jump", "s.e.", "Gov.\\ jump", "s.e.", "Loans"], rows)
M["geoMatch"] = f"{100*W.nonwhite.notna().mean():.0f}"; M["nwMean"] = f"{W.nonwhite.mean():.0f}"; M["incMean"] = f"{W.hinc.mean():.0f}"
# ---- 3. reweighting on the imbalanced covariates ----
X = W.copy(); X["ficom"] = X.fico.fillna(X.fico.median()); X["ficomiss"] = X.fico.isna().astype(int)
Xc = X.dropna(subset=["ltv", "lbal", "tenure", "pre_in", "pre_out"]); cov = [c for c in ["ficom", "ficomiss", "hard", "pre_in", "pre_out", "dq_feb20", "npl", "ltv", "lbal", "tenure"] if Xc[c].astype(float).std() > 1e-8]
for c in cov: Xc[c] = Xc[c].astype(float)
Xc["ficom"] = Xc.ficom / 100; Xc["tenure"] = Xc.tenure / 12; Xc["ltv"] = Xc.ltv / 100; print("reweighting covariates:", cov)
ps = smf.glm("post ~ " + " + ".join(cov), Xc, family=sm.families.Binomial()).fit(); X = X.loc[ps.fittedvalues.index]; p = ps.fittedvalues.values; X["ipw"] = np.where(X.post == 1, 1 / p, 1 / (1 - p)); X["ipw"] = X.ipw.clip(upper=X.ipw.quantile(.99))
wb = lambda c: (np.average(X[X.post == 1][c], weights=X[X.post == 1].ipw) - np.average(X[X.post == 0][c], weights=X[X.post == 0].ipw)); print("FICO gap raw / reweighted:", round(X[X.post == 1].ficom.mean() - X[X.post == 0].ficom.mean(), 1), round(wb("ficom"), 1)); M["ipwFicoRaw"] = f"{X[X.post==1].ficom.mean()-X[X.post==0].ficom.mean():.0f}"; M["ipwFicoW"] = f"{wb('ficom'):.0f}"
rows = []
for y, lab in [("fb", "Forbearance granted"), ("dq_sep20", "Delinquent, September 2020"), ("mod_post", "Modified after April 2020"), ("performing", "Performing, April 2021"), ("fc_path", "On a path to foreclosure, April 2021")]:
    a, b, c, d_ = dm(X, y), dm(X, y, wt="ipw"), ll(X, y), ll(X, y, wt="ipw"); rows.append([lab, f(a), s_(a), f(b), s_(b), f(c), s_(c), f(d_), s_(d_)]); k_ = "ipw" + "".join(ch for ch in y.replace("_", "").capitalize() if not ch.isdigit())[:6]; M[k_] = f"{b[0]:.1f}"; M[k_ + "SE"] = f"{b[1]:.1f}"
tab("tN_ipw.tex", ["", "\\multicolumn{2}{c}{Diff.\\ in means}", "\\multicolumn{2}{c}{Reweighted}", "\\multicolumn{2}{c}{Local linear}", "\\multicolumn{2}{c}{Reweighted}"], rows)
# ---- 4. where borrowers stood in April 2021 ----
cats = [("Performing", ["Performing"]), ("Modification completed or in review", ["Modification Completed", "Modification in Review"]), ("Notice of intent filed, not in foreclosure", ["NOI Filed: Not in FC"]), ("Foreclosure pending or REO", ["Pending Foreclosure Completion", "REO"]), ("Bankruptcy", ["BK"]), ("Short payoff, deed-in-lieu or other", ["Pending Short Payoff", "Pending Deed-in-Lieu", "Rolling Delinquency"])]; rows = []
for lab, v in cats:
    D["y"] = D.disp.isin(v).astype(float).where(D.disp.notna()); a = dm(D[D.Gov == 0], "y"); b = ll(D[D.Gov == 0], "y"); c = dm(D[D.Gov == 1], "y"); x = D[(D.Gov == 0) & (D.r.abs() <= 14)]
    rows.append([lab, f"{100*x[x.post==0].y.mean():.1f}", f"{100*x[x.post==1].y.mean():.1f}", f(a), s_(a), f(b), s_(b), f(c), s_(c)])
D["y"] = D.closed.astype(float); a = dm(D[D.Gov == 0], "y"); b = ll(D[D.Gov == 0], "y"); c = dm(D[D.Gov == 1], "y"); x = D[(D.Gov == 0) & (D.r.abs() <= 14)]; rows += ["MID", ["Loan liquidated or paid off by March 2021 (remittance)", f"{100*x[x.post==0].y.mean():.1f}", f"{100*x[x.post==1].y.mean():.1f}", f(a), s_(a), f(b), s_(b), f(c), s_(c)]]; M["liqDM"] = f"{a[0]:.1f}"; M["liqDMSE"] = f"{a[1]:.1f}"
tab("tN_dispo.tex", ["Status in April 2021", "Asked before", "Asked after", "Diff.", "s.e.", "Local linear", "s.e.", "Gov.\\ diff.", "s.e."], rows)
# ---- 5. equity and occupancy ----
rows = []; CV["highltv"] = (CV.ltv > 90).astype(float).where(CV.ltv.notna()); CV["owner"] = CV.occ.eq("Owner occupied").astype(float).where(CV.occ.notna())
for col, labs, nm in [("highltv", ("Loan-to-value above 90", "Loan-to-value 90 or below"), "Ltv"), ("owner", ("Owner-occupied", "Not owner-occupied"), "Own")]:
    for k, lab in [(1, labs[0]), (0, labs[1])]:
        x = CV[CV[col] == k]; es = [(ll if t == "ll" else dm)(x, y) for y, t in OUTC]; rows.append([lab] + [v for e in es for v in (f(e), s_(e))] + [f"{es[0][3]:,}"]); M[f"fc{nm}{'Hi' if k else 'Lo'}"] = f"{es[4][0]:.1f}"; M[f"fs{nm}{'Hi' if k else 'Lo'}"] = f"{es[0][0]:.1f}"
    pv = [inter(CV, y, col) for y, _ in OUTC]; rows.append(["\\quad $p$-value, difference"] + [v for p_ in pv for v in (f"{p_:.2f}", "")] + [""]); rows.append("MID"); M["p" + nm + "Fc"] = f"{pv[4]:.2f}"
rows.pop(); tab("tN_equity.tex", ["", "\\multicolumn{2}{c}{Forbearance}", "\\multicolumn{2}{c}{Delinq.\\ Sep 2020}", "\\multicolumn{2}{c}{Modified}", "\\multicolumn{2}{c}{Performing Apr 2021}", "\\multicolumn{2}{c}{Foreclosure path}", "Loans"], rows)
M["ownShare"] = f"{100*W.occ.eq('Owner occupied').mean():.0f}"; M["ltvMed"] = f"{W.ltv.median():.0f}"
open(os.path.join(TEX, "numbers_full.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for k, v in M.items(): print(k, "=", v)
for t in ("tN_neigh.tex", "tN_balance.tex", "tN_ipw.tex", "tN_dispo.tex", "tN_equity.tex"): print("\n" + t + "\n" + open(os.path.join(TEX, t), encoding="utf-8").read())

