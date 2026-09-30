"""journal version: is the 7-April discontinuity a servicer-wide rule or a local-market phenomenon? First stage by foreclosure regime, census region, leave-one-state-out."""
import os, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "v"); os.makedirs(TEX, exist_ok=True)
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
if "state" not in D.columns: D = D.join(L[["state"]])
D["state"] = D.state.astype(str).str.strip().str.upper()
JUD = set("CT DE FL HI IL IN IA KS KY LA ME NJ NM NY ND OH OK PA SC SD VT WI".split())   # judicial-foreclosure states (Mian, Sufi, Trebbi 2015 classification)
REG = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD", "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX", "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
R = {s: k for k, v in REG.items() for s in v.split()}; D["region"] = D.state.map(R); D["jud"] = D.state.isin(JUD).astype(int)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def ll(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = 1 - x.r.abs() / (h + 1)
    m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], len(x)
def dm(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]); m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], len(x)
CV = D[(D.Gov == 0) & D.state.ne("NAN") & D.state.ne("NONE")]; W = CV[CV.r.abs() <= 14]; M = {}
M["geoStates"] = f"{W.state.nunique()}"; vc = W.state.value_counts(normalize=True); M["geoTopShare"] = f"{100*vc.iloc[0]:.0f}"; M["geoTopFive"] = f"{100*vc.iloc[:5].sum():.0f}"; print(vc.head(8))
rows = []
def add(lab, d):
    a = ll(d, "fb"); b = dm(d, "dq_sep20"); c = dm(d, "mod_post")
    rows.append([lab, f"{a[0]:.1f}{star(a[2])}", f"({a[1]:.1f})", f"{b[0]:.1f}{star(b[2])}", f"({b[1]:.1f})", f"{c[0]:.1f}{star(c[2])}", f"({c[1]:.1f})", f"{a[3]:,}"]); return a, b, c
a, b, c = add("All conventional requests", CV)
for k, lab, nm in [(1, "Judicial-foreclosure states", "Jud"), (0, "Non-judicial states", "Non")]:
    a, b, c = add(lab, CV[CV.jud == k]); M["fs" + nm] = f"{a[0]:.1f}"; M["fs" + nm + "SE"] = f"{a[1]:.1f}"; M["dq" + nm] = f"{b[0]:.1f}"; M["n" + nm] = f"{a[3]:,}"
rows.append("MID"); fsr = []
for k in REG:
    d = CV[CV.region == k]
    if (d.r.abs() <= 14).sum() >= 80: a, b, c = add(k, d); fsr.append(a[0])
M["fsRegMin"] = f"{max(fsr):.1f}"; M["fsRegMax"] = f"{min(fsr):.1f}"   # jumps are negative: 'Min' = smallest in magnitude
x = CV[CV.r.abs() <= 14].copy(); x["w"] = 1 - x.r.abs() / 15
m = smf.wls("fb ~ post*jud + r*jud + post:r + post:r:jud", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); M["fsJudDiff"] = f"{100*m.params['post:jud']:.1f}"; M["fsJudDiffSE"] = f"{100*m.bse['post:jud']:.1f}"
loo = [ll(CV[CV.state != s], "fb")[0] for s in vc.index[:15]]; M["looMin"] = f"{max(loo):.1f}"; M["looMax"] = f"{min(loo):.1f}"
# state fixed effects in the first stage
x["st"] = x.state; m2 = smf.wls("fb ~ post + r + post:r + C(st)", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); M["fsStateFE"] = f"{100*m2.params['post']:.1f}"; M["fsStateFESE"] = f"{100*m2.bse['post']:.1f}"
s = ["\\begin{tabular}{lccccccc}", "\\toprule", " & \\multicolumn{2}{c}{Forbearance granted} & \\multicolumn{2}{c}{Delinquent Sep 2020} & \\multicolumn{2}{c}{Modified after Apr 2020} & \\\\", " & Jump & s.e. & Diff. & s.e. & Diff. & s.e. & Requests \\\\", "\\midrule"]
for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
open(os.path.join(TEX, "tJ_geo.tex"), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
open(os.path.join(TEX, "numbers_v.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for r in rows: print(r)
for k, v in M.items(): print(k, "=", v)


