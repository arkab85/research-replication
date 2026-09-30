"""Spatial analysis for the journal paper.
(1) How spatially concentrated is the portfolio, and can a servicer rule have neighborhood-level bite?
(2) Continuous neighborhood interactions with a joint test and multiple-testing adjustment, replacing median splits.
(3) Pandemic labor exposure: share of zip employment in occupations that cannot be done from home.
(4) Inference: county fixed effects, two-way clustering, Conley spatial HAC.
Reads only loan id, zip, county and coordinates from the property file - never names or street addresses.
"""
import os, re, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "juefull")
B = r"<DATA>/geo"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
cl = lambda c: re.sub(r"^[^A-Za-z]+", "", c)
num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[,%$]", "", regex=True), errors="coerce")
pct = lambda v: v * 100 if v.median(skipna=True) <= 1.5 else v
M = {}

# ---------------- geography ----------------
g = pd.read_csv(os.path.join(B, "Geography.csv"), encoding="latin1", dtype=str,
                usecols=lambda c: cl(c) in ("LoanID", "Zip", "CountyFIPS", "Latitude", "Longitude"))
g.columns = [cl(c) for c in g.columns]
for c in ("Latitude", "Longitude"): g[c] = pd.to_numeric(g[c], errors="coerce")
g["zip5"] = g.Zip.str.extract(r"(\d{5})")[0]; g["cty"] = g.CountyFIPS.str.extract(r"(\d{4,5})")[0].str.zfill(5)
g = g.dropna(subset=["LoanID"]).drop_duplicates("LoanID").set_index("LoanID")[["zip5", "cty", "Latitude", "Longitude"]]

Z = pd.read_csv(os.path.join(B, "Zip_Demo_Mod.csv"), encoding="latin1", dtype=str)
Z.columns = [cl(c) for c in Z.columns]; Z["zip5"] = Z.ZIP5.str.zfill(5)
need = ["POPCY", "AREALAND", "HHSCY", "White_pc", "Black_pc", "Hisp_pc", "AG_HINC_C", "Blue Collar_pct",
        "Unemp_pc", "LF_EMP_C", "LF_UNEMP_8QAgo_PCT", "House_rent_pct", "House_Vac_pct", "High_edu_pct",
        "OCC_BC_PROT_C", "OCC_BC_FOOD_C", "OCC_BC_MAIN_C", "OCC_BC_CARE_C", "OCC_BC_FARM_C", "OCC_BC_CONS_C",
        "OCC_BC_TRAN_C", "OCC_WC_MAN_C", "OCC_WC_PROF_C", "OCC_WC_HC_C", "OCC_WC_SALE_C", "OCC_WC_ADMN_C"]
for c in need: Z[c] = num(Z[c])
Z = Z.drop_duplicates("zip5").set_index("zip5")
emp = Z[["OCC_BC_PROT_C", "OCC_BC_FOOD_C", "OCC_BC_MAIN_C", "OCC_BC_CARE_C", "OCC_BC_FARM_C", "OCC_BC_CONS_C",
         "OCC_BC_TRAN_C", "OCC_WC_MAN_C", "OCC_WC_PROF_C", "OCC_WC_HC_C", "OCC_WC_SALE_C", "OCC_WC_ADMN_C"]].sum(axis=1)
tele = Z[["OCC_WC_MAN_C", "OCC_WC_PROF_C", "OCC_WC_ADMN_C"]].sum(axis=1)
N = pd.DataFrame(index=Z.index)
N["frontline"] = 100 * (1 - tele / emp.where(emp > 0))              # cannot be done from home (Dingel-Neiman in spirit)
N["nonwhite"] = 100 - pct(Z.White_pc); N["blue"] = pct(Z["Blue Collar_pct"]); N["renter"] = pct(Z.House_rent_pct)
N["vacant"] = pct(Z.House_Vac_pct); N["college"] = pct(Z.High_edu_pct); N["unemp"] = pct(Z.Unemp_pc)
N["unemp_pre"] = pct(Z.LF_UNEMP_8QAgo_PCT)
N["hinc"] = Z.AG_HINC_C / Z.HHSCY.where(Z.HHSCY > 0); N["hinc"] = N.hinc / (1000 if N.hinc.median() > 1000 else 1)
N["dens"] = Z.POPCY / Z.AREALAND.where(Z.AREALAND > 0); N["ldens"] = np.log(N.dens.where(N.dens > 0))
N = N.replace([np.inf, -np.inf], np.nan)

D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); D.index = D.index.astype(str)
D = D.join(g).join(N, on="zip5")
CV = D[D.Gov == 0].copy(); W = CV[CV.r.abs() <= 14].copy()
print("window loans:", len(W), "| zips:", W.zip5.nunique(), "| counties:", W.cty.nunique(), "| coords:", W.Latitude.notna().mean().round(3))

# ---------------- (1) spatial concentration ----------------
vz = W.zip5.value_counts(); vc = W.cty.value_counts()
M.update(geoZips=f"{W.zip5.nunique():,}", geoCtys=f"{W.cty.nunique():,}", geoPerZip=f"{len(W)/W.zip5.nunique():.1f}",
         geoOneZip=f"{100*(vz==1).sum()/len(vz):.0f}", geoHHIz=f"{100*((vz/len(W))**2).sum():.2f}",
         geoMaxZip=f"{100*vz.iloc[0]/len(W):.1f}", geoMaxCty=f"{100*vc.iloc[0]/len(W):.1f}",
         geoCty5=f"{(vc>=5).sum()}", geoCty5Sh=f"{100*vc[vc>=5].sum()/len(W):.0f}")
# how interleaved are treated and control within a place?
both = W.groupby("cty").post.agg(["mean", "size"]); mix = both[(both["size"] >= 3)]
M["geoMixCty"] = f"{len(mix)}"; M["geoMixBoth"] = f"{100*((mix['mean'] > 0) & (mix['mean'] < 1)).mean():.0f}"
print("zips:", M["geoZips"], "loans/zip:", M["geoPerZip"], "| singleton zips:", M["geoOneZip"], "% | HHI(zip):", M["geoHHIz"])
print("counties with >=3 loans:", M["geoMixCty"], "| of those, both regimes present:", M["geoMixBoth"], "%")

# ---------------- map ----------------
mp = W.dropna(subset=["Latitude", "Longitude"]); mp = mp[(mp.Latitude.between(24, 50)) & (mp.Longitude.between(-125, -66))]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, c, lab in [(0, BLUE, "Asked before 7 April (relief on request)"), (1, ORANGE, "Asked from 7 April (full application required)")]:
    s = mp[mp.post == k]; ax.scatter(s.Longitude, s.Latitude, s=11, alpha=.62, c=c, lw=0, label=f"{lab}, n={len(s)}")
ax.set_aspect(1 / np.cos(np.deg2rad(37))); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=1, handletextpad=.2)
ax.set_title(f"{len(mp):,} conventional requests within 14 days of the cutoff, in {mp.zip5.nunique():,} zip codes", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(TEX, "fG_map.pdf")); fig.savefig(os.path.join(TEX, "fG_map.png"), dpi=120); plt.close(fig)

# ---------------- estimators ----------------
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def tri(x, h=14): x = x.copy(); x["w"] = 1 - x.r.abs() / (h + 1); return x
def ll(x, y, h=14, extra="", cl_on="r"):
    sub = [y] if cl_on == "r" else [y, cl_on]
    x = tri(x[x.r.abs() <= h].dropna(subset=sub), h)
    if extra: x = x[x.cty.notna()]
    m = smf.wls(f"{y} ~ post + r + post:r" + extra, x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x[cl_on])[0]})
    return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], int(m.nobs)
def twoway(x, y, h=14):
    x = tri(x[x.r.abs() <= h].dropna(subset=[y, "cty"]), h)
    m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([pd.factorize(x.r)[0], pd.factorize(x.cty)[0]])})
    return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"], int(m.nobs)
def conley(x, y, cut_km, h=14):
    """Conley spatial HAC with a Bartlett kernel on great-circle distance, on the triangular-kernel WLS."""
    x = tri(x[x.r.abs() <= h].dropna(subset=[y, "Latitude", "Longitude"]), h)
    X = np.column_stack([np.ones(len(x)), x.post, x.r, x.post * x.r]); yv = x[y].values; w = x.w.values
    Xw = X * np.sqrt(w)[:, None]; yw = yv * np.sqrt(w)
    XtX_inv = np.linalg.pinv(Xw.T @ Xw); b = XtX_inv @ (Xw.T @ yw); u = (yw - Xw @ b)
    la, lo = np.deg2rad(x.Latitude.values), np.deg2rad(x.Longitude.values)
    dlat = la[:, None] - la[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    d = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    K = np.clip(1 - d / cut_km, 0, None)                      # Bartlett in space
    K = np.maximum(K, (x.r.values[:, None] == x.r.values[None, :]).astype(float))   # same-day residuals fully correlated
    S = (Xw * u[:, None]).T @ K @ (Xw * u[:, None])
    V = XtX_inv @ S @ XtX_inv; se = np.sqrt(np.diag(V))[1]
    from scipy import stats
    return 100 * b[1], 100 * se, 2 * (1 - stats.norm.cdf(abs(b[1] / se))), len(x)
f = lambda e: f"{e[0]:.1f}{star(e[2])}"; s_ = lambda e: f"({e[1]:.1f})"
def tab(name, head, rows, align=None):
    a = align or ("l" + "c" * (sum(2 if "multicolumn{2}" in h else 1 for h in head) - 1))
    open(os.path.join(TEX, name), "w", encoding="utf-8").write("\n".join(
        ["\\begin{tabular}{" + a + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"] +
        ["\\midrule" if r == "MID" else " & ".join(map(str, r)) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]))

# ---------------- (4) inference table ----------------
OUTC = [("fb", "Forbearance granted"), ("dq_sep20", "Delinquent, September 2020"), ("mod_post", "Modified after April 2020"),
        ("performing", "Performing, April 2021"), ("fc_path", "Foreclosure path, April 2021")]
rows = []
for y, lab in OUTC:
    a = ll(CV, y); b = ll(CV, y, cl_on="cty"); c = twoway(CV, y); d1 = conley(CV, y, 100); d2 = conley(CV, y, 250)
    e = ll(CV, y, extra=" + C(cty)")
    rows.append([lab, f(a), s_(a), s_(b), s_(c), s_(d1), s_(d2), f(e), s_(e)])
    M["cnl" + y.replace("_", "")[:7]] = f"{d1[1]:.1f}"; M["cty" + y.replace("_", "")[:7]] = f"{e[0]:.1f}"; M["ctySE" + y.replace("_", "")[:7]] = f"{e[1]:.1f}"
tab("tG_infer.tex", ["Outcome", "Estimate", "Day", "County", "Two-way", "Conley 100km", "Conley 250km", "County FE", "s.e."], rows)
print("\nInference table written.")

# ---------------- (2)+(3) continuous neighborhood interactions ----------------
CHAR = [("nonwhite", "Non-white share"), ("hinc", "Mean household income"), ("ldens", "Log population density"),
        ("frontline", "Share in occupations that cannot be done from home"), ("renter", "Renter share"),
        ("vacant", "Vacant housing share"), ("college", "College-educated share"), ("unemp_pre", "Unemployment rate, two years earlier")]
zs = lambda v: (v - v.mean()) / v.std()
for c, _ in CHAR: CV["z_" + c] = zs(CV[c])
rows = []; pv = []
for c, lab in CHAR:
    x = tri(CV[(CV.r.abs() <= 14)].dropna(subset=["z_" + c]))
    r1 = []
    for y, _ in OUTC:
        xx = x.dropna(subset=[y])
        m = smf.wls(f"{y} ~ post*z_{c} + r + post:r", xx, weights=xx.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(xx.r)[0]})
        k = f"post:z_{c}"; r1 += [f"{100*m.params[k]:.1f}{star(m.pvalues[k])}", f"({100*m.bse[k]:.1f})"]; pv.append(m.pvalues[k])
    rows.append([lab, f"{CV[c].mean():.1f}", f"{CV[c].std():.1f}"] + r1)
from statsmodels.stats.multitest import multipletests
holm = multipletests(pv, method="holm")[1]
M["nIntTests"] = f"{len(pv)}"; M["minIntP"] = f"{min(pv):.3f}"; M["minIntHolm"] = f"{min(holm):.2f}"; M["nIntSig"] = f"{int((np.array(pv)<.05).sum())}"
rows.append("MID"); rows.append(["\\quad Smallest unadjusted $p$-value", "", ""] + [f"{min(pv):.3f}", ""] + [""] * 8)
rows.append(["\\quad Smallest Holm-adjusted $p$-value", "", ""] + [f"{min(holm):.2f}", ""] + [""] * 8)
tab("tG_cont.tex", ["Neighborhood characteristic (property zip code)", "Mean", "s.d."] + [f"\\multicolumn{{2}}{{c}}{{{l}}}" for _, l in OUTC], rows)
print("continuous interactions: min p =", round(min(pv), 4), "| min Holm p =", round(min(holm), 3), "| significant at 5%:", int((np.array(pv) < .05).sum()), "of", len(pv))

# joint test: all interactions zero, one outcome at a time
jt = []
for y, lab in OUTC:
    x = tri(CV[CV.r.abs() <= 14].dropna(subset=[y] + ["z_" + c for c, _ in CHAR]))
    terms = " + ".join(f"post:z_{c} + z_{c}" for c, _ in CHAR)
    m = smf.wls(f"{y} ~ post + r + post:r + {terms}", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
    R = [f"post:z_{c}" for c, _ in CHAR if f"post:z_{c}" in m.params.index]
    w_ = m.f_test(" = 0, ".join(R) + " = 0"); jt.append((lab, float(np.squeeze(w_.fvalue)), float(w_.pvalue), int(m.nobs)))
    print(f"  joint test, {lab}: F={jt[-1][1]:.2f}, p={jt[-1][2]:.2f}, n={jt[-1][3]}")
M["jointPmin"] = f"{min(j[2] for j in jt):.2f}"; M["jointPfc"] = f"{[j[2] for j in jt if 'Foreclosure' in j[0]][0]:.2f}"
tab("tG_joint.tex", ["Outcome", "$F$", "$p$", "Loans"], [[l, f"{F:.2f}", f"{p:.2f}", f"{n:,}"] for l, F, p, n in jt])

# frontline detail for the text
fl = CV.dropna(subset=["frontline"]); q = fl[fl.r.abs() <= 14].frontline.quantile([.1, .5, .9])
M.update(flMean=f"{fl[fl.r.abs()<=14].frontline.mean():.0f}", flP10=f"{q.iloc[0]:.0f}", flP90=f"{q.iloc[2]:.0f}")
x = tri(fl[fl.r.abs() <= 14].dropna(subset=["fc_path"]))
m = smf.wls("fc_path ~ post*z_frontline + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
M["flFc"] = f"{100*m.params['post:z_frontline']:.1f}"; M["flFcSE"] = f"{100*m.bse['post:z_frontline']:.1f}"; M["flFcP"] = f"{m.pvalues['post:z_frontline']:.2f}"
# balance of the neighborhood characteristics, continuous
rows = []
for c, lab in CHAR:
    a = ll(CV, c); b = ll(D[D.Gov == 1], c)
    rows.append([lab, f"{CV[CV.r.abs()<=14][c].mean():.1f}", f"{a[0]/100:.2f}{star(a[2])}", f"({a[1]/100:.2f})", f"{b[0]/100:.2f}{star(b[2])}", f"({b[1]/100:.2f})", f"{a[3]:,}"])
tab("tG_balance.tex", ["Neighborhood characteristic", "Mean before", "Conv.\\ jump", "s.e.", "Gov.\\ jump", "s.e.", "Loans"], rows)
M["geoMatchFull"] = f"{100*W.nonwhite.notna().mean():.0f}"
open(os.path.join(TEX, "numbers_geo.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
json.dump(M, open(os.path.join(OUT, "geo_main.json"), "w"))
print("\n--- macros ---"); [print(" ", k, "=", v) for k, v in M.items()]

