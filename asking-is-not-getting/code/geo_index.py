"""One test, not forty. Build a neighborhood-advantage index (first principal component of the zip
characteristics), estimate a single interaction per outcome, and do inference that does not rely on the
number of days. Then ask whether the pattern is a neighborhood effect or household resources measured twice.
"""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "juefull")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
rng = np.random.default_rng(20200407); M = {}
exec(open(os.path.join(H, "geo_main.py"), encoding="utf-8-sig").read().split("# ---------------- (1) spatial")[0].replace('M = {}', 'M = {}').replace("import matplotlib; matplotlib.use(\"Agg\"); import matplotlib.pyplot as plt", "pass"))

# ---------------- the index ----------------
COMP = [("hinc", 1), ("college", 1), ("nonwhite", -1), ("unemp_pre", -1), ("frontline", -1), ("renter", -1), ("vacant", -1)]
V = CV[[c for c, _ in COMP]].copy()
for c, s in COMP: V[c] = s * (V[c] - V[c].mean()) / V[c].std()
ok = V.notna().all(axis=1); Vm = V[ok].values
U, S, Vt = np.linalg.svd(Vm - Vm.mean(0), full_matrices=False)
load = Vt[0] * (1 if Vt[0][0] > 0 else -1)                       # orient so that higher = more advantaged
CV.loc[ok, "adv"] = (Vm - Vm.mean(0)) @ load
CV["adv"] = (CV.adv - CV.adv.mean()) / CV.adv.std()
var1 = S[0] ** 2 / (S ** 2).sum()
M["pcVar"] = f"{100*var1:.0f}"; M["pcN"] = f"{int(ok.sum()):,}"
print("PC1 explains", round(100 * var1), "% of variance. Loadings:")
for (c, s), l in zip(COMP, load): print(f"   {c:>10} {s:+d}  {l:+.2f}")
W = CV[CV.r.abs() <= 14]
M.update(advMatch=f"{100*W.adv.notna().mean():.0f}", advP10hinc=f"{W[W.adv<W.adv.quantile(.2)].hinc.mean():.0f}",
         advP90hinc=f"{W[W.adv>W.adv.quantile(.8)].hinc.mean():.0f}", advP10nw=f"{W[W.adv<W.adv.quantile(.2)].nonwhite.mean():.0f}",
         advP90nw=f"{W[W.adv>W.adv.quantile(.8)].nonwhite.mean():.0f}", advP10col=f"{W[W.adv<W.adv.quantile(.2)].college.mean():.0f}",
         advP90col=f"{W[W.adv>W.adv.quantile(.8)].college.mean():.0f}")
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def tri(x, h=14): x = x.copy(); x["w"] = 1 - x.r.abs() / (h + 1); return x
def fit(d, y, mod="post*adv + r + post:r", h=14):
    x = tri(d[d.r.abs() <= h].dropna(subset=[y, "adv"]), h)
    return smf.wls(f"{y} ~ {mod}", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}), x
def ri(d, y, mod="post*adv + r + post:r", key="post:adv", B=4999, h=14):
    """Randomization inference: reassign the post indicator across whole days, keeping the day structure."""
    m, x = fit(d, y, mod, h); t0 = abs(m.params[key] / m.bse[key]); days = np.sort(x.r.unique()); k = int((days >= 0).sum()); cnt = 0
    for _ in range(B):
        pick = set(rng.choice(days, k, replace=False)); z = x.copy(); z["post"] = z.r.isin(pick).astype(int)
        try:
            mm = smf.wls(f"{y} ~ {mod}", z, weights=z.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(z.r)[0]})
            if abs(mm.params[key] / mm.bse[key]) >= t0 - 1e-12: cnt += 1
        except Exception: cnt += 1
    return (cnt + 1) / (B + 1)
OUTC = [("fb", "Forbearance granted"), ("dq_sep20", "Delinquent, September 2020"), ("mod_post", "Modified after April 2020"),
        ("performing", "Performing, April 2021"), ("fc_path", "Foreclosure path, April 2021")]
CTRL = " + dq_feb20 + npl + ltv + lbal + tenure + pre_in + pre_out + hard"
rows = []; RES = {}
for y, lab in OUTC:
    m, x = fit(CV, y); b, se, p = 100 * m.params["post:adv"], 100 * m.bse["post:adv"], m.pvalues["post:adv"]
    mc, _ = fit(CV, y, "post*adv + r + post:r" + CTRL); bc, sec = 100 * mc.params["post:adv"], 100 * mc.bse["post:adv"]
    pr = ri(CV, y)
    rows.append([lab, f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})",
                 f"{b:.1f}{star(p)}", f"({se:.1f})", f"{pr:.3f}", f"{bc:.1f}{star(mc.pvalues['post:adv'])}", f"({sec:.1f})", f"{int(m.nobs):,}"])
    RES[y] = dict(main=100 * m.params["post"], inter=b, se=se, p=p, ri=pr, ctrl=bc, ctrlse=sec)
    print(f"{lab:<34} main {100*m.params['post']:6.1f}  interaction {b:6.1f} ({se:.1f}) p={p:.3f} RI p={pr:.3f}  with controls {bc:6.1f} ({sec:.1f})")
def tab(name, head, rows, align=None):
    a = align or ("l" + "c" * (sum(2 if "multicolumn{2}" in h else 1 for h in head) - 1))
    open(os.path.join(TEX, name), "w", encoding="utf-8").write("\n".join(["\\begin{tabular}{" + a + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"] + ["\\midrule" if r == "MID" else " & ".join(map(str, r)) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]))
tab("tG_index.tex", ["Outcome", "Jump at cutoff", "s.e.", "$\\times$ advantage", "s.e.", "RI $p$", "With controls", "s.e.", "Loans"], rows)
for y in ("performing", "fc_path", "dq_sep20"):
    M["ix" + y.replace("_", "")[:6]] = f"{RES[y]['inter']:.1f}"; M["ix" + y.replace("_", "")[:6] + "SE"] = f"{RES[y]['se']:.1f}"
    M["ix" + y.replace("_", "")[:6] + "RI"] = f"{RES[y]['ri']:.3f}"; M["ix" + y.replace("_", "")[:6] + "C"] = f"{RES[y]['ctrl']:.1f}"
M["ixfbI"] = f"{RES['fb']['inter']:.1f}"; M["ixfbSE"] = f"{RES['fb']['se']:.1f}"; M["ixfbRI"] = f"{RES['fb']['ri']:.3f}"

# ---------------- household resources vs neighborhood ----------------
E = None
try:
    ep = pd.read_csv(r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv",
                     dtype=str, usecols=["LoanID", "Advantage.Target.Income.3.0", "Liquid.Resources.2.0", "Advantage.Household.Education"])
    ep.columns = ["LoanID", "inc", "liq", "edu"]; ep = ep.drop_duplicates("LoanID").set_index("LoanID")
    for c in ("inc", "liq", "edu"): ep[c] = pd.to_numeric(ep[c].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    CV = CV.join(ep, rsuffix="_e"); E = True
    CV["hh"] = (CV.inc - CV.inc.mean()) / CV.inc.std()
    print("\nhousehold file matched:", round(100 * CV[CV.r.abs() <= 14].hh.notna().mean()), "% of the window")
    M["hhMatch"] = f"{100*CV[CV.r.abs()<=14].hh.notna().mean():.0f}"
    rows2 = []
    for y, lab in [("performing", "Performing, April 2021"), ("fc_path", "Foreclosure path, April 2021")]:
        a, _ = fit(CV, y, "post*hh + r + post:r"); b, _ = fit(CV, y, "post*hh + post*adv + r + post:r")
        rows2.append([lab, f"{100*a.params['post:hh']:.1f}{star(a.pvalues['post:hh'])}", f"({100*a.bse['post:hh']:.1f})",
                      f"{100*b.params['post:hh']:.1f}{star(b.pvalues['post:hh'])}", f"({100*b.bse['post:hh']:.1f})",
                      f"{100*b.params['post:adv']:.1f}{star(b.pvalues['post:adv'])}", f"({100*b.bse['post:adv']:.1f})", f"{int(b.nobs):,}"])
        if y == "performing":
            M["hhPerf"] = f"{100*a.params['post:hh']:.1f}"; M["hhPerfSE"] = f"{100*a.bse['post:hh']:.1f}"
            M["hhPerfJ"] = f"{100*b.params['post:hh']:.1f}"; M["advPerfJ"] = f"{100*b.params['post:adv']:.1f}"; M["advPerfJSE"] = f"{100*b.bse['post:adv']:.1f}"
        print(f"  {lab}: household {100*a.params['post:hh']:.1f}; jointly household {100*b.params['post:hh']:.1f}, neighborhood {100*b.params['post:adv']:.1f}")
    tab("tG_hh.tex", ["Outcome", "$\\times$ household income", "s.e.", "$\\times$ household income", "s.e.", "$\\times$ neighborhood advantage", "s.e.", "Loans"], rows2)
    M["corrHhAdv"] = f"{CV[['hh','adv']].corr().iloc[0,1]:.2f}"
except Exception as ex: print("household file:", ex)

# ---------------- figure: effect by advantage ----------------
q = pd.qcut(W.adv, 4, labels=False, duplicates="drop"); WW = W.assign(g=q).dropna(subset=["g"])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
for ax, (y, ttl) in zip(axs, [("fb", "Forbearance granted (first stage)"), ("performing", "Performing in April 2021")]):
    pts = []
    for gi in sorted(WW.g.unique()):
        s = WW[WW.g == gi].dropna(subset=[y])
        if len(s) < 25: continue
        m = smf.ols(f"{y} ~ post", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.r)[0]})
        pts.append((s.adv.mean(), 100 * m.params["post"], 196 * m.bse["post"]))
    a_ = np.array(pts); ax.errorbar(a_[:, 0], a_[:, 1], yerr=a_[:, 2], fmt="o-", ms=4, lw=1.2, capsize=2.5, color=ORANGE if y == "fb" else BLUE)
    ax.axhline(0, color=GREY, lw=.7); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("Neighborhood advantage index (quartile means)", fontsize=8)
axs[0].set_ylabel("Difference across 7 April (pp)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(TEX, "fG_index.pdf")); fig.savefig(os.path.join(TEX, "fG_index.png"), dpi=120); plt.close(fig)
open(os.path.join(TEX, "numbers_geo2.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
json.dump(M, open(os.path.join(OUT, "geo_index.json"), "w"))
print("\n--- macros ---"); [print(" ", k, "=", v) for k, v in M.items()]

