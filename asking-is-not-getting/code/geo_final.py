"""Finish the spatial section: clean placebo cutoffs, effects at the ends of the advantage distribution,
and what refusal actually did in disadvantaged neighborhoods (full disposition decomposition)."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "juefull")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#c8551f", "#2f8f5b", "#777777"
rng = np.random.default_rng(20200407); M = {}
exec(open(os.path.join(H, "geo_index.py"), encoding="utf-8-sig").read().split("# ---------------- household resources")[0])
Y, KEY, BASE = "performing", "post:adv", "post*adv + r + post:r"
def fit2(d, y, mod, h=14, donut=0, extra_na=()):
    x = d[(d.r.abs() <= h) & ~((d.r >= -donut) & (d.r < donut))].dropna(subset=[y, "adv", *extra_na]).copy(); x["w"] = 1 - x.r.abs() / (h + 1)
    return smf.wls(f"{y} ~ {mod}", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}), x

# ---------------- clean placebo cutoffs ----------------
print("=== placebo cutoffs (a window of +-14 days around a shifted date contains the true cutoff unless |shift| > 14) ===")
rows = []
for s in (-42, -35, -28, -21, 21, 28, 35, 42):
    z = CV.copy(); z["r"] = z.r - s; z["post"] = (z.r >= 0).astype(int)
    try:
        m, x = fit2(z, Y, BASE)
        if int(m.nobs) < 60: print(f"  shift {s:+3d}: only {int(m.nobs)} loans, skipped"); continue
        rows.append((s, 100 * m.params[KEY], 100 * m.bse[KEY], m.pvalues[KEY], int(m.nobs)))
        print(f"  shift {s:+3d} days: interaction {100*m.params[KEY]:6.1f} ({100*m.bse[KEY]:.1f})  p={m.pvalues[KEY]:.2f}  n={int(m.nobs)}")
    except Exception as e: print(f"  shift {s:+3d}: {e}")
if rows:
    M["plMax"] = f"{max(abs(b) for _, b, _, _, _ in rows):.1f}"; M["plN"] = f"{len(rows)}"; M["plSig"] = f"{sum(1 for *_, p, _ in [(r[0], r[1], r[2], r[3], r[4]) for r in rows] if p < .05)}"
    M["plSig"] = f"{sum(1 for r in rows if r[3] < .05)}"
print("clean placebos:", M.get("plN"), "| largest |interaction|:", M.get("plMax"), "| significant at 5%:", M.get("plSig"))

# ---------------- effect at the ends of the advantage distribution ----------------
m, x = fit2(CV, Y, BASE)
W = x; qs = W.adv.quantile([.125, .5, .875])
print("\n=== implied effect of refusal on performing status, by neighborhood advantage ===")
eff = []
for lab, a in [("Bottom quartile (mean)", W[W.adv <= W.adv.quantile(.25)].adv.mean()), ("Median", W.adv.median()), ("Top quartile (mean)", W[W.adv >= W.adv.quantile(.75)].adv.mean())]:
    c = np.zeros(len(m.params)); c[list(m.params.index).index("post")] = 1; c[list(m.params.index).index(KEY)] = a
    t = m.t_test(c); b_, se_ = 100 * float(np.squeeze(t.effect)), 100 * float(np.squeeze(t.sd))
    eff.append((lab, a, b_, se_, float(np.squeeze(t.pvalue)))); print(f"  {lab:<24} adv={a:+.2f}  effect {b_:+6.1f} (s.e. {se_:.1f})  p={float(np.squeeze(t.pvalue)):.3f}")
M.update(effLo=f"{eff[0][2]:+.1f}", effLoSE=f"{eff[0][3]:.1f}", effLoP=f"{eff[0][4]:.2f}", effHi=f"{eff[2][2]:+.1f}", effHiSE=f"{eff[2][3]:.1f}", effHiP=f"{eff[2][4]:.3f}")

# ---------------- what happened instead, in disadvantaged neighborhoods ----------------
CATS = [("Performing", ["Performing"]), ("Modification completed or in review", ["Modification Completed", "Modification in Review"]),
        ("Notice of intent filed, not in foreclosure", ["NOI Filed: Not in FC"]), ("Foreclosure pending or REO", ["Pending Foreclosure Completion", "REO"]),
        ("Bankruptcy", ["BK"])]
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
CV["lo"] = (CV.adv <= CV[CV.r.abs() <= 14].adv.median()).astype(float).where(CV.adv.notna())
rows = []
print("\n=== April 2021 status, by half of the advantage distribution ===")
for lab, v in CATS:
    CV["yv"] = CV.disp.isin(v).astype(float).where(CV.disp.notna()); cells = []
    for k, nm in [(1, "less advantaged"), (0, "more advantaged")]:
        d = CV[CV.lo == k]; w = d[d.r.abs() <= 14]
        mm = smf.ols("yv ~ post", w.dropna(subset=["yv"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(w.dropna(subset=['yv']).r)[0]})
        cells += [f"{100*w[w.post==0].yv.mean():.1f}", f"{100*w[w.post==1].yv.mean():.1f}", f"{100*mm.params['post']:.1f}{star(mm.pvalues['post'])}", f"({100*mm.bse['post']:.1f})"]
        if k == 1: lo_ = 100 * mm.params["post"]
        else: hi_ = 100 * mm.params["post"]
    xx = CV[CV.r.abs() <= 14].dropna(subset=["yv", "lo"]).copy(); xx["w"] = 1 - xx.r.abs() / 15
    mi = smf.wls("yv ~ post*lo + r + post:r", xx, weights=xx.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(xx.r)[0]})
    rows.append([lab] + cells + [f"{mi.pvalues['post:lo']:.2f}"])
    print(f"  {lab:<44} less adv {lo_:+6.1f}   more adv {hi_:+6.1f}   p(diff)={mi.pvalues['post:lo']:.2f}")
    key = lab.split()[0].lower()[:4]; M["lo" + key] = f"{lo_:+.1f}"; M["hi" + key] = f"{hi_:+.1f}"
def tab(name, head, rows, align=None):
    a = align or ("l" + "c" * (sum(2 if "multicolumn{2}" in h else (4 if "multicolumn{4}" in h else 1) for h in head) - 1))
    open(os.path.join(TEX, name), "w", encoding="utf-8").write("\n".join(["\\begin{tabular}{" + a + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"] + ["\\midrule" if r == "MID" else " & ".join(map(str, r)) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]))
tab("tG_dispo.tex", ["Status in April 2021", "\\multicolumn{4}{c}{Less advantaged neighborhoods}", "\\multicolumn{4}{c}{More advantaged neighborhoods}", "$p$"],
    [r if r == "MID" else r for r in rows], align="l" + "cccc" * 2 + "c")
# header row detail
s = open(os.path.join(TEX, "tG_dispo.tex"), encoding="utf-8").read().replace("\\midrule\n", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n & Before & After & Diff. & s.e. & Before & After & Diff. & s.e. & \\\\\n\\midrule\n", 1)
open(os.path.join(TEX, "tG_dispo.tex"), "w", encoding="utf-8").write(s)

# ---------------- figure: two panels, screen vs outcome ----------------
W2 = CV[CV.r.abs() <= 14].dropna(subset=["adv"]).copy(); W2["g"] = pd.qcut(W2.adv, 4, labels=False, duplicates="drop")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, (y, ttl, col) in zip(axs, [("fb", "The screen: forbearance granted", ORANGE), ("performing", "A year later: performing in April 2021", BLUE)]):
    pts = []
    for gi in sorted(W2.g.dropna().unique()):
        s_ = W2[W2.g == gi].dropna(subset=[y])
        if len(s_) < 25: continue
        mm = smf.ols(f"{y} ~ post", s_).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s_.r)[0]})
        pts.append((s_.adv.mean(), 100 * mm.params["post"], 196 * mm.bse["post"]))
    a_ = np.array(pts); ax.errorbar(a_[:, 0], a_[:, 1], yerr=a_[:, 2], fmt="o-", ms=4.5, lw=1.3, capsize=2.5, color=col)
    ax.axhline(0, color=GREY, lw=.8); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("Neighborhood advantage (quartile means, s.d.)", fontsize=8)
axs[0].set_ylabel("Difference across 7 April (pp)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(TEX, "fG_index.pdf")); fig.savefig(os.path.join(TEX, "fG_index.png"), dpi=120); plt.close(fig)
old = json.load(open(os.path.join(OUT, "geo_index.json"))); old.update(M)
open(os.path.join(TEX, "numbers_geo2.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in old.items()) + "\n")
json.dump(old, open(os.path.join(OUT, "geo_index.json"), "w"))
print("\n--- new macros ---"); [print(" ", k, "=", v) for k, v in M.items()]
