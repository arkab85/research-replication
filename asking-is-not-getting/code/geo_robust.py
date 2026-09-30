"""Stress-test the one interaction that matters: refusal x neighborhood advantage, on performing status a year later.
Balance of the index itself, bandwidths, donut, wild bootstrap, household-vs-neighborhood horse race, placebo dates."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "juefull")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
rng = np.random.default_rng(20200407); M = {}
src = open(os.path.join(H, "geo_index.py"), encoding="utf-8-sig").read()
exec(src.split("# ---------------- household resources")[0])          # rebuilds CV with adv, fit(), ri(), tab()
def fit2(d, y, mod, extra_na=(), h=14, donut=0):
    x = d[(d.r.abs() <= h) & ~((d.r >= -donut) & (d.r < donut))].dropna(subset=[y, "adv", *extra_na]).copy()
    x["w"] = 1 - x.r.abs() / (h + 1)
    return smf.wls(f"{y} ~ {mod}", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}), x
Y = "performing"; KEY = "post:adv"; BASE = "post*adv + r + post:r"
print("=== balance: is the advantage index itself continuous at the cutoff? ===")
for lab, d in [("Conventional", CV), ("Government-backed", D[D.Gov == 1].assign(adv=CV.adv.reindex(D[D.Gov == 1].index)))]:
    dd = d.dropna(subset=["adv"])
    if len(dd) < 50: continue
    m, x = fit2(dd, "adv", "post + r + post:r")
    print(f"  {lab}: jump {m.params['post']:+.3f} s.d. (s.e. {m.bse['post']:.3f}), n={int(m.nobs)}")
    if lab == "Conventional": M["advBal"] = f"{m.params['post']:+.2f}"; M["advBalSE"] = f"{m.bse['post']:.2f}"
gg = D[D.Gov == 1].copy()
gz = CV[["zip5"]].copy()
print("\n=== bandwidth and donut ===")
rows = []
for h in (7, 10, 14, 21, 28):
    m, x = fit2(CV, Y, BASE, h=h); rows.append([f"Bandwidth {h} days", f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})", f"{100*m.params[KEY]:.1f}{star(m.pvalues[KEY])}", f"({100*m.bse[KEY]:.1f})", f"{int(m.nobs):,}"])
    print(f"  h={h}: main {100*m.params['post']:6.1f}  interaction {100*m.params[KEY]:6.1f} ({100*m.bse[KEY]:.1f})")
    if h == 14: M["ixMain"] = f"{100*m.params['post']:.1f}"; M["ixMainSE"] = f"{100*m.bse['post']:.1f}"
m, x = fit2(CV, Y, BASE, donut=1); rows += ["MID", ["Donut: 6 and 7 April excluded", f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})", f"{100*m.params[KEY]:.1f}{star(m.pvalues[KEY])}", f"({100*m.bse[KEY]:.1f})", f"{int(m.nobs):,}"]]
M["ixDonut"] = f"{100*m.params[KEY]:.1f}"
print(f"  donut: interaction {100*m.params[KEY]:.1f} ({100*m.bse[KEY]:.1f})")
CTRL = " + dq_feb20 + npl + ltv + lbal + tenure + pre_in + pre_out + hard"
m, x = fit2(CV, Y, BASE + CTRL); rows.append(["Loan controls", f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})", f"{100*m.params[KEY]:.1f}{star(m.pvalues[KEY])}", f"({100*m.bse[KEY]:.1f})", f"{int(m.nobs):,}"])
m, x = fit2(CV, Y, BASE + CTRL + " + post:dq_feb20 + post:ltv + post:lbal"); rows.append(["\\quad + controls interacted with the cutoff", f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})", f"{100*m.params[KEY]:.1f}{star(m.pvalues[KEY])}", f"({100*m.bse[KEY]:.1f})", f"{int(m.nobs):,}"])
M["ixCtrlX"] = f"{100*m.params[KEY]:.1f}"; M["ixCtrlXSE"] = f"{100*m.bse[KEY]:.1f}"
print(f"  controls interacted: {100*m.params[KEY]:.1f} ({100*m.bse[KEY]:.1f})")
m, x = fit2(CV, Y, BASE + " + C(state)"); rows.append(["State fixed effects", f"{100*m.params['post']:.1f}{star(m.pvalues['post'])}", f"({100*m.bse['post']:.1f})", f"{100*m.params[KEY]:.1f}{star(m.pvalues[KEY])}", f"({100*m.bse[KEY]:.1f})", f"{int(m.nobs):,}"])
M["ixStateFE"] = f"{100*m.params[KEY]:.1f}"; M["ixStateFESE"] = f"{100*m.bse[KEY]:.1f}"
print(f"  state FE: {100*m.params[KEY]:.1f} ({100*m.bse[KEY]:.1f})")

# ---- wild cluster bootstrap on the interaction (null imposed) ----
def wild(d, y, mod, key, B=1999, h=14):
    m, x = fit2(d, y, mod, h=h); t0 = abs(m.params[key] / m.bse[key])
    restricted = mod.replace(" + " + key, "").replace(key + " + ", "")
    terms = [t for t in mod.replace("*", " + ").split(" + ")]
    r0 = smf.wls(f"{y} ~ post + adv + r + post:r", x, weights=x.w).fit()
    u = r0.resid.values; fitv = r0.fittedvalues.values; days = pd.factorize(x.r)[0]; cnt = 0
    for _ in range(B):
        sgn = rng.choice([-1.0, 1.0], days.max() + 1)[days]
        z = x.copy(); z[y] = fitv + u * sgn
        mm = smf.wls(f"{y} ~ {mod}", z, weights=z.w).fit(cov_type="cluster", cov_kwds={"groups": days})
        if abs(mm.params[key] / mm.bse[key]) >= t0 - 1e-12: cnt += 1
    return (cnt + 1) / (B + 1)
pw = wild(CV, Y, BASE, KEY); M["ixWild"] = f"{pw:.3f}"; print(f"\nwild cluster bootstrap p (interaction) = {pw:.3f}")

# ---- placebo cutoff dates for the interaction ----
pl = []
for shift in range(-28, 29, 7):
    if abs(shift) < 7: continue
    z = CV.copy(); z["r"] = z.r - shift; z = z[z.r.abs() <= 14]
    try:
        m, _ = fit2(z, Y, BASE); pl.append((shift, 100 * m.params[KEY], m.pvalues[KEY]))
    except Exception: pass
M["ixPlaceboMax"] = f"{max(abs(b) for _, b, _ in pl):.1f}"; M["ixPlaceboN"] = f"{len(pl)}"
print("placebo interactions:", [(s, round(b, 1)) for s, b, _ in pl])

# ---- household vs neighborhood ----
try:
    ep = pd.read_csv(r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv",
                     dtype=str, usecols=["LoanID", "Advantage.Target.Income.3.0", "Liquid.Resources.2.0"])
    ep.columns = ["LoanID", "inc", "liq"]; ep = ep.drop_duplicates("LoanID").set_index("LoanID")
    for c in ("inc", "liq"): ep[c] = pd.to_numeric(ep[c].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    CV2 = CV.join(ep); CV2["hh"] = (CV2.inc - CV2.inc.mean()) / CV2.inc.std()
    M["hhMatch"] = f"{100*CV2[CV2.r.abs()<=14].hh.notna().mean():.0f}"; M["corrHhAdv"] = f"{CV2[['hh','adv']].corr().iloc[0,1]:.2f}"
    print(f"\nhousehold income matched for {M['hhMatch']}% of the window; corr(household, neighborhood) = {M['corrHhAdv']}")
    r2 = []
    a, _ = fit2(CV2, Y, "post*hh + r + post:r", extra_na=("hh",)); b, _ = fit2(CV2, Y, "post*hh + post*adv + r + post:r", extra_na=("hh",))
    r2.append(["Household income only", f"{100*a.params['post:hh']:.1f}{star(a.pvalues['post:hh'])}", f"({100*a.bse['post:hh']:.1f})", "", "", f"{int(a.nobs):,}"])
    c, _ = fit2(CV2, Y, BASE, extra_na=("hh",))
    r2.append(["Neighborhood advantage only", "", "", f"{100*c.params[KEY]:.1f}{star(c.pvalues[KEY])}", f"({100*c.bse[KEY]:.1f})", f"{int(c.nobs):,}"])
    r2.append(["Both", f"{100*b.params['post:hh']:.1f}{star(b.pvalues['post:hh'])}", f"({100*b.bse['post:hh']:.1f})", f"{100*b.params[KEY]:.1f}{star(b.pvalues[KEY])}", f"({100*b.bse[KEY]:.1f})", f"{int(b.nobs):,}"])
    tab("tG_hh.tex", ["Interacted with the post-cutoff indicator", "$\\times$ household income", "s.e.", "$\\times$ neighborhood advantage", "s.e.", "Loans"], r2)
    M.update(hhAlone=f"{100*a.params['post:hh']:.1f}", hhAloneSE=f"{100*a.bse['post:hh']:.1f}", hhBoth=f"{100*b.params['post:hh']:.1f}",
             hhBothSE=f"{100*b.bse['post:hh']:.1f}", advBoth=f"{100*b.params[KEY]:.1f}", advBothSE=f"{100*b.bse[KEY]:.1f}", advAlone=f"{100*c.params[KEY]:.1f}")
    print(f"  household alone {100*a.params['post:hh']:.1f} ({100*a.bse['post:hh']:.1f}); jointly household {100*b.params['post:hh']:.1f} ({100*b.bse['post:hh']:.1f}), neighborhood {100*b.params[KEY]:.1f} ({100*b.bse[KEY]:.1f})")
except Exception as ex: print("household file:", ex)
tab("tG_robust.tex", ["Specification", "Jump at cutoff", "s.e.", "$\\times$ advantage", "s.e.", "Loans"], rows)

# ---- figure ----
W2 = CV[CV.r.abs() <= 14].dropna(subset=["adv"]).copy(); W2["g"] = pd.qcut(W2.adv, 4, labels=False, duplicates="drop")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75))
for ax, (y, ttl, col) in zip(axs, [("fb", "Forbearance granted (the screen)", ORANGE), ("performing", "Performing in April 2021", BLUE)]):
    pts = []
    for gi in sorted(W2.g.dropna().unique()):
        s = W2[W2.g == gi].dropna(subset=[y])
        if len(s) < 25: continue
        m = smf.ols(f"{y} ~ post", s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.r)[0]})
        pts.append((s.adv.mean(), 100 * m.params["post"], 196 * m.bse["post"]))
    a_ = np.array(pts); ax.errorbar(a_[:, 0], a_[:, 1], yerr=a_[:, 2], fmt="o-", ms=4.5, lw=1.3, capsize=2.5, color=col)
    ax.axhline(0, color=GREY, lw=.7); ax.set_title(ttl, fontsize=8.5)
    ax.set_xlabel("Neighborhood advantage (quartile means, s.d.)", fontsize=8)
axs[0].set_ylabel("Difference across 7 April (pp)", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(TEX, "fG_index.pdf")); fig.savefig(os.path.join(TEX, "fG_index.png"), dpi=120); plt.close(fig)
old = json.load(open(os.path.join(OUT, "geo_index.json"))); old.update(M)
open(os.path.join(TEX, "numbers_geo2.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in old.items()) + "\n")
json.dump(old, open(os.path.join(OUT, "geo_index.json"), "w"))
print("\n--- new macros ---"); [print(" ", k, "=", v) for k, v in M.items()]
