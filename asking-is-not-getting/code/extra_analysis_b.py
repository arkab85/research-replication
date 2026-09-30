"""the journal round: CCT robust RD, density test, attrition (+Lee bounds), monthly payment dynamics,
the provider's ledger, bandwidth sensitivity, multiple-testing adjustment. Writes tables, figures, numbers_ms.tex."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
def save(fig, p): fig.savefig(p); fig.savefig(p[:-4] + ".png", dpi=110); plt.close(fig)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def tab(path, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1)); s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(str(v) for v in r) + " \\\\")
    open(os.path.join(TEX, path), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); Cv, Gv = D[D.Gov == 0].copy(), D[D.Gov == 1].copy(); M = {}; RES = {}
def ll(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy().reset_index(drop=True); x["w"] = 1 - x.r.abs() / (h + 1)
    m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"], len(x)
def dm(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]).reset_index(drop=True); m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"], len(x), x.loc[x.post == 0, y].mean()

# ---------- 1. CCT robust bias-corrected ----------
OUTS = [("fb", "Forbearance", 100), ("dq_sep20", "Delinquent, Sep 2020", 100), ("mod_post", "Modified after Apr 2020", 100), ("performing", "Performing, Apr 2021", 100), ("fc_path", "Foreclosure path, Apr 2021", 100)]
rows = []
try:
    from rdrobust import rdrobust
    for y, lab, sc in OUTS:
        x = Cv.dropna(subset=[y]); x = x[x.r.abs() <= 45]
        r = rdrobust(y=x[y].values.astype(float), x=x.r.values.astype(float), c=0, masspoints="adjust", vce="hc1")
        co, rb = float(r.coef.iloc[0, 0]), float(r.coef.iloc[2, 0]); se_r = float(r.se.iloc[2, 0]); p_r = float(r.pv.iloc[2, 0]); ci = r.ci.iloc[2].values; h = float(r.bws.iloc[0, 0]); nh = int(np.sum(r.N_h))
        rows.append([lab, f"{sc*co:.1f}", f"{sc*rb:.1f}{star(p_r)}", f"[{sc*ci[0]:.1f}, {sc*ci[1]:.1f}]", f"{h:.1f}", f"{nh:,}"]); RES["cct_" + y] = {"conv": co, "rb": rb, "p": p_r, "ci": [float(ci[0]), float(ci[1])], "h": h, "n": nh}
        k = y.replace("_", "").replace("20", "").replace("21", ""); M[f"cct{k}"] = f"{abs(sc*rb):.1f}"; M[f"cctp{k}"] = f"{p_r:.3f}"; M[f"ccth{k}"] = f"{h:.0f}"
        print("CCT", lab, rows[-1][1:])
    tab("tA2_cct.tex", ["Outcome (pp)", "Conventional estimate", "Bias-corrected", "Robust 95\\% CI", "Bandwidth (days)", "Loans"], rows)
except Exception as e: print("rdrobust failed:", repr(e))
# ---------- 2. density ----------
try:
    from rddensity import rddensity
    for lab, d in [("conv", Cv), ("gov", Gv)]:
        x = d[(d.r.abs() <= 45) & (d.inq.dt.dayofweek < 5)]; rr = rddensity(X=x.r.values.astype(float), c=0); p = float(rr.test["p_jk"]); RES["dens_" + lab] = p; M["dens" + lab.capitalize()] = f"{p:.2f}"; print("density p", lab, p)
except Exception as e: print("rddensity failed:", repr(e))
# ---------- 3. attrition + Lee bounds ----------
Cv["obs_sep"] = Cv.dq_sep20.notna().astype(float); a = dm(Cv, "obs_sep"); a2 = ll(Cv, "obs_sep", 14); RES["attr"] = {"dm": a[:3], "ll": a2[:3], "mean": a[4]}
M.update(attrDM=f"{100*a[0]:.1f}", attrDMSE=f"{100*a[1]:.1f}", attrLL=f"{100*a2[0]:.1f}", attrLLSE=f"{100*a2[1]:.1f}", attrMean=f"{100*a[4]:.0f}"); print("attrition: dm", a[:3], "ll", a2[:3])
w = Cv[Cv.r.abs() <= 14]; p1, p0 = w[w.post == 1].obs_sep.mean(), w[w.post == 0].obs_sep.mean()
y1, y0 = w[(w.post == 1) & (w.obs_sep == 1)].dq_sep20, w[(w.post == 0) & (w.obs_sep == 1)].dq_sep20
if p1 > p0:   # trim the group observed more often
    q = (p1 - p0) / p1; k = int(round(q * len(y1))); s1 = np.sort(y1.values); lo, hi = s1[:len(s1) - k].mean() - y0.mean(), s1[k:].mean() - y0.mean()
else:
    q = (p0 - p1) / p0; k = int(round(q * len(y0))); s0 = np.sort(y0.values); lo, hi = y1.mean() - s0[k:].mean(), y1.mean() - s0[:len(s0) - k].mean()
RES["lee"] = (lo, hi, q); M.update(leeLo=f"{100*lo:.1f}", leeHi=f"{100*hi:.1f}", leeTrim=f"{100*q:.1f}"); print("Lee bounds dq_sep20:", lo, hi, "trim", q)
# ---------- 4. monthly payments + 5. provider ledger ----------
rm = pd.read_csv(r"<DATA>/relief\RemittanceGAAP.csv", dtype={"LoanID": str}); rm["m"] = pd.to_datetime(rm.RemittanceDate, errors="coerce").dt.to_period("M"); rm = rm.dropna(subset=["m"])
cols = ["PrincipalPayment", "InterestPayment", "ProceedsOnClosedPositions", "HAMPFunds", "CorporateRecovery", "CorporateAdvance", "EscrowRecovery", "EscrowAdvance", "ServiceFee"]
for c in cols: rm[c] = pd.to_numeric(rm[c], errors="coerce").fillna(0)
rm = rm[rm.LoanID.isin(D.index)]; rm["pi"] = rm.PrincipalPayment + rm.InterestPayment; rm["net"] = rm[cols].sum(axis=1); rm["adv"] = -(rm.CorporateAdvance + rm.EscrowAdvance); rm["fee"] = -rm.ServiceFee
g = rm.groupby(["LoanID", "m"])[["pi", "net", "adv", "fee", "ProceedsOnClosedPositions"]].sum().reset_index()
months = pd.period_range("2020-01", "2021-03", freq="M"); dyn = []
for m in months:
    paid = (g[g.m == m].set_index("LoanID").pi > 0).astype(float)
    for lab, d in [("conv", Cv), ("gov", Gv)]:
        d["_p"] = paid.reindex(d.index).fillna(0).values; a = dm(d, "_p"); dyn.append({"m": str(m), "grp": lab, "b": a[0], "se": a[1], "mean": a[4]})
dyn = pd.DataFrame(dyn); dyn.to_csv(os.path.join(OUT, "v_monthly_payments.csv"), index=False)
fig, ax = plt.subplots(figsize=(6.0, 2.9)); xs = np.arange(len(months))
for lab, col, nm, off in [("conv", ORANGE, "Conventional", -.12), ("gov", BLUE, "Government-backed (placebo)", .12)]:
    z = dyn[dyn.grp == lab]; ax.errorbar(xs + off, 100 * z.b, yerr=196 * z.se, color=col, marker="o", ms=3.5, lw=1.1, capsize=2, label=nm)
ax.axhline(0, color=GREY, lw=.6); ax.axvline(2.5, color=GREY, lw=.6, ls=":"); ax.set_xticks(xs); ax.set_xticklabels([m.strftime("%b\n%y") if m.month in (1, 4, 7, 10) else m.strftime("%b") for m in months], fontsize=7.5)
ax.set_ylabel("Asked after minus before 7 April:\nshare making a payment (pp)"); ax.legend(frameon=False, fontsize=8, loc="upper right"); save(fig, os.path.join(TEX, "f7_monthly.pdf"))
z = dyn[dyn.grp == "conv"].set_index("m"); M.update(payApr=f"{100*z.loc['2020-04','b']:.1f}", payMay=f"{100*z.loc['2020-05','b']:.1f}", payJun=f"{100*z.loc['2020-06','b']:.1f}", payPeak=f"{100*z.b.max():.1f}", payPeakMonth=pd.Period(z.b.idxmax()).strftime("%B %Y"))
# difference-in-discontinuities for payments in May-June 2020 (months with a payment, 0-2) and Jul 2020-Mar 2021 (0-9)
for k, lo_, hi_ in [("mj", "2020-05", "2020-06"), ("late", "2020-07", "2021-03")]:
    s_ = (g[(g.m >= pd.Period(lo_, "M")) & (g.m <= pd.Period(hi_, "M"))].assign(p=lambda z: (z.pi > 0).astype(float)).groupby("LoanID").p.sum())
    D["_y"] = s_.reindex(D.index).fillna(0).values; D["conv"] = 1 - D.Gov
    x = D[D.r.abs() <= 14].reset_index(drop=True); mm = smf.ols("_y ~ post*conv", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
    c_ = dm(D[D.Gov == 0], "_y"); g_ = dm(D[D.Gov == 1], "_y"); RES["pay_" + k] = {"conv": c_[:3], "gov": g_[:3], "dd": (mm.params["post:conv"], mm.bse["post:conv"], mm.pvalues["post:conv"])}
    M.update({f"pay{k.capitalize()}Conv": f"{c_[0]:.2f}", f"pay{k.capitalize()}ConvSE": f"{c_[1]:.2f}", f"pay{k.capitalize()}Gov": f"{g_[0]:.2f}", f"pay{k.capitalize()}DD": f"{mm.params['post:conv']:.2f}", f"pay{k.capitalize()}DDSE": f"{mm.bse['post:conv']:.2f}", f"pay{k.capitalize()}DDP": f"{mm.pvalues['post:conv']:.3f}", f"pay{k.capitalize()}Mean": f"{c_[4]:.2f}"})
    print("PAY", k, "conv", c_[:3], "gov", g_[:3], "DD", mm.params["post:conv"], mm.bse["post:conv"], mm.pvalues["post:conv"])
def win(col, a, b): return g[(g.m >= pd.Period(a, "M")) & (g.m <= pd.Period(b, "M"))].groupby("LoanID")[col].sum()
LED = [("pi3", "Principal and interest, Apr--Jun 2020", win("pi", "2020-04", "2020-06")), ("pi6", "Principal and interest, Apr--Sep 2020", win("pi", "2020-04", "2020-09")), ("pi12", "Principal and interest, Apr 2020--Mar 2021", win("pi", "2020-04", "2021-03")),
       ("adv12", "Servicer advances, Apr 2020--Mar 2021", win("adv", "2020-04", "2021-03")), ("net12", "All cash flows to the investor, Apr 2020--Mar 2021", win("net", "2020-04", "2021-03"))]
rows = []
for k, lab, s in LED:
    for d in (Cv, Gv): d[k] = s.reindex(d.index).fillna(0).values
    for d in (Cv, Gv): d[k + "_w"] = d[k].clip(lower=d[k].quantile(.01), upper=d[k].quantile(.99))
    a, b, gg = dm(Cv, k + "_w"), ll(Cv, k + "_w", 14), dm(Gv, k + "_w"); RES["led_" + k] = {"dm": a[:3], "ll": b[:3], "gov": gg[:3], "mean": a[4]}
    rows += [[lab, f"{a[4]:,.0f}", f"{a[0]:,.0f}{star(a[2])}", f"{b[0]:,.0f}{star(b[2])}", f"{gg[0]:,.0f}{star(gg[2])}"], ["", "", f"({a[1]:,.0f})", f"({b[1]:,.0f})", f"({gg[1]:,.0f})"]]
    M[f"led{k.replace('3','Three').replace('6','Six').replace('12','Twelve')}"] = f"{a[0]:,.0f}"; M[f"led{k.replace('3','Three').replace('6','Six').replace('12','Twelve')}Mean"] = f"{a[4]:,.0f}"
    print("LEDGER", lab, rows[-2][1:])
tab("t15_ledger.tex", ["Dollars per loan", "Mean before", "Diff.\\ in means", "Local linear", "Gov.\\ placebo"], rows)
# ---------- 6. bandwidth sensitivity ----------
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.2)); hs = list(range(5, 29))
for ax, (y, ttl) in zip(axs, [("fb", "Forbearance"), ("dq_sep20", "Delinquent, Sep 2020"), ("mod_post", "Modified after Apr 2020")]):
    e = [ll(Cv, y, h) for h in hs]; b = np.array([100 * v[0] for v in e]); s = np.array([100 * v[1] for v in e])
    ax.fill_between(hs, b - 1.96 * s, b + 1.96 * s, color=ORANGE, alpha=.18, lw=0); ax.plot(hs, b, color=ORANGE, lw=1.3); ax.axhline(0, color=GREY, lw=.6); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("Bandwidth (days)", fontsize=8)
axs[0].set_ylabel("Jump (pp)"); fig.tight_layout(); save(fig, os.path.join(TEX, "fA3_bandwidth.pdf"))
# ---------- 7. multiple testing (Holm) over the four outcomes ----------
R3 = json.load(open(os.path.join(OUT, "round3b.json"))); fam = ["dq_sep20", "mod_post", "performing", "fc_path"]
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(o): run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1, run)
    return adj
hw = holm(np.array([R3["inf_" + y]["p_wild"] for y in fam])); hp = holm(np.array([R3["inf_" + y]["p_ri_day"] for y in fam]))
rows = [[lab, f"{R3['inf_'+y]['p_wild']:.3f}", f"{hw[i]:.3f}", f"{R3['inf_'+y]['p_ri_day']:.3f}", f"{hp[i]:.3f}"] for i, (y, lab) in enumerate(zip(fam, ["Delinquent, Sep 2020", "Modified after Apr 2020", "Performing, Apr 2021", "Foreclosure path, Apr 2021"]))]
tab("tA4_holm.tex", ["Outcome", "Wild-cluster $p$", "Holm-adjusted", "Permutation $p$ (days)", "Holm-adjusted"], rows)
M.update(holmWdq=f"{hw[0]:.3f}", holmWmod=f"{hw[1]:.3f}", holmPdq=f"{hp[0]:.3f}", holmPmod=f"{hp[1]:.3f}", holmPperf=f"{hp[2]:.3f}")
# ---------- 8. numbers that were typed by hand in the text ----------
d6, d78 = Cv[Cv.inq.dt.normalize() == "2020-04-06"], Cv[Cv.inq.dt.normalize().isin(pd.to_datetime(["2020-04-07", "2020-04-08"]))]
W = Cv[(Cv.inq >= "2020-03-16") & (Cv.inq <= "2020-05-31")]; ng, gr = W[(W.post == 0) & (W.fb == 0)], W[(W.post == 0) & (W.fb == 1)]
M.update(aprSixYes=str(int(d6.fb.sum())), aprSixN=str(len(d6)), aprSevenEightYes=str(int(d78.fb.sum())), aprSevenEightN=str(len(d78)), easyNotDQ=f"{100*ng.dq_feb20.mean():.0f}", easyGrantDQ=f"{100*gr.dq_feb20.mean():.0f}",
         twelvefold=f"{Cv[Cv.post==0].fb.mean()/Cv[Cv.post==1].fb.mean():.0f}")
open(os.path.join(TEX, "numbers_ms.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
json.dump(RES, open(os.path.join(OUT, "extra_analysis_b.json"), "w"), indent=1, default=float)
for k, v in M.items(): print(k, "=", v)
