"""RD tables, figures and macros for the journal paper; corrected memory table + macros for journal."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v"); TJ = os.path.join(HERE, "tex", "v")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
def save(fig, p): fig.savefig(p); fig.savefig(p[:-4] + ".png", dpi=110); plt.close(fig)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); F = pd.read_parquet(os.path.join(OUT, "inq_note_flags.parquet")); D = D.join(F.add_prefix("nf_"))
for c, b in [("fico", [0, 580, 620, 660, 700, 900]), ("ltv", [0, 80, 90, 97, 105, 1000])]: D[c + "_b"] = pd.cut(D[c], b).astype(object).fillna("missing").astype(str)
D["bal_b"] = pd.qcut(D.bal, 5, duplicates="drop").astype(object).fillna("missing").astype(str)
for k in ["docs", "denied", "repay", "incomplete"]: D["m_" + k] = (D["nf_" + k] > 0).astype(float)
COV = " + dq_feb20 + npl + pre_in + pre_out + hard + C(fico_b) + C(ltv_b) + C(bal_b) + tenure"
def ll(d, y, h, cov="", donut=0):
    x = d[(d.r.abs() <= h) & ~((d.r >= -donut) & (d.r < donut))].dropna(subset=[y]).copy().reset_index(drop=True); x["w"] = 1 - x.r.abs() / (h + 1)
    mod = smf.wls(f"{y} ~ post + r + post:r" + cov, x, weights=x.w, missing="drop"); m = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.loc[mod.data.row_labels, "r"])[0]})
    return m.params["post"], m.bse["post"], m.pvalues["post"], int(m.nobs), x.loc[x.post == 0, y].mean()
def dd(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = 1 - x.r.abs() / (h + 1)
    m = smf.wls(f"{y} ~ post*conv + r*conv + post:r + post:r:conv", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post:conv"], m.bse["post:conv"], m.pvalues["post:conv"]
def iv(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]); X = np.column_stack([np.ones(len(x)), x.fb, x.r, x.post * x.r]); Z = np.column_stack([np.ones(len(x)), x.post, x.r, x.post * x.r])
    m = IV2SLS(x[y].values, X, Z).fit(); return m.params[1], m.bse[1], m.pvalues[1]
def dm(d, y, h):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy().reset_index(drop=True)
    m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"]
Cv, Gv = D[D.Gov == 0], D[D.Gov == 1]
def c(t, sc=100, d=1): return f"{sc*t[0]:.{d}f}{star(t[2])}", f"({sc*t[1]:.{d}f})"
def tab(path, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1)); s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(str(v) for v in r) + " \\\\")
    open(os.path.join(TEX, path), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
M = {}
# ---- first stage ----
rows = []
for lab, a, b, e in [(f"Local linear, $h={h}$ days", ll(Cv, "fb", h), ll(Gv, "fb", h), dd(D, "fb", h)) for h in (7, 10, 14, 21)]:
    rows += [[lab, c(a)[0], c(b)[0], c(e)[0], f"{a[3]:,}"], ["", c(a)[1], c(b)[1], c(e)[1], ""]]
a = ll(Cv, "fb", 14, COV); rows += [["$h=14$, with covariates", c(a)[0], "", "", f"{a[3]:,}"], ["", c(a)[1], "", "", ""]]; M["fsAdj"], M["fsAdjSE"] = f"{-100*a[0]:.1f}", f"{100*a[1]:.1f}"
a = ll(Cv, "fb", 14, donut=1); rows += [["$h=14$, excluding 6--7 April", c(a)[0], "", "", f"{a[3]:,}"], ["", c(a)[1], "", "", ""]]; M["fsDonut"] = f"{-100*a[0]:.1f}"
tab("t7_first.tex", ["", "Conventional", "Government-backed", "Difference", "Conv.\\ loans"], rows)
a, b = ll(Cv, "fb", 14), ll(Gv, "fb", 14); M.update(fsMain=f"{-100*a[0]:.1f}", fsMainSE=f"{100*a[1]:.1f}", fsGov=f"{100*b[0]:.1f}", fsGovSE=f"{100*b[1]:.1f}", fsN=f"{a[3]:,}")
a = ll(Cv, "fb", 7); M.update(fsSeven=f"{-100*a[0]:.1f}", fsSevenSE=f"{100*a[1]:.1f}")
# ---- balance ----
BAL = [("dq_feb20", "Delinquent, February 2020", 100), ("npl", "Non-performing pool", 100), ("pre_in", "Prior inbound contact rate", 100), ("pre_out", "Prior outbound contact rate", 100), ("hard", "Prior hardship flag", 100),
       ("fico", "Credit score", 1), ("ltv", "Loan-to-value", 1), ("lbal", "Log balance", 1), ("tenure", "Months at servicer", 1), ("pay_pre", "Months with payment, Jan--Mar 2020", 1)]
rows = []
for y, lab, sc in BAL:
    a, b = ll(Cv, y, 14), ll(Gv, y, 14); dg = 1 if sc == 100 else 2
    rows += [[lab, f"{sc*a[4]:.{dg}f}", c(a, sc, dg)[0], c(b, sc, dg)[0]], ["", "", c(a, sc, dg)[1], c(b, sc, dg)[1]]]
tab("t8_balance.tex", ["", "Mean before (conv.)", "Jump, conventional", "Jump, government-backed"], rows)
n = lambda d, lo, hi: int(((d.r >= lo) & (d.r < hi)).sum())
M.update(densCpre=f"{n(Cv,-14,0):,}", densCpost=f"{n(Cv,0,14):,}", densGpre=f"{n(Gv,-14,0):,}", densGpost=f"{n(Gv,0,14):,}", densC=f"{n(Cv,0,14)/n(Cv,-14,0):.2f}", densG=f"{n(Gv,0,14)/n(Gv,-14,0):.2f}")
# ---- outcomes ----
OUTC = [("dq_sep20", "Delinquent, September 2020 (pp)", 100, 1), ("mod_post", "Modified after April 2020 (pp)", 100, 1), ("dollars_octmar", "Principal and interest paid, Oct 2020--Mar 2021 (\\$)", 1, 0),
        ("pay_octmar", "Months with a payment, Oct 2020--Mar 2021", 1, 2), ("performing", "Performing, April 2021 (pp)", 100, 1), ("fc_path", "Foreclosure path, April 2021 (pp)", 100, 1), ("closed", "Paid off, sold or liquidated by Mar 2021 (pp)", 100, 1)]
rows = []
for y, lab, sc, dg in OUTC:
    a, aa, b, e, v, z = ll(Cv, y, 14), ll(Cv, y, 14, COV), ll(Gv, y, 14), dd(D, y, 14), iv(Cv, y, 14), dm(Cv, y, 14)
    rows += [[lab, f"{sc*a[4]:,.{dg}f}", c(z, sc, dg)[0], c(a, sc, dg)[0], c(aa, sc, dg)[0], c(b, sc, dg)[0], c(e, sc, dg)[0], c(v, sc, dg)[0]], ["", "", c(z, sc, dg)[1], c(a, sc, dg)[1], c(aa, sc, dg)[1], c(b, sc, dg)[1], c(e, sc, dg)[1], c(v, sc, dg)[1]]]
    M[f"o{y.replace('_','').replace('20','').replace('21','')}"] = f"{abs(sc*a[0]):,.{dg}f}"
tab("t9_outcomes.tex", ["Outcome", "Mean before", "Diff.\\ in means", "Local linear", "With covariates", "Gov.\\ placebo", "Difference", "Effect of forbearance (2SLS)"], rows)
for y, k in [("dq_sep20", "dq"), ("mod_post", "mod"), ("dollars_octmar", "dol"), ("performing", "perf"), ("fc_path", "fc")]:
    a, aa, b = ll(Cv, y, 14), ll(Cv, y, 14, COV), ll(Gv, y, 14); sc = 1 if k == "dol" else 100; dg = 0 if k == "dol" else 1; z = dm(Cv, y, 14); M.update({k + "DM": f"{abs(sc*z[0]):,.{dg}f}", k + "DMSE": f"{sc*z[1]:,.{dg}f}"})
    M.update({k + "RF": f"{abs(sc*a[0]):,.{dg}f}", k + "RFSE": f"{sc*a[1]:,.{dg}f}", k + "Adj": f"{abs(sc*aa[0]):,.{dg}f}", k + "AdjSE": f"{sc*aa[1]:,.{dg}f}", k + "Gov": f"{sc*b[0]:,.{dg}f}", k + "GovSE": f"{sc*b[1]:,.{dg}f}", k + "Mean": f"{sc*a[4]:,.{dg}f}"})
# ---- mechanism ----
rows = []
for y, lab, sc in [("m_docs", "Documents or assistance package mentioned", 100), ("m_denied", "Ineligible or denied mentioned", 100), ("m_repay", "Repayment plan or reinstatement mentioned", 100), ("m_incomplete", "Incomplete or missing items mentioned", 100), ("nf_notes60", "Servicer notes in the 60 days after inquiry", 1)]:
    a, b = ll(Cv, y, 14), ll(Gv, y, 14); rows += [[lab, f"{sc*a[4]:.1f}", c(a, sc)[0], c(b, sc)[0]], ["", "", c(a, sc)[1], c(b, sc)[1]]]
    M["mech" + "".join(ch for ch in y.split("_")[1].capitalize() if ch.isalpha())] = f"{sc*a[0]:.1f}"
tab("t10_mech.tex", ["", "Mean before (conv.)", "Jump, conventional", "Jump, government-backed"], rows)
# ---- period split + placebo-date macros ----
R2 = json.load(open(os.path.join(OUT, "round2c.json"))); P = R2["period"]
M.update(cPreN=f"{int(P['c_pre'][0]):,}", cPre=f"{100*P['c_pre'][1]:.1f}", cPostN=f"{int(P['c_post'][0]):,}", cPost=f"{100*P['c_post'][1]:.1f}", gPre=f"{100*P['g_pre'][1]:.1f}", gPost=f"{100*P['g_post'][1]:.1f}",
         cPreDays=f"{Cv[(Cv.post==0)&(Cv.fb==1)].days.median():.0f}", cPostDays=f"{Cv[(Cv.post==1)&(Cv.fb==1)].days.median():.0f}", placeboN=str(len(R2["placebo"])),
         placeboMax=f"{max(abs(v) for d_, v in R2['placebo'] if d_ not in ('04-06','04-07','04-08')):.0f}", respWk=f"{100*R2['resp_pooled + week FE'][0]:.1f}", respWkSE=f"{100*R2['resp_pooled + week FE'][1]:.1f}",
         fbMonthsC=f"{Cv.fb_months.median():.0f}", fbMonthsG=f"{Gv.fb_months.median():.0f}")
open(os.path.join(TEX, "numbers_rd.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())); print("RD macros", len(M))
# ---- figures ----
def binned(d, y): g = d[(d.r.abs() <= 21) & (d.inq.dt.dayofweek < 5)].groupby("r")[y].agg(["mean", "size"]); return g[g["size"] >= 8]
fig, ax = plt.subplots(figsize=(5.8, 3.0))
for d, col, lb in [(Gv, BLUE, "Government-backed"), (Cv, ORANGE, "Conventional")]:
    g = binned(d, "fb"); ax.scatter(g.index, 100 * g["mean"], s=np.clip(g["size"] / 2, 6, 40), color=col, alpha=.85, label=lb, zorder=3)
    for side in (0, 1):
        x = d[(d.r.abs() <= 21) & (d.post == side)]; b = np.polyfit(x.r, x.fb, 1); xs = np.array([-21, -0.5]) if side == 0 else np.array([0, 21]); ax.plot(xs, 100 * (b[0] * xs + b[1]), color=col, lw=1.4)
ax.axvline(-.5, color=GREY, lw=.7, ls=":"); ax.set_xlabel("Day of first hardship inquiry, relative to 7 April 2020"); ax.set_ylabel("Requests ending in forbearance (%)"); ax.set_ylim(-3, 105); ax.legend(frameon=False, fontsize=8, loc="lower left")
save(fig, os.path.join(TEX, "f5_rd_first.pdf"))
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.3))
for ax, (y, ttl, sc) in zip(axs, [("dq_sep20", "Delinquent, Sep 2020 (%)", 100), ("mod_post", "Modified after Apr 2020 (%)", 100), ("m_docs", "Documents mentioned in notes (%)", 100)]):
    x = Cv[Cv.r.abs() <= 21].dropna(subset=[y]).copy(); x["bin"] = (np.floor(x.r / 3) * 3 + 1.5); g = x.groupby("bin")[y].mean(); ax.scatter(g.index, sc * g, s=14, color=ORANGE, zorder=3)
    for side in (0, 1):
        z = x[x.post == side]; b = np.polyfit(z.r, z[y], 1); xs = np.array([-21, -0.5]) if side == 0 else np.array([0, 21]); ax.plot(xs, sc * (b[0] * xs + b[1]), color=ORANGE, lw=1.3)
    ax.axvline(-.5, color=GREY, lw=.7, ls=":"); ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("Days from 7 April", fontsize=8)
fig.tight_layout(); save(fig, os.path.join(TEX, "f6_rd_outcomes.pdf"))
pl = R2["placebo"]; fig, ax = plt.subplots(figsize=(5.6, 2.2)); ax.bar(range(len(pl)), [v for _, v in pl], color=[ORANGE if d_ == "04-07" else GREY for d_, _ in pl], width=.7)
ax.set_xticks(range(0, len(pl), 5)); ax.set_xticklabels([pl[i][0] for i in range(0, len(pl), 5)], fontsize=7); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("Jump in completion (pp)"); ax.set_xlabel("Candidate cutoff date, 2020")
save(fig, os.path.join(TEX, "fA2_placebo.pdf"))
# ---- journal: corrected memory table ----
rows = []
for lab, k in [("No timing control", "mem_baseline"), ("Post-7-April indicator", "mem_+ post-cutoff dummy"), ("Inquiry-week effects", "mem_+ inquiry-week FE"), ("First inquiry before 7 April", "mem_pre-cutoff only"), ("First inquiry from 7 April", "mem_post-cutoff only")]:
    v = R2[k]; rows += [[lab, f"{100*v[0]:.1f}{star(v[2])}", f"({100*v[1]:.1f})", f"{int(v[3]):,}", f"{100*v[4]:.1f}"]]
v = R2["mem_gov_wk"]; rows += ["MID", ["Government-backed, inquiry-week effects", f"{100*v[0]:.1f}{star(v[2])}", f"({100*v[1]:.1f})", f"{int(v[3]):,}", "97.9"]]
s = ["\\begin{tabular}{lcccc}", "\\toprule", "Specification & Hardship marker (pp) & s.e. & Inquiring loans & Granted (\\%) \\\\", "\\midrule"]
for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
open(os.path.join(TJ, "t_memory.tex"), "w").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
JM = {"memBase": f"{abs(100*R2['mem_baseline'][0]):.1f}", "memBaseSE": f"{100*R2['mem_baseline'][1]:.1f}", "memWk": f"{abs(100*R2['mem_+ inquiry-week FE'][0]):.1f}", "memWkSE": f"{100*R2['mem_+ inquiry-week FE'][1]:.1f}",
      "memPre": f"{100*R2['mem_pre-cutoff only'][0]:.1f}", "memPreSE": f"{100*R2['mem_pre-cutoff only'][1]:.1f}", "memPost": f"{abs(100*R2['mem_post-cutoff only'][0]):.1f}", "memPostSE": f"{100*R2['mem_post-cutoff only'][1]:.1f}",
      "memGovWk": f"{100*R2['mem_gov_wk'][0]:.1f}", "memGovWkSE": f"{100*R2['mem_gov_wk'][1]:.1f}", "jcPre": M["cPre"], "jcPost": M["cPost"], "jfs": M["fsMain"]}
open(os.path.join(TJ, "numbers_mem.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in JM.items())); print("done", JM)
