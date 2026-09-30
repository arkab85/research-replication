"""journal round 3 assets: inference table, screening by stated need, targeting, who asks. Writes tex tables + numbers_r3.tex."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v")
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def tab(path, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1)); s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(str(v) for v in r) + " \\\\")
    open(os.path.join(TEX, path), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
M = {}
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet"))
e = pd.read_csv(r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv", dtype=str, low_memory=False,
                usecols=["LoanID", "Liquid.Resources.2.0", "Advantage.Target.Income.3.0", "Advantage.Household.Age..Enhanced."]).drop_duplicates("LoanID").set_index("LoanID")
e.columns = ["liq", "inc", "age"]; e = e.apply(pd.to_numeric, errors="coerce"); D = D.join(e)
D["lowinc"] = np.where(D.inc.isna(), np.nan, (D.inc <= 4).astype(float))
N = pd.read_parquet(os.path.join(OUT, "inq_need_flags.parquet")); NW = pd.read_parquet(os.path.join(OUT, "inq_need_flags_wide.parquet"))
D = D.join(N.add_prefix("nd_"))
D["job"] = np.where(D.nd_n_call_notes > 0, (D.nd_jobloss > 0).astype(float), np.nan)
D["nojob"] = 1 - D.job
Cv, Gv = D[D.Gov == 0].copy(), D[D.Gov == 1].copy(); W = Cv[Cv.r.abs() <= 14]
M.update(needCover=f"{100*W.job.notna().mean():.0f}", needJob=f"{100*W.job.mean():.0f}", needJobPost=f"{100*W[(W.post==1)&(W.fb==0)].job.mean():.0f}", needJobPostN=f"{int(W[(W.post==1)&(W.fb==0)].job.sum()):,}")
def dm(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]).reset_index(drop=True); m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], m.pvalues["post"], len(x)
def inter(d, y, s, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y, s]).reset_index(drop=True); m = smf.ols(f"{y} ~ post*{s}", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); k = f"post:{s}"; return m.params[k], m.bse[k], m.pvalues[k]
c = lambda t, sc=100: (f"{sc*t[0]:.1f}{star(t[2])}", f"({sc*t[1]:.1f})")
# ---- balance of stated need ----
a, g = dm(Cv, "job"), dm(Gv, "job"); M.update(needBal=f"{100*a[0]:.1f}", needBalSE=f"{100*a[1]:.1f}")
Cv["jobw"] = np.where(NW.n_call_notes.reindex(Cv.index) > 0, (NW.jobloss.reindex(Cv.index) > 0).astype(float), np.nan); aw = dm(Cv, "jobw"); M.update(needBalWide=f"{100*aw[0]:.1f}", needBalWideSE=f"{100*aw[1]:.1f}")
# ---- heterogeneity table ----
OUTS = [("fb", "Forbearance"), ("dq_sep20", "Delinquent, Sep 2020"), ("mod_post", "Modified after Apr 2020"), ("performing", "Performing, Apr 2021"), ("fc_path", "Foreclosure path, Apr 2021")]
SPL = [("job", "Said job or income loss", "Did not"), ("dq_feb20", "Delinquent, Feb 2020", "Current"), ("lowinc", "Income below \\$40k (modeled)", "Above")]
hdr = ["Outcome (pp)"]; [hdr.extend([a_, b_, "Difference"]) for _, a_, b_ in SPL]
rows = []
for y, yl in OUTS:
    r1, r2 = [yl], [""]
    for s, _, _ in SPL:
        a1, a0, it = dm(Cv[Cv[s] == 1], y), dm(Cv[Cv[s] == 0], y), inter(Cv, y, s)
        r1 += [c(a1)[0], c(a0)[0], c(it)[0]]; r2 += [c(a1)[1], c(a0)[1], c(it)[1]]
        if s == "job": M[f"job{y.replace('_','').replace('20','').replace('21','')}Yes"] = f"{100*a1[0]:.1f}"; M[f"job{y.replace('_','').replace('20','').replace('21','')}No"] = f"{100*a0[0]:.1f}"; M[f"job{y.replace('_','').replace('20','').replace('21','')}P"] = f"{it[2]:.2f}"
    rows += [r1, r2]
nrow = ["Loans"]; [nrow.extend([f"{int((W[s]==1).sum()):,}", f"{int((W[s]==0).sum()):,}", ""]) for s, _, _ in SPL]; rows += ["MID", nrow]
tab("t12_het.tex", hdr, rows)
# ---- inference table ----
R3 = json.load(open(os.path.join(OUT, "round3b.json"))); rows = []
for y, yl in OUTS:
    v = R3["inf_" + y]; rows.append([yl, f"{100*v['b']:.1f}", f"{v['p_wild']:.3f}", f"{100*v['dm7']:.1f}", f"{v['p_ri_unit']:.3f}", f"{v['p_ri_day']:.3f}"])
    k = y.replace("_", "").replace("20", "").replace("21", ""); M[f"pw{k}"] = f"{v['p_wild']:.3f}"; M[f"pri{k}"] = f"{v['p_ri_day']:.3f}"
tab("t11_infer.tex", ["Outcome", "Local linear jump", "Wild-cluster $p$", "Diff.\\ in means, $\\pm$7 days", "Permutation $p$ (loans)", "Permutation $p$ (days)"], rows)
M["inferClusters"] = str(R3["inf_fb"]["clusters"])
# ---- targeting ----
Wd = Cv[(Cv.inq >= "2020-03-16") & (Cv.inq <= "2020-05-31")]
G = [("Easy regime, granted", Wd[(Wd.post == 0) & (Wd.fb == 1)]), ("Easy regime, not granted", Wd[(Wd.post == 0) & (Wd.fb == 0)]), ("Paperwork regime, granted", Wd[(Wd.post == 1) & (Wd.fb == 1)]), ("Paperwork regime, not granted", Wd[(Wd.post == 1) & (Wd.fb == 0)])]
rows = [[lab, f"{len(d):,}", f"{100*d.dq_feb20.mean():.1f}", f"{100*d.hard.mean():.1f}", f"{d.fico.median():.0f}", f"{100*d.lowinc.mean():.1f}", f"{100*d.job.mean():.1f}", f"{100*d.performing.mean():.1f}", f"{100*d.fc_path.mean():.1f}"] for lab, d in G]
tab("t13_target.tex", ["", "Loans", "DQ Feb-20", "Hardship flag", "FICO", "Low income", "Said job loss", "Performing Apr-21", "FC path Apr-21"], rows)
pg = G[2][1]; M.update(tgtN=f"{len(pg):,}", tgtJob=f"{100*pg.job.mean():.0f}", tgtJobEasy=f"{100*G[0][1].job.mean():.0f}", tgtPerf=f"{100*pg.performing.mean():.0f}", tgtPerfEasy=f"{100*G[0][1].performing.mean():.0f}")
# ---- who asks ----
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); L = L[L.in_covid_file == 1].join(e)
L["inq"] = (L.inq_first.notna() & (L.inq_first <= "2020-09-30")).astype(int); L["hard"] = ((L.pre_unemp.fillna(0) + L.pre_curtail.fillna(0)) > 0).astype(int); L["state"] = L.state.fillna("NA")
L["inc10"] = L.inc; L["liq10"] = L.liq; L["young"] = (L.age <= 3).astype(float).where(L.age.notna()); L["old"] = (L.age >= 6).astype(float).where(L.age.notna())
rows = []; spec = "inq ~ pre_in + pre_out + hard + dq_feb20 + inc10 + liq10 + young + old + C(state)"
res = {}
for lab, d in [("Conventional", L[L.Gov == 0]), ("Government-backed", L[L.Gov == 1])]:
    d = d.dropna(subset=["inc10", "liq10", "young", "pre_in"]).reset_index(drop=True); m = smf.ols(spec, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]}); res[lab] = (m, len(d), d.inq.mean())
for k, nm in [("pre_in", "Prior inbound contact rate (0--1)"), ("pre_out", "Prior outbound contact rate (0--1)"), ("hard", "Prior hardship flag"), ("dq_feb20", "Delinquent, February 2020"), ("inc10", "Income category (1--13, modeled)"), ("liq10", "Liquid savings category (1--11, modeled)"), ("young", "Head of household under 45"), ("old", "Head of household 65 or older")]:
    r1, r2 = [nm], [""]
    for lab in res: m = res[lab][0]; r1.append(f"{100*m.params[k]:.1f}{star(m.pvalues[k])}"); r2.append(f"({100*m.bse[k]:.1f})")
    rows += [r1, r2]
rows += ["MID", ["State effects", "Yes", "Yes"], ["Loans"] + [f"{res[l][1]:,}" for l in res], ["Mean of dependent variable (\\%)"] + [f"{100*res[l][2]:.1f}" for l in res]]
tab("t14_whoasks.tex", ["Hardship inquiry (pp)", "Conventional", "Government-backed"], rows)
m = res["Conventional"][0]; M.update(askInc=f"{100*m.params['inc10']:.1f}", askIncP=f"{m.pvalues['inc10']:.2f}", askResp=f"{100*m.params['pre_in']:.1f}", askDQ=f"{100*m.params['dq_feb20']:.1f}", askOld=f"{100*m.params['old']:.1f}")
open(os.path.join(TEX, "numbers_r3.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
for k, v in M.items(): print(k, "=", v)
for t in ("t12_het", "t13_target", "t14_whoasks"): print("\n==", t); print("\n".join(open(os.path.join(TEX, t + ".tex")).read().split("\n")[3:-2]))
