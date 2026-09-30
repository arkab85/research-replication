"""Does accumulated narrative enter a discretionary decision? FB | inquiry on pre-March-2020 note history.
Conventional = discretion (memory can matter). Government-backed = statutory right (placebo: it cannot)."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v")
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); N = pd.read_parquet(os.path.join(OUT, "v_loan_level.parquet"))
K = L[L.in_covid_file == 1].join(N, how="left")
K["inq"] = (K.inq_first.notna() & (K.inq_first <= "2020-09-30")).astype(int); K["fb"] = (K.fb_agree.notna() & (K.fb_agree <= "2020-09-30")).astype(int)
K = K[K.board < "2020-03-01"].copy()
K["months_hist"] = (pd.Timestamp("2020-03-01") - K.board).dt.days / 30.44
for c in ["notes_pre", "mk_employment", "mk_health", "mk_family"]: K[c] = K[c].fillna(0)
K["lnotes"] = np.log1p(K.notes_pre); K["emp"] = (K.mk_employment > 0).astype(int); K["health"] = (K.mk_health > 0).astype(int); K["family"] = (K.mk_family > 0).astype(int)
K["anyhard"] = ((K.emp + K.health + K.family) > 0).astype(int); K["nhard"] = np.log1p(K.mk_employment + K.mk_health + K.mk_family)
K["state"] = K.state.fillna("NA"); K["fico_b"] = pd.cut(K.fico, [0, 580, 620, 660, 700, 900]).astype(object).fillna("missing").astype(str); K["ltv_b"] = pd.cut(K.ltv, [0, 80, 90, 97, 105, 1000]).astype(object).fillna("missing").astype(str)
K["bal_b"] = pd.qcut(K.bal, 5, duplicates="drop").astype(object).fillna("missing").astype(str); K["npl"] = K.AssetType.eq("NPL").astype(int)
def fit(f, d):
    d = d.reset_index(drop=True); m = smf.ols(f, d, missing="drop"); return m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[m.data.row_labels, "state"])[0]})
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
HARD = " + C(status_feb20) + npl + C(fico_b) + C(ltv_b) + C(bal_b) + C(state)"
CONT = " + pre_in + pre_out"
res = {}; rows = []
samples = [("Conventional (discretion)", K[(K.Gov == 0) & (K.inq == 1)]), ("Government-backed (right; placebo)", K[(K.Gov == 1) & (K.inq == 1)])]
for lab, d in samples:
    res[lab] = {"n": len(d), "fb": d.fb.mean(), "anyhard": d.anyhard.mean(), "notes_med": d.notes_pre.median(), "hist_med": d.months_hist.median()}
    print(lab, res[lab])
    for spec, f in [("A", "fb ~ anyhard + lnotes + months_hist" + HARD), ("B", "fb ~ emp + health + family + lnotes + months_hist" + HARD + CONT)]:
        m = fit(f, d); res[lab][spec] = {k: (m.params[k], m.bse[k], m.pvalues[k]) for k in m.params.index if k in ("anyhard", "emp", "health", "family", "lnotes", "months_hist", "pre_in", "pre_out")}
        res[lab][spec]["N"] = int(m.nobs); res[lab][spec]["r2"] = m.rsquared
        print(" ", spec, {k: f"{100*v[0]:.1f} ({100*v[1]:.1f}){star(v[2])}" for k, v in res[lab][spec].items() if isinstance(v, tuple)}, "N", int(m.nobs))
# by-marker raw conversion among conventional inquirers
c = samples[0][1]
print("conv by anyhard:", c.groupby("anyhard").fb.agg(["mean", "size"]).round(3).to_dict())
print("conv by notes tercile:", c.groupby(pd.qcut(c.notes_pre, 3, duplicates="drop"), observed=True).fb.agg(["mean", "size"]).round(3))
res["raw_anyhard"] = c.groupby("anyhard").fb.agg(["mean", "size"]).to_dict()
res["raw_notes_terc"] = [(str(k), float(v["mean"]), int(v["size"])) for k, v in c.groupby(pd.qcut(c.notes_pre, 3, duplicates="drop"), observed=True).fb.agg(["mean", "size"]).iterrows()]
# also: does history predict ASKING (household side)?
for lab, d in [("Conventional", K[K.Gov == 0]), ("Government-backed", K[K.Gov == 1])]:
    m = fit("inq ~ anyhard + lnotes + months_hist" + HARD + CONT, d); res["ask_" + lab] = {k: (m.params[k], m.bse[k], m.pvalues[k]) for k in ("anyhard", "lnotes", "months_hist")}; res["ask_" + lab]["N"] = int(m.nobs)
    print("ASK", lab, {k: f"{100*v[0]:.1f} ({100*v[1]:.1f}){star(v[2])}" for k, v in res["ask_" + lab].items() if isinstance(v, tuple)})
# where do marked conventional inquirers go instead?
K["mod_post"] = (K.moddate >= "2020-03-01").astype(int); K["perf"] = K.disp.eq("Performing").astype(int); K["fcpath"] = K.disp.isin(["Pending Foreclosure Completion", "REO"]).astype(int)
ci = K[(K.Gov == 0) & (K.inq == 1)]
res["alt"] = {}
for y in ["mod_post", "perf", "fcpath"]:
    m = fit(f"{y} ~ anyhard + lnotes + months_hist" + HARD, ci); res["alt"][y] = (m.params["anyhard"], m.bse["anyhard"], m.pvalues["anyhard"], ci[y].mean())
    print("ALT", y, f"{100*m.params['anyhard']:.1f} ({100*m.bse['anyhard']:.1f}){star(m.pvalues['anyhard'])}  mean {100*ci[y].mean():.1f}")
# robustness: marker intensity, and dropping the soft-information-derived controls entirely / adding them
for nm, f in [("intensity", "fb ~ nhard + lnotes + months_hist" + HARD), ("no_controls", "fb ~ anyhard"), ("within_current", "fb ~ anyhard + lnotes + months_hist" + HARD)]:
    d = ci[ci.dq_feb20 == 0] if nm == "within_current" else ci
    m = fit(f, d); k = "nhard" if nm == "intensity" else "anyhard"; res["rob_" + nm] = (m.params[k], m.bse[k], m.pvalues[k], int(m.nobs))
    print("ROB", nm, f"{100*m.params[k]:.1f} ({100*m.bse[k]:.1f}){star(m.pvalues[k])} N {int(m.nobs)}")
# LaTeX table
def c2(r, k):
    if k not in r: return "", ""
    b, se, p = r[k]; return f"{100*b:.1f}{star(p)}", f"({100*se:.1f})"
names = [("anyhard", "Any hardship marker in prior notes"), ("emp", "\\quad Employment marker"), ("health", "\\quad Health marker"), ("family", "\\quad Family marker"),
         ("lnotes", "Log (1 + prior servicer notes)"), ("months_hist", "Months since boarding"), ("pre_in", "Prior inbound contact rate"), ("pre_out", "Prior outbound contact rate")]
cols = [res[s][sp] for s, _ in samples for sp in ("A", "B")]
lines = ["\\begin{tabular}{lcccc}", "\\toprule", " & \\multicolumn{2}{c}{Conventional (discretion)} & \\multicolumn{2}{c}{Government-backed (right)} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", " & (1) & (2) & (3) & (4) \\\\", "\\midrule"]
for k, nm in names:
    a = [c2(r, k) for r in cols]
    if not any(x[0] for x in a): continue
    lines.append(nm + " & " + " & ".join(x[0] for x in a) + " \\\\"); lines.append(" & " + " & ".join(x[1] for x in a) + " \\\\")
lines += ["\\midrule", "Hard-data controls and state effects & Yes & Yes & Yes & Yes \\\\", "Inquiring loans & " + " & ".join(f"{r['N']:,}" for r in cols) + " \\\\",
          "Mean of dependent variable (\\%) & " + " & ".join(f"{100*res[s]['fb']:.1f}" for s, _ in samples for _ in (0, 1)) + " \\\\", "\\bottomrule", "\\end{tabular}"]
open(os.path.join(TEX, "t_memory.tex"), "w").write("\n".join(lines))
json.dump(res, open(os.path.join(OUT, "memory_analysis.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
