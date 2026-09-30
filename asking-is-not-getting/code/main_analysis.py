"""journal rewrite: all tables (LaTeX fragments) and figures (PDF) from the loan frame + monthly panel."""
import os, json, time, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "out"); TEX = os.path.join(HERE, "tex", "v"); os.makedirs(TEX, exist_ok=True)
T0 = time.time()
def log(*a): print("[%4.0fs]" % (time.time()-T0), *a, flush=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .6, "figure.dpi": 150, "savefig.bbox": "tight"})
def save(fig, path): fig.savefig(path); fig.savefig(path[:-4] + ".png", dpi=110)
BLUE, ORANGE, GREY = "#1f5fa8", "#c8551f", "#777777"
R = {}   # every number quoted in the text

L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
L["state"] = L["state"].fillna("NA"); L["Investor"] = L["Investor"].astype(str).fillna("NA")
L["npl"] = L["AssetType"].eq("NPL").astype(int)
L["inq"] = L["inq_first"].notna().astype(int)
L["fb"] = (L["fb_agree"].notna() & (L["fb_agree"] <= "2020-09-30")).astype(int)
L["fb_ever"] = L["fb_agree"].notna().astype(int)
L["inq_sep"] = (L["inq_first"].notna() & (L["inq_first"] <= "2020-09-30")).astype(int)
L["days"] = (L["fb_agree"] - L["inq_first"]).dt.days
L["hard_pre"] = ((L["pre_unemp"].fillna(0) + L["pre_curtail"].fillna(0)) > 0).astype(int)
for c, bins in [("fico", [0, 580, 620, 660, 700, 900]), ("ltv", [0, 80, 90, 97, 105, 1000])]:
    L[c + "_b"] = pd.cut(L[c], bins).astype(object).fillna("missing").astype(str)
L["bal_b"] = pd.qcut(L["bal"], 5, duplicates="drop").astype(object).fillna("missing").astype(str)
K = L[L["in_covid_file"] == 1].copy()          # linked sample (primary)
R["n_frame"], R["n_linked"] = len(L), len(K)
R["n_gov"], R["n_conv"] = int(K.Gov.sum()), int((1-K.Gov).sum())
R["link_gov"], R["link_conv"] = L[L.Gov==1].in_covid_file.mean(), L[L.Gov==0].in_covid_file.mean()

def tex_table(path, header, rows, align=None, note=None):
    align = align or ("l" + "c" * (len(header) - 1))
    s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        s.append("\\midrule" if r == "MID" else " & ".join(str(x) for x in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TEX, path), "w", encoding="utf-8").write("\n".join(s))
def pct(x, d=1): return "--" if pd.isna(x) else f"{100*x:.{d}f}"
def num(x, d=2): return "--" if pd.isna(x) else f"{x:,.{d}f}"
def star(p): return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def fit(formula, d, cl="state"):
    d = d.dropna(subset=[cl]).reset_index(drop=True)
    mod = smf.ols(formula, d, missing="drop")
    return mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[mod.data.row_labels, cl])[0]})
def cell(m, k, scale=100, d=1): return f"{scale*m.params[k]:.{d}f}{star(m.pvalues[k])}", f"({scale*m.bse[k]:.{d}f})"

# ---------------- Table 1: sample ----------------
log("T1")
def desc(d):
    return [f"{len(d):,}", pct(d.dq_feb20.mean()), pct(d.npl.mean()), pct(d.pre_in.mean()), pct(d.pre_out.mean()), pct(d.hard_pre.mean()),
            num(d.fico.median(), 0), num(d.bal.median()/1000, 0), num(d.ltv.median(), 0), num(d.pre_months.median(), 0)]
hdr = ["", "Loans", "DQ Feb-20", "NPL pool", "Inbound", "Outbound", "Hardship flag", "FICO", "Balance (\\$k)", "LTV", "Pre months"]
rows = [["Government-backed"] + desc(K[K.Gov == 1]), ["\\quad FHA"] + desc(K[K.loan_type == "FHA"]), ["\\quad VA"] + desc(K[K.loan_type == "VA"]),
        ["\\quad USDA"] + desc(K[K.loan_type == "USDA"]), ["Conventional"] + desc(K[K.Gov == 0]), "MID",
        ["All Feb-2020 active, typed"] + desc(L), ["\\quad not linked to relief file"] + desc(L[L.in_covid_file == 0])]
tex_table("t1_sample.tex", hdr, rows)

# ---------------- Table 2: funnel ----------------
log("T2 funnel")
def funnel(d):
    i = d[d.inq_sep == 1]
    return [f"{len(d):,}", pct(d.inq_sep.mean()), pct(i.fb.mean()), pct(d.fb.mean()),
            num(i.loc[i.fb == 1, "days"].median(), 0), pct((i.loc[i.fb == 1, "days"] <= 7).mean()), pct(d.optout.notna().mean())]
hdr = ["", "Loans", "Inquiry", "Forbearance $|$ inquiry", "Forbearance", "Median days", "Within 7 days", "Opt-out"]
rows = []
for lab, d in [("Government-backed", K[K.Gov == 1]), ("Conventional", K[K.Gov == 0])]: rows.append([lab] + funnel(d))
rows.append("MID")
for st, sl in [(0, "Current Feb-20"), (1, "Delinquent Feb-20")]:
    for g, gl in [(1, "Gov"), (0, "Conv")]: rows.append([f"{sl}, {gl}"] + funnel(K[(K.Gov == g) & (K.dq_feb20 == st)]))
rows.append("MID")
for t in ["FHA", "VA", "USDA"]: rows.append([t] + funnel(K[K.loan_type == t]))
tex_table("t2_funnel.tex", hdr, rows)
g, c = K[K.Gov == 1], K[K.Gov == 0]
pi_g, pi_c = g.inq_sep.mean(), c.inq_sep.mean(); cv_g, cv_c = g[g.inq_sep == 1].fb.mean(), c[c.inq_sep == 1].fb.mean()
R.update(inq_g=pi_g, inq_c=pi_c, conv_g=cv_g, conv_c=cv_c, fb_g=g.fb.mean(), fb_c=c.fb.mean(),
         gap=g.fb.mean()-c.fb.mean(), part_inq=(pi_g-pi_c)*(cv_g+cv_c)/2, part_conv=(cv_g-cv_c)*(pi_g+pi_c)/2,
         fb_noinq=K[K.inq_sep == 0].fb.mean(), days_g=g.loc[(g.inq_sep==1)&(g.fb==1), "days"].median(), days_c=c.loc[(c.inq_sep==1)&(c.fb==1), "days"].median())
# counterfactual: CONV inquiry rate with Gov conversion, and vice versa
R["cf_conv_with_govconv"] = pi_c * cv_g; R["cf_gov_with_convconv"] = pi_g * cv_c

# ---------------- Table 3: margins with controls ----------------
log("T3 regressions")
K["board_year"] = K["board"].dt.year.fillna(0).astype(int).astype(str)
X1 = "Gov"; X2 = X1 + " + C(state) + C(status_feb20) + npl"; X3 = X2 + " + C(board_year) + hard_pre + pre_in + pre_out + C(fico_b) + C(ltv_b) + C(bal_b)"
I = K[K.inq_sep == 1]
rows, mods = [], {}
for lab, y, d in [("Inquiry", "inq_sep", K), ("Forbearance $|$ inquiry", "fb", I), ("Forbearance (unconditional)", "fb", K)]:
    cs = []
    for j, X in enumerate([X1, X2, X3]):
        m = fit(f"{y} ~ {X}", d); mods[(lab, j)] = m; cs.append(cell(m, "Gov"))
    rows.append([lab] + [x[0] for x in cs] + [f"{int(m.nobs):,}"]); rows.append([""] + [x[1] for x in cs] + [""])
rows += ["MID", ["State, Feb-20 status, NPL indicator", "", "Yes", "Yes", ""], ["Boarding year, hardship, contact history, FICO/LTV/balance bins", "", "", "Yes", ""]]
tex_table("t3_margins.tex", ["Outcome (pp)", "(1)", "(2)", "(3)", "Loans"], rows)
# pool structure (appendix) and within-cell stability of the conversion gap
dl = pd.read_parquet(os.path.join(OUT, "loan_deal.parquet")); K["deal"] = dl["DealName"].reindex(K.index).fillna("(none)")
# ANONYMISED: the investor's internal deal names are replaced by generic labels before anything is written out.
_d = K.deal.where(K.deal.str.startswith("EBO"), "PL"); _o = _d.value_counts().index
_lab = {}; _ne = _ni = 0
for _k in _o:
    if str(_k).startswith("EBO"): _lab[_k] = "Early-buyout pool " + chr(65 + _ne); _ne += 1
    else: _lab[_k] = "Private-label fund " + chr(65 + _ni); _ni += 1
K["pool"] = _d.map(_lab)
rows = []
for p, d in sorted(K.groupby("pool"), key=lambda x: -len(x[1])):
    if len(d) < 50: continue
    i = d[d.inq_sep == 1]; rows.append([p.replace("&", "\\&"), f"{len(d):,}", pct(d.Gov.mean(), 0), pct(d.dq_feb20.mean()), pct(d.inq_sep.mean()), pct(i.fb.mean()) if len(i) >= 10 else "--", f"{len(i):,}"])
tex_table("tA1_pools.tex", ["Pool", "Loans", "Gov share", "DQ Feb-20", "Inquiry", "Forbearance $|$ inquiry", "Inquiries"], rows)
cells = []
for nm, col in [("state", "state"), ("Feb-20 status", "status_feb20"), ("boarding year", "board_year")]:
    for v, d in I.groupby(col):
        a, b = d[d.Gov == 1], d[d.Gov == 0]
        if len(a) >= 20 and len(b) >= 20: cells.append((nm, v, a.fb.mean(), b.fb.mean(), len(a), len(b)))
cd = pd.DataFrame(cells, columns=["dim", "cell", "g", "c", "ng", "nc"]); cd["gap"] = cd.g - cd.c
R["cells_n"] = len(cd); R["cells_gap_min"] = cd.gap.min(); R["cells_gap_max"] = cd.gap.max(); R["cells_g_min"] = cd.g.min(); R["cells_c_max"] = cd.c.max()
R["cells_states"] = int((cd.dim == "state").sum()); cd.to_csv(os.path.join(OUT, "v_cells.csv"), index=False)
R["conv_gap_ctrl"] = mods[("Forbearance $|$ inquiry", 2)].params["Gov"]; R["conv_gap_ctrl_se"] = mods[("Forbearance $|$ inquiry", 2)].bse["Gov"]
R["inq_gap_ctrl"] = mods[("Inquiry", 2)].params["Gov"]; R["inq_gap_ctrl_se"] = mods[("Inquiry", 2)].bse["Gov"]
R["fb_gap_ctrl"] = mods[("Forbearance (unconditional)", 2)].params["Gov"]; R["fb_gap_ctrl_se"] = mods[("Forbearance (unconditional)", 2)].bse["Gov"]

# ---------------- Table 4: responsiveness (Proposition 2 test) ----------------
log("T4 responsiveness")
H = K[(K.pre_months >= 6) & (K.pre_out_n >= 1)].copy()        # contacted at least once pre-2020, observed >= 6 months
pos = H.loc[H.pre_in_n > 0, "pre_in"]
med = pos.median()
H["resp"] = np.where(H.pre_in_n == 0, "Never", np.where(H.pre_in <= med, "Low", "High"))
H["resp"] = pd.Categorical(H["resp"], ["Never", "Low", "High"], ordered=True)
R["n_resp"] = len(H); R["resp_share"] = H.resp.value_counts(normalize=True).to_dict()
rows = []
for r in ["Never", "Low", "High"]:
    a, b = H[(H.resp == r) & (H.Gov == 1)], H[(H.resp == r) & (H.Gov == 0)]
    ai, bi = a[a.inq_sep == 1], b[b.inq_sep == 1]
    rows.append([r, f"{len(a):,}", f"{len(b):,}", pct(a.inq_sep.mean()), pct(b.inq_sep.mean()), pct(ai.fb.mean()), pct(bi.fb.mean()),
                 f"{len(ai):,}", f"{len(bi):,}", pct(a.fb.mean()), pct(b.fb.mean())])
    R[f"conv_{r}_g"], R[f"conv_{r}_c"] = ai.fb.mean(), bi.fb.mean(); R[f"inq_{r}_g"], R[f"inq_{r}_c"] = a.inq_sep.mean(), b.inq_sep.mean()
grp = (" & \\multicolumn{2}{c}{Loans} & \\multicolumn{2}{c}{Inquiry (\\%)} & \\multicolumn{2}{c}{Forbearance $|$ inquiry (\\%)} & \\multicolumn{2}{c}{Inquiries} & \\multicolumn{2}{c}{Forbearance (\\%)} \\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\nPrior responsiveness")
tex_table("t4_resp.tex", [grp, "Gov", "Conv", "Gov", "Conv", "Gov", "Conv", "Gov", "Conv", "Gov", "Conv"], rows)
H["never"] = (H.resp == "Never").astype(int); H["high"] = (H.resp == "High").astype(int)
HI = H[H.inq_sep == 1]
Xc = " + C(state) + C(status_feb20) + npl + hard_pre + pre_out + C(fico_b) + C(ltv_b)"
rows = []
for lab, y, d in [("Inquiry", "inq_sep", H), ("Forbearance $|$ inquiry", "fb", HI), ("Forbearance", "fb", H)]:
    m = fit(f"{y} ~ Gov*pre_in" + Xc, d); R[f"slope_{y}_{len(d)}"] = (m.params["pre_in"], m.params["Gov:pre_in"])
    a, b, cc = cell(m, "Gov"), cell(m, "pre_in"), cell(m, "Gov:pre_in")
    rows.append([lab, a[0], b[0], cc[0], f"{int(m.nobs):,}"]); rows.append(["", a[1], b[1], cc[1], ""])
    if lab.startswith("Forbearance $"): R["resp_slope_conv"], R["resp_slope_int"], R["resp_slope_int_p"] = m.params["pre_in"], m.params["Gov:pre_in"], m.pvalues["Gov:pre_in"]; R["resp_slope_conv_p"] = m.pvalues["pre_in"]
tex_table("t4b_resp_reg.tex", ["Outcome (pp)", "Gov", "Responsiveness", "Gov $\\times$ Responsiveness", "Loans"], rows)

# ---------------- Table 5 + Fig 1: event study, linked sample, state clusters ----------------
log("T5 event study (linked)")
ib = pd.read_csv(r"<DATA>/panel\IB_OB_Flags_Full_Nov3.csv", usecols=["LoanId", "Month t", "LoanStatus"], dtype={"LoanId": str}).rename(columns={"LoanId": "LoanID"})
ib["month"] = pd.to_datetime(ib["Month t"], errors="coerce").dt.to_period("M")
ib = ib.dropna(subset=["month"]).drop_duplicates(["LoanID", "month"])
ib = ib[~ib.LoanStatus.isin(["Pending Servicing Transfer", "Not on Servicer File"])]
ib = ib[(ib.month >= "2019-03") & (ib.month <= "2020-09")]
def panel(frame):
    p = ib.join(frame[["Gov", "state", "fb_start", "fb_end", "dq_feb20"]], on="LoanID", how="inner")
    ms, me = p.month.dt.start_time, p.month.dt.end_time
    p["fbm"] = ((p.fb_start <= me) & (p.fb_end >= ms)).fillna(False).astype(float); return p
def es(p, cl="state"):
    REF = pd.Period("2020-02", "M"); months = sorted(p.month.unique()); post = [m for m in months if m > REF]
    X = pd.DataFrame(index=p.index)
    for m in months[1:]: X["m_" + str(m)] = (p.month == m).astype(float)
    for m in post: X["e_" + str(m)] = ((p.month == m) & (p.Gov == 1)).astype(float)
    Xd = X - X.groupby(p.LoanID).transform("mean"); yd = p.fbm - p.fbm.groupby(p.LoanID).transform("mean")
    r = sm.OLS(yd.values, Xd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(p[cl])[0]})
    ix = {n: i for i, n in enumerate(X.columns)}
    return pd.DataFrame([{"month": str(m), "b": r.params[ix["e_" + str(m)]], "se": r.bse[ix["e_" + str(m)]]} for m in post])
def did(p, cl="state"):
    d = p[p.month != pd.Period("2020-03", "M")].copy(); d["gp"] = d.Gov * (d.month >= "2020-04")
    X = pd.get_dummies(d.month.astype(str), drop_first=True, dtype=float); X["gp"] = d.gp.values
    Xd = X - X.groupby(d.LoanID.values).transform("mean"); yd = d.fbm - d.fbm.groupby(d.LoanID.values).transform("mean")
    r = sm.OLS(yd.values, Xd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cl])[0]})
    j = list(X.columns).index("gp"); return r.params[j], r.bse[j], d.LoanID.nunique(), len(d)
pk, pl = panel(K), panel(L)
E = es(pk); E.to_csv(os.path.join(OUT, "v_event_linked.csv"), index=False)
specs = [("Linked loans (primary)", pk), ("All typed loans, unlinked coded zero", pl), ("Linked, current in Feb-2020", pk[pk.dq_feb20 == 0]),
         ("Linked, delinquent in Feb-2020", pk[pk.dq_feb20 == 1])]
rows = []
for lab, p in specs:
    b, se, nl, n = did(p); b2, se2, _, _ = did(p, "LoanID")
    rows.append([lab, f"{100*b:.1f}", f"({100*se:.1f})", f"({100*se2:.1f})", f"{nl:,}", f"{n:,}"]); R["did_" + lab[:12]] = (b, se)
tex_table("t5_did.tex", ["Sample", "Gov $\\times$ Post (pp)", "SE state", "SE loan", "Loans", "Loan-months"], rows, align="lccccc")
R["es_last"] = (E.iloc[-1].b, E.iloc[-1].se, E.iloc[-1].month)
fig, ax = plt.subplots(figsize=(5.6, 2.9))
x = np.arange(len(E) + 2); b = np.r_[0, 0, E.b.values] * 100; s = np.r_[0, 0, E.se.values] * 100
ax.fill_between(x, b - 1.96*s, b + 1.96*s, color=BLUE, alpha=.15, lw=0); ax.plot(x, b, color=BLUE, lw=1.6, marker="o", ms=3.5)
ax.axhline(0, color=GREY, lw=.6); ax.axvline(1.5, color=GREY, lw=.6, ls=":"); ax.text(1.6, b.max()*.95, "CARES Act", fontsize=8, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(["Jan", "Feb"] + [pd.Period(m).strftime("%b") for m in E.month]); ax.set_ylabel("Gov $-$ Conv, pp (Feb 2020 = 0)"); ax.set_xlabel("2020")
save(fig, os.path.join(TEX, "f1_event.pdf")); plt.close(fig)

# ---------------- Fig 2: funnel; Fig 3: conversion by responsiveness; Fig 4: timing ----------------
log("figures")
fig, ax = plt.subplots(figsize=(5.6, 2.6)); w = .36; xs = np.arange(3)
ax.bar(xs - w/2, [100*pi_g, 100*cv_g, 100*g.fb.mean()], w, color=BLUE, label="Government-backed"); ax.bar(xs + w/2, [100*pi_c, 100*cv_c, 100*c.fb.mean()], w, color=ORANGE, label="Conventional")
for i, (a_, b_) in enumerate([(pi_g, pi_c), (cv_g, cv_c), (g.fb.mean(), c.fb.mean())]):
    ax.text(i - w/2, 100*a_ + 1.5, f"{100*a_:.0f}", ha="center", fontsize=8); ax.text(i + w/2, 100*b_ + 1.5, f"{100*b_:.0f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(["Inquiry\n(asks)", "Forbearance | inquiry\n(completes)", "Forbearance\n(unconditional)"]); ax.set_ylabel("% of loans"); ax.set_ylim(0, 110); ax.legend(frameon=False, fontsize=8)
save(fig, os.path.join(TEX, "f2_funnel.pdf")); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=False)
for ax, (y, ttl, d0) in zip(axs, [("inq_sep", "Inquiry", H), ("fb", "Forbearance | inquiry", HI)]):
    for gv, col, lb in [(1, BLUE, "Government-backed"), (0, ORANGE, "Conventional")]:
        mm = d0[d0.Gov == gv].groupby("resp", observed=True)[y].agg(["mean", "count"]); se_ = np.sqrt(mm["mean"]*(1-mm["mean"])/mm["count"])
        ax.errorbar(range(3), 100*mm["mean"], yerr=196*se_, color=col, marker="o", ms=4, lw=1.4, capsize=2, label=lb)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Never", "Low", "High"]); ax.set_title(ttl, fontsize=9); ax.set_xlabel("Pre-2020 responsiveness"); ax.set_ylim(0, 105)
axs[0].set_ylabel("% of loans"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, os.path.join(TEX, "f3_resp.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 2.6))
for gv, col, lb in [(1, BLUE, "Government-backed"), (0, ORANGE, "Conventional")]:
    d = K[(K.Gov == gv) & (K.inq_sep == 1)]; t = np.arange(0, 121)
    ax.step(t, [100*((d.fb == 1) & (d.days <= k)).mean() for k in t], color=col, lw=1.5, label=lb, where="post")
ax.set_xlabel("Days since first COVID inquiry"); ax.set_ylabel("% in forbearance"); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8)
save(fig, os.path.join(TEX, "f4_timing.pdf")); plt.close(fig)

# ---------------- Table 6: what happened next (descriptive) ----------------
log("T6")
F = K[K.fb == 1]
def after(d): return [f"{len(d):,}", pct(d.disp.eq("Performing").mean()), pct(d.dq_apr21.isin(["0-29"]).mean()), pct((d.moddate >= "2020-03-01").mean()),
                      pct(d.disp.isin(["Pending Foreclosure Completion", "REO"]).mean()), pct(d.fb_end.ge("2021-03-01").mean())]
rows = [["Forbearance, Gov"] + after(F[F.Gov == 1]), ["Forbearance, Conv"] + after(F[F.Gov == 0]), "MID",
        ["Inquiry, no forbearance, Conv"] + after(K[(K.inq_sep == 1) & (K.fb == 0) & (K.Gov == 0)]),
        ["No inquiry, Gov"] + after(K[(K.inq_sep == 0) & (K.Gov == 1)]), ["No inquiry, Conv"] + after(K[(K.inq_sep == 0) & (K.Gov == 0)])]
tex_table("t6_after.tex", ["Group", "Loans", "Performing", "0--29 days", "Modified after Mar-20", "Foreclosure path", "Still in forbearance"], rows)
d = K[(K.inq_sep == 1) & (K.fb == 0) & (K.Gov == 0)]
R["denied_n"] = len(d); R["denied_perf"] = d.disp.eq("Performing").mean(); R["denied_fc"] = d.disp.isin(["Pending Foreclosure Completion", "REO"]).mean()

# ---------------- Appendix: pre-trend failure on behavioral outcomes ----------------
cj = json.load(open(os.path.join(OUT, "cares_results.json")))
fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.4))
for ax, (k, ttl) in zip(axs, [("extended:dq30", "30+ day delinquency"), ("extended:inbound", "Inbound contact")]):
    e = pd.DataFrame(cj["event"][k]); xx = np.arange(len(e)); ax.fill_between(xx, 100*(e.beta-1.96*e.se), 100*(e.beta+1.96*e.se), color=GREY, alpha=.2, lw=0)
    ax.plot(xx, 100*e.beta, color="k", lw=1.1); ax.axhline(0, color=GREY, lw=.6); ax.axvline(list(e.month).index("2020-03") - .5, color=GREY, lw=.6, ls=":")
    ax.set_xticks(xx[::6]); ax.set_xticklabels([m[2:] for m in e.month[::6]], fontsize=7); ax.set_title(ttl, fontsize=9)
axs[0].set_ylabel("Gov $-$ Conv, pp"); save(fig, os.path.join(TEX, "fA1_pretrend.pdf")); plt.close(fig)
R["did_dq30"] = cj["did"]["extended:dq30"]["did"]; R["did_inbound"] = cj["did"]["extended:inbound"]["did"]
R["pre_gov_dq"] = cj["did"]["extended:dq30"]["pre_mean_gov"]; R["pre_conv_dq"] = cj["did"]["extended:dq30"]["pre_mean_conv"]

json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in R.items()}, open(os.path.join(OUT, "v_numbers.json"), "w"), indent=1, default=float)
for k, v in R.items(): print(k, "=", v)
log("DONE")
