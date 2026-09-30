"""Short paper: exhibits and numbers. Fig1 tenure; Fig2 windows at the process change; Table1 spurious regression + accounting;
Fig3 markers around modifications; Table2 what predicts a marker. Writes tex/sp/*."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "sp"); os.makedirs(TEX, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#c8551f", "#2f8f5b", "#777777"
def save(fig, p): fig.savefig(p); fig.savefig(p[:-4] + ".png", dpi=110); plt.close(fig)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def tab(path, header, rows, align=None):
    align = align or ("l" + "c" * (len(header) - 1)); s = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: s.append("\\midrule" if r == "MID" else " & ".join(str(v) for v in r) + " \\\\")
    open(os.path.join(TEX, path), "w", encoding="utf-8").write("\n".join(s + ["\\bottomrule", "\\end{tabular}"]))
M = {}
# ---------------- Fig 1: the record accumulates with exposure ----------------
RL = json.load(open(os.path.join(OUT, "relearn_analysis.json"))); F = pd.read_parquet(os.path.join(OUT, "v_first_marker.parquet"))
for c in ["board", "note", "employment", "health", "family"]: F[c] = pd.to_datetime(F[c], errors="coerce")
END = pd.Timestamp("2020-02-29"); F = F[(F.board >= "2016-01-01") & (F.board <= "2019-06-30") & F.note.notna()].copy(); F["any"] = F[["employment", "health", "family"]].min(axis=1); F["fu"] = (END - F.board).dt.days / 30.44
t = (F["any"] - F.board).dt.days / 30.44; ev = t.notna() & (F["any"] <= END); dur = np.clip(np.where(ev, t, F.fu), 0, None); Hz, Cu, s_ = [], [], 1.0
for m in range(30):
    at = (dur >= m).sum(); d = (ev & (dur >= m) & (dur < m + 1)).sum(); h = d / at; Hz.append(h); s_ *= 1 - h; Cu.append(1 - s_)
late = float(np.mean(Hz[12:24])); flat = [1 - (1 - late) ** (m + 1) for m in range(30)]
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5)); mm = np.arange(1, 31)
axs[0].plot(mm, 100 * np.array(Hz), color=BLUE, lw=1.5); axs[0].axhline(100 * late, color=GREY, lw=.8, ls="--"); axs[0].text(17, 100 * late + .5, "second-year rate", fontsize=7.5, color=GREY); axs[0].set_title("Monthly hazard of a first marker (%)", fontsize=9)
axs[1].plot(mm, 100 * np.array(Cu), color=BLUE, lw=1.5, label="Observed"); axs[1].plot(mm, 100 * np.array(flat), color=GREY, lw=1, ls="--", label="If markers arrived at the\nsecond-year rate"); axs[1].set_title("Loans with any marker (%)", fontsize=9); axs[1].legend(frameon=False, fontsize=7.5, loc="lower right")
for ax in axs: ax.set_xlabel("Months since the loan boarded with the servicer")
fig.tight_layout(); save(fig, os.path.join(TEX, "f1_tenure.pdf"))
M.update(tenN=f"{len(F):,}", hazEarly=f"{100*np.mean(Hz[1:3]):.1f}", hazLate=f"{100*late:.1f}", hazRatio=f"{np.mean(Hz[1:3])/late:.0f}", cumTwelve=f"{100*Cu[11]:.0f}", flatTwelve=f"{100*flat[11]:.0f}", excessTwelve=f"{100*(Cu[11]-flat[11]):.0f}", cumTwentyFour=f"{100*Cu[23]:.0f}")
# ---------------- Table 2: what predicts a marker ----------------
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); N = pd.read_parquet(os.path.join(OUT, "v_loan_level.parquet")); X = L.join(N, how="inner"); X = X[X.board < "2020-03-01"].copy()
X["marker"] = ((X.mk_employment + X.mk_health + X.mk_family) > 0).astype(float); X["tenure"] = (pd.Timestamp("2020-03-01") - X.board).dt.days / 30.44 / 12; X["lnotes"] = np.log1p(X.notes_pre); X["state"] = X.state.fillna("NA")
for c, b in [("fico", [0, 580, 620, 660, 700, 900]), ("ltv", [0, 80, 90, 97, 105, 1000])]: X[c + "_b"] = pd.cut(X[c], b).astype(object).fillna("missing").astype(str)
X["bal_b"] = pd.qcut(X.bal, 5, duplicates="drop").astype(object).fillna("missing").astype(str)
def fit(f, d): d = d.reset_index(drop=True); mod = smf.ols(f, d, missing="drop"); return mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[mod.data.row_labels, "state"])[0]})
HARD = "C(status_feb20) + C(loan_type) + C(fico_b) + C(ltv_b) + C(bal_b) + C(state)"
m1 = fit("marker ~ " + HARD, X); m2 = fit("marker ~ tenure + " + HARD, X); m3 = fit("marker ~ tenure + lnotes + " + HARD, X)
rows = [["Years since boarding", "", f"{100*m2.params['tenure']:.1f}{star(m2.pvalues['tenure'])}", f"{100*m3.params['tenure']:.1f}{star(m3.pvalues['tenure'])}"], ["", "", f"({100*m2.bse['tenure']:.1f})", f"({100*m3.bse['tenure']:.1f})"],
        ["Log (1 + servicer notes on file)", "", "", f"{100*m3.params['lnotes']:.1f}{star(m3.pvalues['lnotes'])}"], ["", "", "", f"({100*m3.bse['lnotes']:.1f})"], "MID",
        ["Delinquency status, loan type, credit score, LTV, balance, state", "Yes", "Yes", "Yes"], ["$R^2$", f"{m1.rsquared:.3f}", f"{m2.rsquared:.3f}", f"{m3.rsquared:.3f}"], ["Loans", f"{int(m1.nobs):,}", f"{int(m2.nobs):,}", f"{int(m3.nobs):,}"]]
tab("t2_predict.tex", ["Dependent variable: loan has a hardship marker (pp)", "(1)", "(2)", "(3)"], rows)
M.update(rsqHard=f"{m1.rsquared:.2f}", rsqTen=f"{m2.rsquared:.2f}", rsqNotes=f"{m3.rsquared:.2f}", tenCoef=f"{100*m2.params['tenure']:.1f}", notesCoef=f"{100*m3.params['lnotes']:.1f}", markShare=f"{100*X.marker.mean():.0f}", predN=f"{int(m3.nobs):,}")
print("R2:", m1.rsquared, m2.rsquared, m3.rsquared, "| tenure", m2.params["tenure"], "| lnotes", m3.params["lnotes"])
# ---------------- Table 1: the spurious regression, its dissolution, and the accounting ----------------
R2 = json.load(open(os.path.join(OUT, "round2c.json"))); P = R2["period"]
rows = []
for lab, k in [("Hard-data controls and state effects", "mem_baseline"), ("\\quad + indicator for requests from 7 April", "mem_+ post-cutoff dummy"), ("\\quad + week-of-request effects", "mem_+ inquiry-week FE"), ("Requests before 7 April only", "mem_pre-cutoff only"), ("Requests from 7 April only", "mem_post-cutoff only")]:
    v = R2[k]; rows.append([lab, f"{100*v[0]:.1f}{star(v[2])}", f"({100*v[1]:.1f})", f"{int(v[3]):,}"])
v = R2["mem_gov_wk"]; rows += ["MID", ["Placebo regime (statutory right), week effects", f"{100*v[0]:.1f}{star(v[2])}", f"({100*v[1]:.1f})", f"{int(v[3]):,}"]]
tab("t1_spurious.tex", ["Specification", "Coefficient on hardship marker (pp)", "s.e.", "Requests"], rows, align="lccc")
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")).join(N); D = D[(D.Gov == 0) & (D.board < "2020-03-01")]; D["marker"] = ((D.mk_employment.fillna(0) + D.mk_health.fillna(0) + D.mk_family.fillna(0)) > 0).astype(int)
p1, p0 = D[D.marker == 1].post.mean(), D[D.marker == 0].post.mean(); d1, d0 = D[D.post == 1].fb.mean(), D[D.post == 0].fb.mean(); raw = D[D.marker == 1].fb.mean() - D[D.marker == 0].fb.mean()
M.update(accPostMark=f"{100*p1:.0f}", accPostNo=f"{100*p0:.0f}", accDelta=f"{100*(d1-d0):.0f}", accImplied=f"{100*(d1-d0)*(p1-p0):.1f}", accRaw=f"{100*raw:.1f}", accShare=f"{100*(d1-d0)*(p1-p0)/raw:.0f}", spBase=f"{abs(100*R2['mem_baseline'][0]):.1f}", spBaseSE=f"{100*R2['mem_baseline'][1]:.1f}",
         spWk=f"{abs(100*R2['mem_+ inquiry-week FE'][0]):.1f}", spWkSE=f"{100*R2['mem_+ inquiry-week FE'][1]:.1f}", spPre=f"{100*R2['mem_pre-cutoff only'][0]:.1f}", spPost=f"{abs(100*R2['mem_post-cutoff only'][0]):.1f}", spPostSE=f"{100*R2['mem_post-cutoff only'][1]:.1f}", spDummy=f"{abs(100*R2['mem_+ post-cutoff dummy'][0]):.1f}", spGov=f"{100*R2['mem_gov_wk'][0]:.1f}", spGovSE=f"{100*R2['mem_gov_wk'][1]:.1f}", spN=f"{int(R2['mem_baseline'][3]):,}",
         markPre=f"{100*D[D.post==0].marker.mean():.0f}", markPost=f"{100*D[D.post==1].marker.mean():.0f}", grantPre=f"{100*d0:.0f}", grantPost=f"{100*d1:.0f}")
print("accounting: raw", raw, "implied", (d1 - d0) * (p1 - p0))
# ---------------- Fig 2: windows at the process change ----------------
wp = os.path.join(OUT, "sp_windows.parquet")
if os.path.exists(wp):
    Wd = pd.read_parquet(wp); RD = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet"))
    def dm(d, y):
        x = d[d.r.abs() <= 14].dropna(subset=[y]).reset_index(drop=True); m = smf.ols(f"{y} ~ post", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return m.params["post"], m.bse["post"], x.loc[x.post == 0, y].mean()
    res = []
    for w in [2, 5, 10, 30, 60]:
        z = Wd[Wd.w == w].set_index("LoanID")
        for grp, g in [("conv", 0), ("gov", 1)]:
            d = RD[RD.Gov == g].copy(); d["job"] = np.where(z.notes.reindex(d.index).fillna(0) > 0, (z.job.reindex(d.index).fillna(0) > 0).astype(float), np.nan); d["nn"] = z.notes.reindex(d.index).fillna(0)
            a, b = dm(d, "job"), dm(d, "nn"); res.append({"w": w, "grp": grp, "job": a[0], "job_se": a[1], "job_mean": a[2], "notes": b[0], "notes_se": b[1], "notes_mean": b[2], "cover": d[d.r.abs() <= 14].job.notna().mean()})
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "sp_windows_rd.csv"), index=False); print(R.round(3).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5)); xs = np.arange(5)
    for grp, col, nm, off in [("conv", ORANGE, "Process changed (conventional)", -.08), ("gov", BLUE, "Process unchanged (government-backed)", .08)]:
        z = R[R.grp == grp]; axs[0].errorbar(xs + off, 100 * z.job, yerr=196 * z.job_se, color=col, marker="o", ms=4, lw=1.2, capsize=2, label=nm); axs[1].errorbar(xs + off, z.notes, yerr=1.96 * z.notes_se, color=col, marker="o", ms=4, lw=1.2, capsize=2)
    for ax, ttl in zip(axs, ["Jump in share `reported job or income loss' (pp)", "Jump in notes written per borrower"]):
        ax.axhline(0, color=GREY, lw=.6); ax.set_xticks(xs); ax.set_xticklabels([2, 5, 10, 30, 60]); ax.set_xlabel("Measurement window (days after request)", fontsize=8); ax.set_title(ttl, fontsize=8.5)
    axs[1].legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="lower left"); fig.tight_layout(); save(fig, os.path.join(TEX, "f2_windows.pdf"))
    c = R[R.grp == "conv"].set_index("w"); gg = R[R.grp == "gov"].set_index("w")
    M.update(winTwo=f"{100*c.loc[2,'job']:.1f}", winTwoSE=f"{100*c.loc[2,'job_se']:.1f}", winTen=f"{100*c.loc[10,'job']:.1f}", winTenSE=f"{100*c.loc[10,'job_se']:.1f}", winThirty=f"{100*c.loc[30,'job']:.1f}", winThirtySE=f"{100*c.loc[30,'job_se']:.1f}", winSixty=f"{100*c.loc[60,'job']:.1f}", winSixtySE=f"{100*c.loc[60,'job_se']:.1f}",
             winSixtyGov=f"{100*gg.loc[60,'job']:.1f}", winSixtyGovSE=f"{100*gg.loc[60,'job_se']:.1f}", notesSixty=f"{c.loc[60,'notes']:.1f}", notesSixtySE=f"{c.loc[60,'notes_se']:.1f}", notesSixtyMean=f"{c.loc[60,'notes_mean']:.1f}", jobMeanTwo=f"{100*c.loc[2,'job_mean']:.0f}", jobMeanSixty=f"{100*c.loc[60,'job_mean']:.0f}")
# ---------------- Fig 3: markers around a modification ----------------
ep = os.path.join(OUT, "sp_modevent.parquet")
if os.path.exists(ep):
    Ev = pd.read_parquet(ep); ML = pd.read_parquet(os.path.join(OUT, "sp_modloans.parquet")); nl = len(ML); ms = list(range(-12, 6))
    share = [((Ev[Ev.m == m].marker > 0).sum()) / nl for m in ms]; notes = [Ev[Ev.m == m].notes.sum() / nl for m in ms]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
    axs[0].plot(ms, 100 * np.array(share), color=BLUE, lw=1.5, marker="o", ms=3); axs[0].set_title("Loans with a marker note in the month (%)", fontsize=9); axs[1].plot(ms, notes, color=GREY, lw=1.5, marker="o", ms=3); axs[1].set_title("Notes written per loan in the month", fontsize=9)
    for ax in axs: ax.axvline(-.5, color=GREY, lw=.6, ls=":"); ax.set_xlabel("Months relative to the modification date")
    fig.tight_layout(); save(fig, os.path.join(TEX, "f3_modification.pdf"))
    base = np.mean(share[:6]); pk = max(share[6:13]); pm = ms[6 + int(np.argmax(share[6:13]))]
    M.update(modN=f"{nl:,}", modBase=f"{100*base:.1f}", modPeak=f"{100*pk:.1f}", modPeakMonth=str(pm), modRatio=f"{pk/base:.1f}", modAfter=f"{100*np.mean(share[14:]):.1f}")
    print("mod event: base", base, "peak", pk, "at", pm, "after", np.mean(share[14:]))
open(os.path.join(TEX, "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
for k, v in M.items(): print(k, "=", v)
