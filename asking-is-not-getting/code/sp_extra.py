"""Short paper, robustness numbers: hazard profile by cohort / loan type / status / marker type; R2 decomposition. Writes tex/sp/numbers_extra.tex."""
import os, json, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); TEX = os.path.join(H, "tex", "sp"); M = {}
F = pd.read_parquet(os.path.join(OUT, "v_first_marker.parquet"))
for c in F.columns: F[c] = pd.to_datetime(F[c], errors="coerce")
END = pd.Timestamp("2020-02-29"); F = F[(F.board >= "2016-01-01") & (F.board <= "2019-06-30") & F.note.notna()].copy(); F["any"] = F[["employment", "health", "family"]].min(axis=1)
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); F = F.join(L[["Gov", "status_feb20"]], how="left")
def haz(d, col):
    fu = (END - d.board).dt.days / 30.44; t = (d[col] - d.board).dt.days / 30.44; ev = t.notna() & (d[col] <= END); dur = np.clip(np.where(ev, t, fu), 0, None); hz = []
    for m in range(24):
        at = (dur >= m).sum(); hz.append(((ev & (dur >= m) & (dur < m + 1)).sum() / at) if at > 50 else np.nan)
    return np.array(hz)
def prof(d, col="any"): h = haz(d, col); return 100 * np.nanmean(h[1:3]), 100 * np.nanmean(h[12:24]), 100 * h[0]
names = {2016: "Sixteen", 2017: "Seventeen", 2018: "Eighteen", 2019: "Nineteen"}; rat = []
for y, nm in names.items():
    d = F[F.board.dt.year == y]; e, l, _ = prof(d); print(y, len(d), round(e, 1), round(l, 1))
    M["hazE" + nm] = f"{e:.1f}"; M["hazL" + nm] = f"{l:.1f}" if np.isfinite(l) else "n.a."
    h = haz(d, "any"); l2 = 100 * np.nanmean(h[6:12]); M["hazM" + nm] = f"{l2:.1f}"; rat.append(e / l2)
M["cohRatioMin"] = f"{min(rat):.1f}"; M["cohRatioMax"] = f"{max(rat):.1f}"
for k, nm in [(1, "Gov"), (0, "Conv")]:
    e, l, _ = prof(F[F.Gov == k]); M["hazE" + nm] = f"{e:.1f}"; M["hazL" + nm] = f"{l:.1f}"; print(nm, round(e, 1), round(l, 1))
cur = F.status_feb20.astype(str).str.lower().str.contains("current|perform") & ~F.status_feb20.astype(str).str.lower().str.contains("non"); print(F.status_feb20.value_counts().head(8))
for k, nm in [(True, "Cur"), (False, "Dq")]:
    d = F[(cur == k) & F.status_feb20.notna()]; e, l, _ = prof(d); M["hazE" + nm] = f"{e:.1f}"; M["hazL" + nm] = f"{l:.1f}"; M["n" + nm] = f"{len(d):,}"; print(nm, len(d), round(e, 1), round(l, 1))
for col, nm in [("employment", "Emp"), ("health", "Hea"), ("family", "Fam")]:
    e, l, f1 = prof(F, col); M["hazE" + nm] = f"{e:.1f}"; M["hazL" + nm] = f"{l:.1f}"; M["annE" + nm] = f"{100*(1-(1-e/100)**12):.0f}"; print(nm, round(e, 1), round(l, 1))
e, l, f1 = prof(F); M["hazFirst"] = f"{f1:.1f}"
RL = json.load(open(os.path.join(OUT, "relearn_analysis.json"))); M["firstNoteMedian"] = f"{RL['first_note_days_median']:.0f}"
# R2 decomposition
N = pd.read_parquet(os.path.join(OUT, "v_loan_level.parquet")); X = L.join(N, how="inner"); X = X[X.board < "2020-03-01"].copy()
X["marker"] = ((X.mk_employment + X.mk_health + X.mk_family) > 0).astype(float); X["tenure"] = (pd.Timestamp("2020-03-01") - X.board).dt.days / 30.44 / 12; X["lnotes"] = np.log1p(X.notes_pre); X["state"] = X.state.fillna("NA")
for c, b in [("fico", [0, 580, 620, 660, 700, 900]), ("ltv", [0, 80, 90, 97, 105, 1000])]: X[c + "_b"] = pd.cut(X[c], b).astype(object).fillna("missing").astype(str)
X["bal_b"] = pd.qcut(X.bal, 5, duplicates="drop").astype(object).fillna("missing").astype(str)
B = {"Status": "C(status_feb20)", "Type": "C(loan_type)", "Fico": "C(fico_b)", "Ltv": "C(ltv_b)", "Bal": "C(bal_b)", "State": "C(state)"}
r2 = lambda f: smf.ols("marker ~ " + f, X).fit().rsquared
full = r2(" + ".join(B.values()) + " + tenure + lnotes"); M["rsqFull"] = f"{full:.3f}"; M["rsqExpOnly"] = f"{r2('tenure + lnotes'):.3f}"; M["rsqHardOnly"] = f"{r2(' + '.join(B.values())):.3f}"
M["dropExp"] = f"{full - r2(' + '.join(B.values())):.3f}"; big = ("", 0)
for k, v in B.items():
    d = full - r2(" + ".join(x for kk, x in B.items() if kk != k) + " + tenure + lnotes"); M["drop" + k] = f"{d:.3f}"; big = (k, d) if d > big[1] else big; print("drop", k, round(d, 4))
M["dropAllHard"] = f"{full - r2('tenure + lnotes'):.3f}"; print("full", full, "exp only", M["rsqExpOnly"], "hard only", M["rsqHardOnly"], "drop exp", M["dropExp"], "drop all hard", M["dropAllHard"], "largest hard block", big)
open(os.path.join(TEX, "numbers_extra.tex"), "w", encoding="utf-8").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")
for k, v in M.items(): print(k, "=", v)
