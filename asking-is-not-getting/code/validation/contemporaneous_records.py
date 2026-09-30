"""Were the request and grant dates recorded at the time?  Rebuilds the analysis dates from the servicer's month-end loads of the
relief feed (one file per month-end, Jan-2020 to May-2021, as loaded into the investor's database at each run date) and compares
what was on file at each month-end with the dates the paper uses.  Loan-level data never leave the machine; output is aggregate JSON.
Inputs (paths set below): FB_<Mon>_<YY>.csv month-end loads (two extracts), Monthly_Loan_Detail.csv (loan type, as in loanframe.py)."""
import os, re, glob, json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings("ignore")
# Paths are redacted in the public package, as elsewhere in code/. Set them through the environment on the secure machine.
LOADS = [os.environ.get("AING_LOADS_2021JUN", "<DATA>/relief_loads_Jun2021"), os.environ.get("AING_LOADS_2021APR", "<DATA>/relief_loads_Apr2021")]
MLD = os.environ.get("AING_MLD", "<DATA>/panel/Monthly_Loan_Detail.csv"); TMP = os.environ.get("AING_SECURE_TMP", "<SECURE_TMP>")
OUTJ = os.environ.get("AING_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "validation", "contemporaneous_records.json"))
CUT = pd.Timestamp("2020-04-07"); MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
P = lambda s: pd.Period(s, "M")
# ---------------- month-end loads ----------------
KEEP = ["LoanID","RunDate","LoadDate","COVID_Inquiry_Date","FB_Agreement_Date","Forbearance_Flag","FBType","LSMIT_Status","LoanStatusBucket"]
rows = []
for src, folder in enumerate(LOADS):
    for f in sorted(glob.glob(os.path.join(folder, "FB_*_*.csv"))):
        mo, yy = re.search(r"FB_(\w{3})_(\d\d)\.csv", f).groups()
        d = pd.read_csv(f, dtype=str, na_values=["NULL", ""], encoding="latin1"); d.columns = [c.strip().lstrip("﻿").lstrip("ï»¿") for c in d.columns]
        d = d.loc[:, ~d.columns.duplicated()]
        if "RunDate" not in d or d.RunDate.isna().all(): d["RunDate"] = d["LoadDate"]
        d = d[[c for c in KEEP if c in d]].copy(); d["snap"] = P(f"20{yy}-{MON[mo]:02d}"); d["src"] = src; rows.append(d)
S = pd.concat(rows, ignore_index=True)
for c in ["RunDate","LoadDate","COVID_Inquiry_Date","FB_Agreement_Date"]: S[c] = pd.to_datetime(S[c], errors="coerce", format="mixed")
for c in ["FBType","LSMIT_Status","LoanStatusBucket"]: S[c] = S[c].str.strip()
S["run"] = S.groupby(["src","snap"]).RunDate.transform("max")           # the month-end run date of the load
S = S.sort_values(["LoanID","snap","src"]).drop_duplicates(["LoanID","snap"], keep="first")
# ---------------- loan type (as in loanframe.py) and pre-pandemic residence ----------------
M = pd.read_csv(MLD, usecols=["LoanID","loan_type"], dtype=str, na_values=["NULL"]).dropna()
M["t"] = M.loan_type.str.strip().str.upper().replace({"CONVENTIONAL": "CONV"})
lt = M[M.t.isin(["FHA","VA","USDA","CONV"])].groupby("LoanID").t.agg(lambda s: s.value_counts().index[0])
feb = S[(S.snap == P("2020-02")) & S.LoanStatusBucket.fillna("").str.startswith("Active")].LoanID.unique()
# ---------------- request and grant dates, as the paper defines them ----------------
I = S[S.COVID_Inquiry_Date >= "2020-03-01"]
F = pd.DataFrame({"inq": I.groupby("LoanID").COVID_Inquiry_Date.min()}).join(S.groupby("LoanID").FB_Agreement_Date.min().rename("agree")).join(lt.rename("ltype"))
F = F[F.ltype.notna() & F.index.isin(feb) & (F.inq <= "2020-09-30")].copy()
F["Gov"] = F.ltype.isin(["FHA","VA","USDA"]).astype(int); F["fb"] = (F.agree <= "2020-09-30").astype(int)
F["r"] = (F.inq.dt.normalize() - CUT).dt.days; F["post"] = (F.r >= 0).astype(int)
# when each date first appeared in a month-end load
fq = S[S.COVID_Inquiry_Date.notna()].groupby(["LoanID","COVID_Inquiry_Date"]).run.min()
fa = S[S.FB_Agreement_Date.notna()].groupby(["LoanID","FB_Agreement_Date"]).run.min()
F["inq_lag"] = (pd.to_datetime(fq.reindex(list(zip(F.index, F.inq))).values) - pd.DatetimeIndex(F.inq)).days.values
ag = F.agree.fillna(pd.Timestamp("1900-01-01"))
F["agr_lag"] = np.where(F.agree.notna(), (pd.to_datetime(fa.reindex(list(zip(F.index, ag))).values) - pd.DatetimeIndex(ag)).days.values, np.nan)
F["late_inq"] = (F.inq_lag > 45).astype(int); F["late_agr"] = ((F.fb == 1) & (F.agr_lag > 45)).astype(int)
F["fb_timely"] = ((F.fb == 1) & (F.agr_lag <= 45)).astype(int)
F["fb21_timely"] = ((F.fb == 1) & ((F.agree - F.inq).dt.days <= 21) & (F.agr_lag <= 45)).astype(int)
on = lambda snap, col: S[S.snap == P(snap)].set_index("LoanID")[col]
F["inq_apr"] = on("2020-04", "COVID_Inquiry_Date").reindex(F.index); F["agr_apr"] = on("2020-04", "FB_Agreement_Date").reindex(F.index)
F["agr_may"] = on("2020-05", "FB_Agreement_Date").reindex(F.index)
F["fb_onfile_apr"] = F.agr_apr.notna().astype(int); F["fb_onfile_may"] = (F.agr_apr.notna() | F.agr_may.notna()).astype(int)
Z = S[(S.snap >= P("2020-04")) & (S.snap <= P("2020-09"))]
anyflag = (Z.Forbearance_Flag.notna() | Z.FBType.fillna("No FB").ne("No FB") | Z.LSMIT_Status.fillna("").str.contains("Forbearance|FB Plan", case=False)).groupby(Z.LoanID).max()
F["anyflag"] = anyflag.reindex(F.index).fillna(False).astype(int)
F.to_parquet(os.path.join(TMP, "frame_contemp.parquet"))
# ---------------- estimates ----------------
def ll(d, y, h=14):
    x = d[d.r.abs() <= h].dropna(subset=[y]).copy(); x["w"] = 1 - x.r.abs() / (h + 1); x = x.reset_index(drop=True)
    m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]})
    return {"est": 100 * m.params["post"], "se": 100 * m.bse["post"], "p": m.pvalues["post"], "n": int(len(x)), "left_mean": 100 * x.loc[x.post == 0, y].mean()}
R = {"n_frame": int(len(F)), "n_conv": int((F.Gov == 0).sum()), "n_gov": int(F.Gov.sum())}
for g, lab in [(0, "conv"), (1, "gov")]:
    d = F[F.Gov == g]; w = d[d.r.between(-14, 14)]; w0 = w[w.inq <= "2020-04-28"]
    R[lab] = {
      "n14_pre": int((w.r < 0).sum()), "n14_post": int((w.r >= 0).sum()),
      "grant_pre": 100 * d[d.r < 0].fb.mean(), "grant_post": 100 * d[d.r >= 0].fb.mean(),
      "fs": ll(d, "fb"), "fs_timely": ll(d, "fb_timely"), "fs21_timely": ll(d, "fb21_timely"),
      "fs_onfile_apr": ll(d[d.inq <= "2020-04-21"], "fb_onfile_apr"), "fs_onfile_may": ll(d[d.inq <= "2020-04-21"], "fb_onfile_may"),
      "fs_timely_inq_only": ll(d[d.late_inq == 0], "fb_timely"),
      "jump_late_inq": ll(d, "late_inq"), "jump_late_agr": ll(d, "late_agr"),
      "inq_onfile_apr_same": 100 * (w0.inq_apr.dt.normalize() == w0.inq.dt.normalize()).mean(), "n_inq_onfile_test": int(len(w0)),
      "agr_onfile_apr_same": 100 * (w0[w0.agree <= "2020-04-28"].agr_apr.dt.normalize() == w0[w0.agree <= "2020-04-28"].agree.dt.normalize()).mean(),
      "n_agr_onfile_test": int((w0.agree <= "2020-04-28").sum()),
      "late_inq_share": 100 * w.late_inq.mean(), "late_agr_share_pre": 100 * w[(w.post == 0) & (w.fb == 1)].late_agr.mean(),
      "late_agr_share_post": 100 * w[(w.post == 1) & (w.fb == 1)].late_agr.mean(),
      "median_agr_lag": float(w[w.fb == 1].agr_lag.median()), "median_inq_lag": float(w.inq_lag.median()),
      "flag_if_fb": 100 * w[w.fb == 1].anyflag.mean(), "flag_if_nofb": 100 * w[w.fb == 0].anyflag.mean(),
      "flag_if_nofb_post": 100 * w[(w.fb == 0) & (w.post == 1)].anyflag.mean(), "n_nofb_post": int(((w.fb == 0) & (w.post == 1)).sum()),
      "n_nofb_pre": int(((w.fb == 0) & (w.post == 0)).sum()), "k_flag_nofb_pre": int(w[(w.fb == 0) & (w.post == 0)].anyflag.sum()),
      "k_flag_nofb_post": int(w[(w.fb == 0) & (w.post == 1)].anyflag.sum()),
      "fs_fb_or_flag": ll(d.assign(fbx=((d.fb == 1) | (d.anyflag == 1)).astype(int)), "fbx"),
      "min_agreement_date": str(d.agree.min().date())}
# two extractions of the same month-end runs, made in April and June 2021: do they agree?
ext = {}
for snap in ("2020-04", "2020-06", "2020-09"):
    a = pd.concat([r for r in rows if r.src.iat[0] == 0 and r.snap.iat[0] == P(snap)]).drop_duplicates("LoanID").set_index("LoanID")
    b = pd.concat([r for r in rows if r.src.iat[0] == 1 and r.snap.iat[0] == P(snap)]).drop_duplicates("LoanID").set_index("LoanID")
    j = a.join(b, rsuffix="_b", how="inner"); out = {"overlap_loans": int(len(j))}
    for c in ("COVID_Inquiry_Date", "FB_Agreement_Date"):
        x, y = pd.to_datetime(j[c], errors="coerce", format="mixed").dt.normalize(), pd.to_datetime(j[c + "_b"], errors="coerce", format="mixed").dt.normalize()
        k = x.notna() | y.notna(); out[c] = {"n": int(k.sum()), "identical": 100 * float((x[k] == y[k]).mean())}
    win = set(F[F.r.between(-14, 14)].index); jw = j[j.index.isin(win)]; out["window_loans"] = int(len(jw))
    for c in ("COVID_Inquiry_Date", "FB_Agreement_Date"):
        x, y = pd.to_datetime(jw[c], errors="coerce", format="mixed").dt.normalize(), pd.to_datetime(jw[c + "_b"], errors="coerce", format="mixed").dt.normalize()
        k = x.notna() | y.notna(); out[c + "_window"] = {"n": int(k.sum()), "identical": 100 * float((x[k] == y[k]).mean()) if k.sum() else None}
    ext[snap] = out
R["two_extractions"] = ext
R["fields_in_march_load"] = {c: int(S[(S.snap == P("2020-03"))][c].notna().sum()) for c in ("COVID_Inquiry_Date", "FB_Agreement_Date")}
# the inquiry field holds the latest inquiry: how the April-2020 value evolves in later loads
A4 = S[S.snap == P("2020-04")].set_index("LoanID").COVID_Inquiry_Date.dropna(); A9 = S[S.snap == P("2020-09")].set_index("LoanID").COVID_Inquiry_Date.reindex(A4.index)
R["field_evolution_apr_to_sep"] = {"n": int(len(A4)), "same": 100 * float((A9 == A4).mean()), "later": 100 * float((A9 > A4).mean()), "earlier": 100 * float((A9 < A4).mean())}
w = F[F.r.between(-14, 14)]
R["max_lag_days"] = {"inq": {int(k): float(v) for k, v in w.groupby("Gov").inq_lag.max().items()}, "agr": {int(k): float(v) for k, v in w[w.fb == 1].groupby("Gov").agr_lag.max().items()}}
os.makedirs(os.path.dirname(OUTJ), exist_ok=True); json.dump(R, open(OUTJ, "w"), indent=1, default=float)
for k in ("conv", "gov"):
    r = R[k]; print(k, {kk: (round(v["est"], 1), round(v["se"], 1), v["n"]) if isinstance(v, dict) else (round(v, 1) if isinstance(v, float) else v) for kk, v in r.items()})
