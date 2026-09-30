"""(A) Inquiry timing around CARES enactment (27 Mar 2020), by backing. (B) Why conventional requests fail: LSMIT status."""
import os, pandas as pd, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "out")
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); K = L[L.in_covid_file == 1][["Gov", "loan_type", "dq_feb20", "state"]]
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv", dtype=str, na_values=["NULL"],
                 usecols=["LoanID", "COVID_Inquiry_Date", "FB_Agreement_Date", "FB_NonPay_Start_Date", "LSMIT_Status", "LSMIT_StatusDate", "RepaymentPlan", "DispositionPath", "CreatedAt"])
for c in ["COVID_Inquiry_Date", "FB_Agreement_Date", "FB_NonPay_Start_Date", "LSMIT_StatusDate"]: cv[c] = pd.to_datetime(cv[c], errors="coerce")
cv = cv[cv.LoanID.isin(K.index)]
print("rows", len(cv), "| inquiry dates before 2020-03-01:", (cv.COVID_Inquiry_Date < "2020-03-01").sum(), "| distinct CreatedAt:", cv.CreatedAt.nunique())
print("is LSMIT_Status constant within loan?", (cv.groupby("LoanID").LSMIT_Status.nunique(dropna=False) > 1).mean().round(3), " FB_Agreement constant?", (cv.groupby("LoanID").FB_Agreement_Date.nunique() > 1).mean().round(3))

# ---------- A: first inquiry (any date in 2020) and agreement ----------
first = cv[cv.COVID_Inquiry_Date >= "2020-01-01"].groupby("LoanID").COVID_Inquiry_Date.min()
ag = cv.groupby("LoanID").FB_Agreement_Date.min()
D = K.join(first.rename("inq")).join(ag.rename("agree")); D = D[D.inq.notna() & (D.inq <= "2020-09-30")]
ACT = pd.Timestamp("2020-03-27")
D["wk"] = D.inq.dt.to_period("W-THU").dt.start_time
D["fb"] = (D.agree.notna() & (D.agree <= "2020-09-30")).astype(int); D["days"] = (D.agree - D.inq).dt.days
print("\nFirst inquiries by week (Feb-May 2020): n / conversion / median days to agreement")
t = D[(D.inq >= "2020-02-15") & (D.inq <= "2020-05-15")].groupby(["wk", "Gov"]).agg(n=("fb", "size"), conv=("fb", "mean"), days=("days", "median")).round(2).unstack("Gov")
print(t.to_string())
pre = D[D.inq < ACT]; post = D[(D.inq >= ACT) & (D.inq <= "2020-05-31")]
for lab, d in [("PRE-ACT inquiries (before 27 Mar)", pre), ("POST-ACT inquiries (27 Mar-31 May)", post)]:
    print("\n" + lab); print(d.groupby("Gov").agg(n=("fb", "size"), conv=("fb", "mean"), days_med=("days", "median"), within7=("days", lambda s: (s <= 7).mean())).round(3).to_string())
print("\nearliest agreement dates:", sorted(D.agree.dropna().unique())[:5])
print("agreements dated before the Act:", (D.agree < ACT).sum(), " by Gov:", D[D.agree < ACT].groupby("Gov").size().to_dict())
# calendar-date hazard for pre-Act inquirers: share with agreement by date
for g in (1, 0):
    d = pre[pre.Gov == g]; print(f"\npre-Act inquirers Gov={g} (n={len(d)}): share with agreement by date")
    for dt in ["2020-03-27", "2020-03-31", "2020-04-03", "2020-04-10", "2020-04-17", "2020-04-30", "2020-05-31", "2020-09-30"]:
        print("   ", dt, round(float((d.agree <= dt).mean()), 3))
D.to_parquet(os.path.join(OUT, "inq_timing.parquet"))

# ---------- B: why conventional requests fail ----------
last = cv.sort_values("LSMIT_StatusDate").groupby("LoanID").agg(status=("LSMIT_Status", "last"), sdate=("LSMIT_StatusDate", "last"), rp=("RepaymentPlan", "last"), disp=("DispositionPath", "last"))
E = D.join(last)
for lab, d in [("CONV inquirers WITHOUT forbearance", E[(E.Gov == 0) & (E.fb == 0)]), ("CONV inquirers WITH forbearance", E[(E.Gov == 0) & (E.fb == 1)]), ("GOV inquirers WITHOUT forbearance", E[(E.Gov == 1) & (E.fb == 0)])]:
    print(f"\n{lab} (n={len(d)}): LSMIT status (Apr-2021 snapshot)"); print((d.status.fillna("<none>").value_counts(normalize=True).head(12) * 100).round(1).to_string())
    print("  status dated Mar-Sep 2020:", round(float(((d.sdate >= "2020-03-01") & (d.sdate <= "2020-09-30")).mean()), 3), "| repayment plan flag:", d.rp.fillna("<none>").value_counts().head(3).to_dict())
