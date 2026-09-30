"""CARES event-study rebuild.
Panel: IB_OB_Flags_Full_Nov3 (2018-01..2020-09, single source for pre & post).
Treatment: Gov = loan_type in {FHA,VA,USDA} vs CONV (from Monthly_Loan_Detail modal type);
           also the paper's own 19,421-loan Gov sample for a Table-5-comparable run.
Outcomes: dq30, dq90, known inbound/outbound flag, collections, admin forbearance (monthly, from
          FB_NonPay_Start/End), covid-forbearance message flag.
Spec: Y_it = a_i + l_t + sum_k b_k 1[t=k]Gov_i + e ; loan FE absorbed by within-demeaning,
      month dummies explicit; SE clustered by loan (primary) and state.
Also: Table-5-style 2x2 (Jan-Feb vs Apr-May) and a 2016-2020 pre-trend on Merged_Svc.
"""
import os, json, csv, sys, time
import numpy as np, pandas as pd
import statsmodels.api as sm
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time()-T0), *a, flush=True)

# ---------- loan-level attributes ----------
log("loan_type + state from Monthly_Loan_Detail")
mld = pd.read_csv(r"<DATA>/panel\Monthly_Loan_Detail.csv",
                  usecols=["LoanID", "loan_type", "state"], dtype=str, na_values=["NULL"])
mld["loan_type"] = mld["loan_type"].str.strip().str.upper()
mld["loan_type"] = mld["loan_type"].replace({"CONVENTIONAL": "CONV"})
mld = mld[mld["loan_type"].isin(["FHA", "VA", "USDA", "CONV"])]
lt = mld.groupby("LoanID")["loan_type"].agg(lambda s: s.value_counts().index[0])
st = mld.groupby("LoanID")["state"].agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else np.nan)
attr = pd.DataFrame({"loan_type": lt, "state": st})
attr["Gov"] = attr["loan_type"].isin(["FHA", "VA", "USDA"]).astype(int)
log("loans with clean type:", len(attr), attr["loan_type"].value_counts().to_dict())

gov_paper = pd.read_csv(r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv",
                        usecols=["LoanID", "Gov"], dtype=str).drop_duplicates("LoanID").set_index("LoanID")["Gov"].astype(int)
log("paper Gov sample:", len(gov_paper))

# ---------- administrative forbearance ----------
log("admin forbearance from COVID file")
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv",
                 usecols=["LoanID", "FB_Agreement_Date", "FB_NonPay_Start_Date", "FB_NonPay_End_Date"],
                 dtype=str, na_values=["NULL"])
for c in ["FB_Agreement_Date", "FB_NonPay_Start_Date", "FB_NonPay_End_Date"]:
    cv[c] = pd.to_datetime(cv[c], errors="coerce")
fb = cv.groupby("LoanID").agg(fb_agree=("FB_Agreement_Date", "min"),
                              fb_start=("FB_NonPay_Start_Date", "min"),
                              fb_end=("FB_NonPay_End_Date", "max"))
log("loans with FB agreement:", fb["fb_agree"].notna().sum())

# ---------- monthly panel ----------
log("IB_OB panel")
ib = pd.read_csv(r"<DATA>/panel\IB_OB_Flags_Full_Nov3.csv",
                 usecols=["LoanId", "Month t", "LoanStatus", "Known_Inbound_Borrower_Flag",
                          "Known_Outbound_Borrower_Flag", "Collections", "Covid_Forbearance_Flag_Msg"],
                 dtype={"LoanId": str}, na_values=["NULL"])
ib = ib.rename(columns={"LoanId": "LoanID"})
ib["month"] = pd.to_datetime(ib["Month t"], errors="coerce").dt.to_period("M")
ib = ib.dropna(subset=["month"])
ib = ib.drop_duplicates(["LoanID", "month"])
s = ib["LoanStatus"].fillna("")
ib = ib[~s.isin(["Pending Servicing Transfer", "Not on Servicer File"])]
s = ib["LoanStatus"]
ib["dq30"] = s.isin(["Active - 30 Days DQ", "Active - 60 Days DQ", "Active - 90 Days DQ",
                     "Active - 120+ Days DQ", "Active - FC", "Active - BK"]).astype(float)
ib["dq90"] = s.isin(["Active - 90 Days DQ", "Active - 120+ Days DQ", "Active - FC", "Active - BK"]).astype(float)
ib["inbound"] = ib["Known_Inbound_Borrower_Flag"].fillna(0).clip(upper=1).astype(float)
ib["outbound"] = ib["Known_Outbound_Borrower_Flag"].fillna(0).clip(upper=1).astype(float)
ib["fb_msg"] = ib["Covid_Forbearance_Flag_Msg"].fillna(0).clip(upper=1).astype(float)
ib["collections"] = pd.to_numeric(ib["Collections"], errors="coerce")
ib = ib.join(attr, on="LoanID").join(fb, on="LoanID")
ib["gov_paper"] = ib["LoanID"].map(gov_paper)
ms, me = ib["month"].dt.start_time, ib["month"].dt.end_time
ib["fb_admin"] = ((ib["fb_start"] <= me) & (ib["fb_end"] >= ms)).astype(float)
ib.loc[ib["fb_start"].isna(), "fb_admin"] = 0.0
ib = ib[(ib["month"] >= "2018-01") & (ib["month"] <= "2020-09")]
log("panel rows", len(ib), "loans", ib["LoanID"].nunique(),
    "with clean type", ib.dropna(subset=["Gov"])["LoanID"].nunique())

OUTCOMES = ["dq30", "dq90", "inbound", "outbound", "fb_admin", "fb_msg", "collections"]
REF = pd.Period("2020-02", "M")

def twfe_event(df, govcol, y, cluster):
    d = df.dropna(subset=[govcol, y]).copy()
    d["g"] = d[govcol].astype(float)
    months = sorted(d["month"].unique())
    ev = [m for m in months if m != REF]
    X = pd.DataFrame(index=d.index)
    for m in months[1:]:
        X["m_" + str(m)] = (d["month"] == m).astype(float)
    for m in ev:
        X["e_" + str(m)] = ((d["month"] == m) & (d["g"] == 1)).astype(float)
    # within-loan demeaning (absorbs loan FE; Gov main effect is absorbed)
    grp = d["LoanID"]
    Xd = X - X.groupby(grp).transform("mean")
    yd = d[y] - d[y].groupby(grp).transform("mean")
    res = sm.OLS(yd.values, Xd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    idx = {n: i for i, n in enumerate(X.columns)}
    rows = []
    for m in ev:
        i = idx["e_" + str(m)]
        rows.append({"month": str(m), "beta": res.params[i], "se": res.bse[i]})
    rows.append({"month": str(REF), "beta": 0.0, "se": 0.0})
    es = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    # DiD: post = >= 2020-04 (drop March), pre = 2018-01..2020-02
    dd = d[d["month"] != pd.Period("2020-03", "M")].copy()
    dd["post"] = (dd["month"] >= "2020-04").astype(float)
    dd["gp"] = dd["g"] * dd["post"]
    Xm = pd.get_dummies(dd["month"].astype(str), drop_first=True, dtype=float)
    Xm["gp"] = dd["gp"].values
    grp = dd["LoanID"]
    Xd = Xm - Xm.groupby(grp.values).transform("mean")
    yd = dd[y] - dd[y].groupby(grp.values).transform("mean")
    r2 = sm.OLS(yd.values, Xd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd[cluster])[0]})
    j = list(Xm.columns).index("gp")
    # joint pre-trend test: all pre-period event coefs = 0
    pre_ix = [idx["e_" + str(m)] for m in ev if m < REF]
    R = np.zeros((len(pre_ix), len(X.columns)))
    for r_, i in enumerate(pre_ix): R[r_, i] = 1
    ft = res.wald_test(R, scalar=True)
    return es, {"did": r2.params[j], "did_se": r2.bse[j], "n": int(len(dd)),
                "loans": int(dd["LoanID"].nunique()), "n_gov": int(dd.loc[dd.g == 1, "LoanID"].nunique()),
                "n_conv": int(dd.loc[dd.g == 0, "LoanID"].nunique()),
                "pre_mean_gov": float(dd.loc[(dd.g == 1) & (dd.post == 0), y].mean()),
                "pre_mean_conv": float(dd.loc[(dd.g == 0) & (dd.post == 0), y].mean()),
                "pretrend_wald_p": float(ft.pvalue), "pretrend_df": len(pre_ix)}

def table5(df, govcol, y):
    """Paper's diagnostic: loan-level (Apr:May mean) - (Jan:Feb mean), balanced on 4 months; placebo Feb-Jan."""
    d = df.dropna(subset=[govcol, y])
    d = d[d["month"].isin([pd.Period(x, "M") for x in ["2020-01", "2020-02", "2020-04", "2020-05"]])]
    w = d.pivot_table(index="LoanID", columns="month", values=y)
    w = w.dropna()
    g = d.drop_duplicates("LoanID").set_index("LoanID")[govcol].reindex(w.index)
    P = lambda s: pd.Period(s, "M")
    delta = (w[P("2020-04")] + w[P("2020-05")]) / 2 - (w[P("2020-01")] + w[P("2020-02")]) / 2
    plc = w[P("2020-02")] - w[P("2020-01")]
    def dm(x):
        a, b = x[g == 1], x[g == 0]
        return a.mean() - b.mean(), np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    did, se = dm(delta); pl, pse = dm(plc)
    return {"n_gov": int((g == 1).sum()), "n_conv": int((g == 0).sum()), "did": did, "se": se, "placebo": pl, "placebo_se": pse}

results = {"event": {}, "did": {}, "table5": {}}
for samp, govcol in [("extended", "Gov"), ("paper", "gov_paper")]:
    for y in OUTCOMES:
        log(samp, y)
        es, dd = twfe_event(ib, govcol, y, "LoanID")
        _, dds = twfe_event(ib, govcol, y, "state") if samp == "extended" else (None, None)
        if dds: dd["did_se_state"] = dds["did_se"]; dd["pretrend_wald_p_state"] = dds["pretrend_wald_p"]
        results["event"][f"{samp}:{y}"] = es.to_dict("records")
        results["did"][f"{samp}:{y}"] = dd
        results["table5"][f"{samp}:{y}"] = table5(ib, govcol, y)
        es.to_csv(os.path.join(OUT, f"event_{samp}_{y}.csv"), index=False)

# ---------- long pre-trend on Merged_Svc (2016-01..2020-02) ----------
log("Merged_Svc pre-trend")
sv = pd.read_csv(r"<DATA>/relief\Merged_Svc.csv",
                 usecols=["LoanID", "Yearmonth", "FiservDelqStatusMBA"], dtype=str, na_values=["NULL"])
sv["month"] = pd.PeriodIndex(pd.to_datetime(sv["Yearmonth"].str.slice(0, 6), format="%Y%m", errors="coerce"), freq="M")
sv = sv.dropna(subset=["month"]).drop_duplicates(["LoanID", "month"])
sv["dq30"] = sv["FiservDelqStatusMBA"].isin(["30", "60", "90", "120+", "FC", "BK", "REO"]).astype(float)
sv = sv.join(attr[["Gov"]], on="LoanID").dropna(subset=["Gov"])
sv = sv[(sv["month"] >= "2016-01") & (sv["month"] <= "2020-02")]
raw = sv.groupby(["month", "Gov"])["dq30"].mean().unstack()
raw["gap"] = raw[1] - raw[0]
raw.index = raw.index.astype(str)
results["svc_pretrend"] = {"months": list(raw.index), "gov": raw[1].round(4).tolist(),
                           "conv": raw[0].round(4).tolist(), "gap": raw["gap"].round(4).tolist(),
                           "loans": int(sv["LoanID"].nunique())}
es_sv, _ = twfe_event(sv, "Gov", "dq30", "LoanID")
results["svc_event_dq30"] = es_sv.to_dict("records")
raw.to_csv(os.path.join(OUT, "svc_pretrend_dq30.csv"))

# ---------- raw monthly means by Gov (for the page) ----------
means = ib.dropna(subset=["Gov"]).groupby(["month", "Gov"])[OUTCOMES].mean().round(4)
means.index = means.index.set_levels(means.index.levels[0].astype(str), level=0)
results["means"] = {y: {"months": [m for m, g in means.index if g == 1],
                        "gov": means.xs(1, level=1)[y].tolist(),
                        "conv": means.xs(0, level=1)[y].tolist()} for y in OUTCOMES}
results["meta"] = {"panel_rows": int(len(ib)), "panel_loans": int(ib["LoanID"].nunique()),
                   "ext_loans": int(ib.dropna(subset=["Gov"])["LoanID"].nunique()),
                   "paper_loans": int(ib.dropna(subset=["gov_paper"])["LoanID"].nunique()),
                   "months": [str(m) for m in sorted(ib["month"].unique())]}
with open(os.path.join(OUT, "cares_results.json"), "w") as f:
    json.dump(results, f, indent=1, default=float)
log("DONE")
