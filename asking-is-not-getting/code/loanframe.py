"""Loan-level frame for the journal rewrite. One row per loan active in Feb 2020.
Predetermined: Gov, loan_type, state, pool, Feb-20 status, pre-period responsiveness & hardship flags.
Pandemic: first COVID inquiry, FB agreement/start/end/opt-out, Apr-21 disposition."""
import os, time, numpy as np, pandas as pd
OUT = os.path.join(os.path.dirname(__file__), "out"); T0 = time.time()
def log(*a): print("[%4.0fs]" % (time.time()-T0), *a, flush=True)

mld = pd.read_csv(r"<DATA>/panel\Monthly_Loan_Detail.csv",
                  usecols=["LoanID", "loan_type", "state", "original_fico", "current_balance", "orig_ltv", "cutoff_date"],
                  dtype=str, na_values=["NULL"])
mld["loan_type"] = mld["loan_type"].str.strip().str.upper().replace({"CONVENTIONAL": "CONV"})
ok = mld[mld["loan_type"].isin(["FHA", "VA", "USDA", "CONV"])]
mode = lambda s: s.dropna().value_counts().index[0] if s.notna().any() else np.nan
attr = ok.groupby("LoanID").agg(loan_type=("loan_type", mode), state=("state", mode))
pre = mld[mld["cutoff_date"] < "2020-03"].sort_values("cutoff_date").groupby("LoanID").agg(
        fico=("original_fico", "last"), bal=("current_balance", "last"), ltv=("orig_ltv", "last"))
attr = attr.join(pre)
for c in ["fico", "bal", "ltv"]: attr[c] = pd.to_numeric(attr[c], errors="coerce")
attr["Gov"] = attr["loan_type"].isin(["FHA", "VA", "USDA"]).astype(int)
log("typed loans", len(attr))

ib = pd.read_csv(r"<DATA>/panel\IB_OB_Flags_Full_Nov3.csv", dtype={"LoanId": str}, na_values=["NULL"],
                 usecols=["LoanId", "Investor", "AssetType", "LoanStatus", "Month t", "Known_Inbound_Borrower_Flag",
                          "Known_Outbound_Borrower_Flag", "Unemployed_Flag_Msg", "Curtailment_of_Income_Flag_Msg",
                          "IB_Intent_Keep_Property_Msg", "Collections"]).rename(columns={"LoanId": "LoanID"})
ib["month"] = pd.to_datetime(ib["Month t"], errors="coerce").dt.to_period("M")
ib = ib.dropna(subset=["month"]).drop_duplicates(["LoanID", "month"])
for c in ["Known_Inbound_Borrower_Flag", "Known_Outbound_Borrower_Flag", "Unemployed_Flag_Msg",
          "Curtailment_of_Income_Flag_Msg", "IB_Intent_Keep_Property_Msg"]:
    ib[c] = pd.to_numeric(ib[c], errors="coerce").fillna(0).clip(upper=1)
P = lambda s: pd.Period(s, "M")
prep = ib[(ib["month"] >= P("2018-01")) & (ib["month"] <= P("2020-02"))]
g = prep.groupby("LoanID")
resp = pd.DataFrame({"pre_months": g.size(),
                     "pre_in": g["Known_Inbound_Borrower_Flag"].mean(),
                     "pre_out": g["Known_Outbound_Borrower_Flag"].mean(),
                     "pre_in_n": g["Known_Inbound_Borrower_Flag"].sum(),
                     "pre_out_n": g["Known_Outbound_Borrower_Flag"].sum(),
                     "pre_unemp": g["Unemployed_Flag_Msg"].max(),
                     "pre_curtail": g["Curtailment_of_Income_Flag_Msg"].max(),
                     "pre_keep": g["IB_Intent_Keep_Property_Msg"].max()})
# responsiveness conditional on being contacted: inbound months / outbound months
resp["resp_ratio"] = resp["pre_in_n"] / resp["pre_out_n"].replace(0, np.nan)
feb = ib[ib["month"] == P("2020-02")].set_index("LoanID")[["LoanStatus", "Investor", "AssetType"]]
feb = feb[~feb["LoanStatus"].isin(["Pending Servicing Transfer", "Not on Servicer File"])]
feb["dq_feb20"] = (~feb["LoanStatus"].eq("Active - Current")).astype(int)
feb["status_feb20"] = feb["LoanStatus"].str.replace("Active - ", "", regex=False)
# post-period contact (Mar-Apr 2020) and presence through Sep 2020
post = ib[(ib["month"] >= P("2020-03")) & (ib["month"] <= P("2020-09"))].groupby("LoanID")
postc = pd.DataFrame({"post_months": post.size(),
                      "out_marapr": ib[(ib["month"] >= P("2020-03")) & (ib["month"] <= P("2020-04"))].groupby("LoanID")["Known_Outbound_Borrower_Flag"].max(),
                      "in_marapr": ib[(ib["month"] >= P("2020-03")) & (ib["month"] <= P("2020-04"))].groupby("LoanID")["Known_Inbound_Borrower_Flag"].max()})
log("Feb-20 active loans", len(feb))

cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv", dtype=str, na_values=["NULL"],
                 usecols=["LoanID", "COVID_Inquiry_Date", "FB_Agreement_Date", "FB_NonPay_Start_Date", "FB_NonPay_End_Date",
                          "COVID_FB_OptOut_Request_Date", "ServicingTransferDate", "DispositionPath", "LSMIT_Status",
                          "FiservDelqStatusMBA", "ModificationDate", "ModificationType", "OccupancyDescription"])
for c in ["COVID_Inquiry_Date", "FB_Agreement_Date", "FB_NonPay_Start_Date", "FB_NonPay_End_Date",
          "COVID_FB_OptOut_Request_Date", "ServicingTransferDate", "ModificationDate"]:
    cv[c] = pd.to_datetime(cv[c], errors="coerce")
inq = cv[cv["COVID_Inquiry_Date"] >= "2020-03-01"]
c1 = cv.groupby("LoanID").agg(fb_agree=("FB_Agreement_Date", "min"), fb_start=("FB_NonPay_Start_Date", "min"),
                              fb_end=("FB_NonPay_End_Date", "max"), optout=("COVID_FB_OptOut_Request_Date", "min"),
                              board=("ServicingTransferDate", "min"), disp=("DispositionPath", "last"),
                              dq_apr21=("FiservDelqStatusMBA", "last"), occ=("OccupancyDescription", "last"),
                              moddate=("ModificationDate", "max"), modtype=("ModificationType", "last"))
c2 = inq.groupby("LoanID").agg(inq_first=("COVID_Inquiry_Date", "min"), inq_n=("COVID_Inquiry_Date", "nunique"))
c1 = c1.join(c2); c1["in_covid_file"] = 1
log("covid-file loans", len(c1), "with inquiry", c1["inq_first"].notna().sum(), "with FB", c1["fb_agree"].notna().sum())

L = feb.join(attr, how="inner").join(resp).join(postc).join(c1)
L["in_covid_file"] = L["in_covid_file"].fillna(0).astype(int)
L.to_parquet(os.path.join(OUT, "loan_frame.parquet"))
log("frame", L.shape, "Gov", int(L.Gov.sum()), "linked", int(L.in_covid_file.sum()))
print(L.groupby("Gov")[["in_covid_file", "dq_feb20", "pre_in", "pre_out", "post_months"]].mean().round(3))
print("linked by Gov x dq:\n", L.groupby(["Gov", "dq_feb20"])["in_covid_file"].agg(["mean", "size"]).round(3))
lk = L[L.in_covid_file == 1]
print("inquiry / FB among linked:\n", lk.assign(inq=lk.inq_first.notna(), fb=lk.fb_agree.notna()).groupby(["Gov"])[["inq", "fb"]].mean().round(3))
print("FB | inquiry:\n", lk[lk.inq_first.notna()].assign(fb=lambda d: d.fb_agree.notna()).groupby("Gov")["fb"].agg(["mean", "size"]).round(3))
print("FB | no inquiry:\n", lk[lk.inq_first.isna()].assign(fb=lambda d: d.fb_agree.notna()).groupby("Gov")["fb"].agg(["mean", "size"]).round(3))
