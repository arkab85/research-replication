"""(1) Is there loan-level follow-up after April 2021? (2) Epsilon household fields available for the RD sample?"""
import os, csv, pandas as pd, numpy as np
csv.field_size_limit(10**9)
OUT = os.path.join(os.path.dirname(__file__), "out")
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); S = set(D.index[(D.r.abs() <= 21)]); C14 = set(D.index[(D.r.abs() <= 14) & (D.Gov == 0)])
print("RD sample (|r|<=21):", len(S), " conventional |r|<=14:", len(C14))
# ---- (1) Monthly_Loan_Detail coverage after 2020 ----
m = pd.read_csv(r"<DATA>/panel\Monthly_Loan_Detail.csv", usecols=["LoanID", "cutoff_date", "DealName", "dq_counter", "fc_flag", "reo_flag", "mod_flag", "current_balance", "DetailType"], dtype=str, na_values=["NULL"])
m = m[m.LoanID.isin(S)]; m["ym"] = m.cutoff_date.str.slice(0, 7)
print("\nMLD rows for RD sample:", len(m)); cov = m.groupby("ym").LoanID.nunique(); print("loans per month (2019-10 on):"); print(cov[cov.index >= "2019-10"].to_string())
print("DealName sample (latest months):", m[m.ym >= "2020-10"].DealName.value_counts().head(8).to_dict())
# ---- other candidates with later dates ----
for p, cols in [(r"<DATA>/relief\WBO2021.csv", None), (r"<DATA>/panel\HardSofttotal.csv", ["loanid", "yearmonth", "loanstatus"])]:
    if not os.path.exists(p): continue
    with open(p, encoding="utf-8", errors="replace") as f: h = [x.strip().lstrip("\ufeff") for x in next(csv.reader(f))]
    print("\n", os.path.basename(p), len(h), "cols; date-like:", [x for x in h if any(k in x.lower() for k in ("date", "month", "asof", "cutoff", "period", "run"))][:20])
w = pd.read_csv(r"<DATA>/relief\WBO2021.csv", dtype=str, nrows=200000, low_memory=False); w.columns = [c.strip().lstrip("\ufeff") for c in w.columns]
idc = [c for c in w.columns if c.lower() in ("loanid", "loan_id", "loan", "loannumber")]; print("WBO id cols:", idc, "| rows:", len(w))
for c in [c for c in w.columns if any(k in c.lower() for k in ("date", "cutoff", "asof"))][:12]: print("   ", c, "->", w[c].dropna().min(), "..", w[c].dropna().max())
if idc: print("   overlap with RD sample:", w[idc[0]].isin(S).sum())
hs = pd.read_csv(r"<DATA>/panel\HardSofttotal.csv", usecols=["loanid", "yearmonth", "loanstatus"], dtype=str); hs = hs[hs.loanid.isin(S)]
print("\nHardSofttotal for RD sample: months", hs.yearmonth.min(), "..", hs.yearmonth.max(), "| loans by month (2020-09 on):", hs[hs.yearmonth >= "202009"].groupby("yearmonth").loanid.nunique().to_dict())
print("   status values Dec-2020:", hs[hs.yearmonth == "202012"].drop_duplicates("loanid").loanstatus.value_counts().head(12).to_dict())
# ---- (2) Epsilon household fields ----
p = r"<DATA>/panel\Forbearance_Epsilon_Claritas_updated_IB_OB_Latest2023.csv"
with open(p, encoding="utf-8", errors="replace") as f: h = [x.strip().lstrip("\ufeff") for x in next(csv.reader(f))]
keys = ("income", "educ", "age", "occup", "marital", "child", "net.worth", "networth", "wealth", "home.value", "length.of.res", "ethnic", "language", "household.size", "credit", "liquid", "invest")
print("\nEpsilon-type columns:", [x for x in h if any(k in x.lower() for k in keys)][:60])
e = pd.read_csv(p, dtype=str, usecols=lambda c: c.strip().lstrip("\ufeff") in set(["LoanID"] + [x for x in h if any(k in x.lower() for k in keys)]), low_memory=False).drop_duplicates("LoanID").set_index("LoanID")
print("Epsilon loans:", len(e), "| in RD sample:", e.index.isin(S).sum(), "| conventional |r|<=14:", e.index.isin(C14).sum())
for c in list(e.columns)[:40]:
    vc = e[c].value_counts(dropna=False); print(f"   {c[:60]:<60} nonnull {e[c].notna().mean():.2f}  top: {dict(list(vc.head(5).items()))}")
