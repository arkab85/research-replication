"""Does the relief file carry the servicer's own coded description of each COVID inquiry? If so, tabulate it across 7 April 2020 by loan group. Only values occurring 10+ times are shown (codes, not free text)."""
import os, pandas as pd, re
p = r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv"; cols = pd.read_csv(p, nrows=0).columns.tolist(); print(cols)
dc = [c for c in cols if re.search(r"desc|status|lsmit|reason|type|result|outcome", c, re.I)]; print("candidate columns:", dc)
d = pd.read_csv(p, dtype=str, na_values=["NULL"], usecols=["LoanID", "COVID_Inquiry_Date"] + dc); d["dt"] = pd.to_datetime(d.COVID_Inquiry_Date, errors="coerce"); d = d[(d.dt >= "2020-03-16") & (d.dt <= "2020-05-15")]
L = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "loan_frame.parquet"))[["Gov"]]; d = d.join(L, on="LoanID").dropna(subset=["Gov"]); d["period"] = (d.dt >= "2020-04-07").map({False: "pre 7 Apr", True: "from 7 Apr"}); d["grp"] = d.Gov.map({1: "Gov", 0: "Conv"})
for c in dc:
    v = d[c].fillna("(blank)").str.strip().str[:70]; keep = v.value_counts(); keep = keep[keep >= 10].index
    if len(keep) == 0 or len(keep) > 40: print("\n", c, ": no repeated codes / too many values:", v.nunique()); continue
    t = pd.crosstab(v.where(v.isin(keep), "(other)"), [d.grp, d.period]); print("\n##", c, "(rows of the call log, 16 Mar-15 May 2020)"); print(t.to_string())
