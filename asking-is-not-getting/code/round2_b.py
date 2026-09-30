"""Locate the April 2020 break in conventional completion (daily), and check who was asking on either side."""
import os, pandas as pd, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "out")
D = pd.read_parquet(os.path.join(OUT, "inq_timing.parquet")); L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet"))
D = D.join(L[["status_feb20", "AssetType", "pre_in", "pre_out", "fico", "ltv", "bal", "Investor", "pre_unemp", "pre_curtail", "board"]])
d = D[(D.inq >= "2020-03-16") & (D.inq <= "2020-05-08")].copy(); d["day"] = d.inq.dt.normalize()
t = d.groupby(["day", "Gov"]).agg(n=("fb", "size"), conv=("fb", "mean")).round(2).unstack("Gov")
print(t[(t.index >= "2020-03-23") & (t.index <= "2020-04-24")].to_string())
# data-driven break for conventional: date maximizing the drop in completion (30 Mar - 17 Apr candidates)
c = d[d.Gov == 0]; best = None
for cut in pd.date_range("2020-03-30", "2020-04-17"):
    a, b = c[(c.day < cut) & (c.day >= cut - pd.Timedelta(days=10))], c[(c.day >= cut) & (c.day < cut + pd.Timedelta(days=10))]
    if len(a) > 30 and len(b) > 30:
        drop = a.fb.mean() - b.fb.mean()
        if best is None or drop > best[1]: best = (cut, drop, len(a), len(b), a.fb.mean(), b.fb.mean())
print("\nlargest 10-day drop at:", best)
# same for days-to-agreement and by investor
print("\nconventional by investor, before/after 6 Apr:")
c2 = c.assign(post=(c.day >= "2020-04-06")); print(c2.groupby(["Investor", "post"]).agg(n=("fb", "size"), conv=("fb", "mean"), days=("days", "median")).round(2).to_string())
# composition of askers either side (conventional and gov), +-10 days around 6 Apr
w = d[(d.day >= "2020-03-27") & (d.day <= "2020-04-16")].assign(post=lambda x: (x.day >= "2020-04-06").astype(int))
w["npl"] = w.AssetType.eq("NPL").astype(int); w["hard"] = ((w.pre_unemp.fillna(0) + w.pre_curtail.fillna(0)) > 0).astype(int); w["tenure"] = (pd.Timestamp("2020-03-01") - w.board).dt.days / 30.44
print("\nbalance of askers, 27 Mar-5 Apr vs 6-16 Apr:")
print(w.groupby(["Gov", "post"])[["dq_feb20", "npl", "pre_in", "pre_out", "hard", "fico", "ltv", "bal", "tenure", "fb"]].mean().round(3).assign(n=w.groupby(["Gov", "post"]).size()).to_string())
# weekday pattern of the daily counts (density check)
print("\ndaily first-inquiry counts, conventional:", c.groupby("day").size().loc["2020-03-27":"2020-04-17"].to_dict())
# remittance header
import csv
for p in [r"<DATA>/relief\RemittanceGAAP.csv"]:
    with open(p, encoding="utf-8", errors="replace") as f: h = next(csv.reader(f)); r1 = next(csv.reader(f))
    print("\n", os.path.basename(p), len(h), "cols:", h); print("  ex:", r1)
