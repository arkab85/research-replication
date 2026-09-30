"""Where did the loans go?

Every vested option resolves into one of five states within the disclosure's tracking
window. The paper so far has studied one branch (repurchase). If the buyout channel
closed for nonbanks, the loans had to go somewhere: cured and paid off, into agency
loss mitigation, into foreclosure with an FHA claim, or nowhere -- still delinquent,
still in the pool, still being advanced on.

Which of those it was determines whether intermediary constraints REALLOCATE distressed
assets or FREEZE them, and whether the cost landed on the insurer.
"""
import pandas as pd, numpy as np, os, json
from config import OUT, EXTRACT as DATA, ISSUER_TYPES


LAB = {0: "still in pool", 1: "payoff", 2: "repurchase",
       3: "foreclosure w/ claim", 4: "loss mitigation", 5: "substitution", 6: "other"}

USE = ["as_of_date", "issuer_id", "removal_in_n_mth_code", "removal_reason",
       "current_liquidation_flag", "months_dlq", "agency", "interest_rate",
       "fb_flag", "upb", "year", "state"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d["res"] = d.removal_in_n_mth_code.map(LAB)

iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
base = pd.read_csv(ISSUER_TYPES)
BM = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
tmap = {**{k: BM.get(v) for k, v in zip(base.IssuerID, base.bank_type)},
        **iss.ext.to_dict()}
d["ext"] = d.issuer_id.map(tmap)

def line(m): print("\n" + "=" * 86); print(m); print("=" * 86)

line("A. THE FULL DISPOSITION TREE (all vested options, 2013-2020)")
t = d.res.value_counts(dropna=False)
print((t / len(d) * 100).round(2).to_string())

line("B. DISPOSITION BY ISSUER TYPE, BEFORE vs AFTER MARCH 2020")
d["per"] = np.where(d.ym.between(201901, 202002), "2019-Feb2020",
           np.where(d.ym.between(202003, 202009), "Mar-Sep2020", None))
s = d[d.per.notna() & d.ext.isin(["depository", "nonbank"])]
for t_ in ["depository", "nonbank"]:
    x = s[s.ext == t_]
    tab = pd.crosstab(x.per, x.res, normalize="index") * 100
    n = x.groupby("per").size()
    print(f"\n--- {t_} ---   N: {n.to_dict()}")
    print(tab.round(1).to_string())

line("C. THE KEY QUESTION: did the closed buyout channel reallocate, or freeze?")
piv = (s.pivot_table(index=["ext", "per"], columns="res", values="ym",
                     aggfunc="size", fill_value=0))
piv = piv.div(piv.sum(axis=1), axis=0) * 100
for t_ in ["depository", "nonbank"]:
    a = piv.loc[(t_, "2019-Feb2020")]; b = piv.loc[(t_, "Mar-Sep2020")]
    ch = (b - a).sort_values()
    print(f"\n{t_}: change in disposition shares (pp)")
    for k, v in ch.items():
        print(f"    {k:<24} {v:+7.1f}   ({a.get(k,0):5.1f}% -> {b.get(k,0):5.1f}%)")

line("D. FHA CLAIMS AND LOSS MITIGATION: absolute counts")
for t_ in ["depository", "nonbank"]:
    x = s[s.ext == t_]
    g = x.groupby(["per", "res"]).size().unstack(fill_value=0)
    print(f"\n{t_}:"); print(g.to_string())

line("E. FORBEARANCE AND THE FROZEN LOANS")
f = s[s.res == "still in pool"]
print("share of 'still in pool' that were in forbearance, by type and period:")
print((f.groupby(["ext", "per"]).fb_flag.apply(
    lambda z: (z.astype(str).str.upper() == "Y").mean() * 100)).round(1).to_string())
print("\nmean UPB of frozen loans ($):")
print(s[s.res == "still in pool"].groupby(["ext", "per"]).upb.mean().round(0).to_string())

d[["ym", "issuer_id", "ext", "res", "removal_in_n_mth_code", "agency",
   "interest_rate", "fb_flag", "upb", "state"]].to_parquet(
    os.path.join(OUT, "disposition.parquet"), index=False)
print("\nwrote disposition.parquet")
