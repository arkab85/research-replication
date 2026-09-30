"""Can the comparison be made WITHIN a pool?

In Ginnie Mae II multi-issuer pools, several issuers contribute loans and each continues
to service its own. If that structure is present in this extract, then two loans sitting
in the SAME pool in the SAME month, identical in everything the investor sees, face
different buyout probabilities purely because different issuers service them. Pool-by-month
fixed effects would then absorb the collateral, the coupon, the vintage, the geography and
any pool-level shock, and the pre-trend objection would largely go away.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, EXTRACT as DATA, ISSUER_TYPES
warnings.filterwarnings("ignore")

USE = ["as_of_date", "pool_id", "seq_num", "issuer_id", "issue_type", "pool_type",
       "bEBO", "interest_rate", "credit_score", "ltv_current", "loan_age", "state",
       "agency", "upb", "removal_in_n_mth_code"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d["buyout"] = d.bEBO.astype(str).str.upper().eq("TRUE").astype(float)

base = pd.read_csv(ISSUER_TYPES)
BM = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
tmap = {k: BM.get(v) for k, v in zip(base.IssuerID, base.bank_type)}
try:
    iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
    tmap = {**tmap, **iss.ext.to_dict()}
except Exception: pass
d["ext"] = d.issuer_id.map(tmap)
def line(m): print("\n" + "=" * 84); print(m); print("=" * 84, flush=True)

line("1. HOW MANY ISSUERS PER POOL?")
pi = d.groupby("pool_id").issuer_id.nunique()
print(f"   pools: {len(pi):,}")
print(f"   with 1 issuer:  {(pi == 1).sum():,}  ({(pi==1).mean()*100:.1f}%)")
print(f"   with 2+ issuers:{(pi > 1).sum():,}  ({(pi>1).mean()*100:.1f}%)")
print(f"   max issuers in a pool: {pi.max()}")
print("\n   by issue_type:")
for t in d.issue_type.dropna().unique():
    q = d[d.issue_type == t].groupby("pool_id").issuer_id.nunique()
    print(f"     {t}: pools={len(q):,}  multi-issuer={(q>1).sum():,} ({(q>1).mean()*100:.1f}%)"
          f"  loans={int((d.issue_type==t).sum()):,}")

line("2. POOL x MONTH CELLS WITH BOTH A NONBANK AND A DEPOSITORY")
w = d[d.ym.between(201901, 202009) & d.ext.isin(["depository", "nonbank"])].copy()
w["pm"] = w.pool_id.astype(str) + "_" + w.ym.astype(str)
g = w.groupby("pm").ext.nunique()
mixed = set(g[g > 1].index)
w["mixed"] = w.pm.isin(mixed)
print(f"   pool-month cells: {len(g):,}")
print(f"   cells containing BOTH types: {len(mixed):,} ({len(mixed)/len(g)*100:.1f}%)")
print(f"   decisions in those cells: {int(w.mixed.sum()):,} "
      f"({w.mixed.mean()*100:.1f}% of the window)")
if w.mixed.sum() > 0:
    mm = w[w.mixed]
    print(f"   issuers appearing in mixed cells: {mm.issuer_id.nunique()}")
    print(f"   raw buyout rate in mixed cells: depository "
          f"{mm[mm.ext=='depository'].buyout.mean()*100:.1f}%   nonbank "
          f"{mm[mm.ext=='nonbank'].buyout.mean()*100:.1f}%")
    pre = mm[mm.ym <= 202002]; post = mm[mm.ym >= 202003]
    for nm, x in [("2019-Feb2020", pre), ("Mar-Sep2020", post)]:
        if len(x):
            print(f"     {nm}: dep {x[x.ext=='depository'].buyout.mean()*100:5.1f}%  "
                  f"nb {x[x.ext=='nonbank'].buyout.mean()*100:5.1f}%  "
                  f"N={len(x):,}")

line("3. SAME-POOL PAIRS: are the loans comparable?")
if w.mixed.sum() > 0:
    mm = w[w.mixed]
    print(mm.groupby("ext")[["interest_rate", "credit_score", "ltv_current",
                             "loan_age", "upb"]].mean().round(2).to_string())

res = {"pools": int(len(pi)), "multi_issuer_pools": int((pi > 1).sum()),
       "multi_share": round(float((pi > 1).mean() * 100), 1),
       "mixed_cells": int(len(mixed)),
       "decisions_in_mixed": int(w.mixed.sum())}
json.dump(res, open(os.path.join(OUT, "results_withinpool.json"), "w"), indent=1)
if w.mixed.sum() > 0:
    w[w.mixed].to_parquet(os.path.join(OUT, "mixed_pools.parquet"), index=False)
    print("\nwrote mixed_pools.parquet")
print("wrote results_withinpool.json")
