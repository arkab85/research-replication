"""Test servicing transfers directly, instead of inferring them from the flow.

The disposition file dropped the loan key; the panel keeps pool_id and seq_num, and a
Ginnie Mae loan sequence number is stable for a loan within its pool. If a loan reaches
the buyout threshold twice and the issuer_id differs between the two records, the
servicing moved. That is a direct test of the claim the title rests on.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, EXTRACT as RAW
warnings.filterwarnings("ignore")


def line(m):
    print("\n" + "=" * 92); print(m); print("=" * 92, flush=True)

use = ["as_of_date", "pool_id", "seq_num", "issuer_id", "upb", "interest_rate",
       "credit_score", "bEBO", "removal_reason"]
d = pd.read_csv(RAW, usecols=lambda c: c in use, low_memory=False)
raw = pd.to_numeric(d.as_of_date, errors="coerce")
if raw.max() > 19000000:                                   # YYYYMMDD
    d["ym"] = (raw // 100).astype("Int64")
elif raw.max() > 190000:                                   # already YYYYMM
    d["ym"] = raw.astype("Int64")
else:
    dt = pd.to_datetime(d.as_of_date, errors="coerce")
    d["ym"] = (dt.dt.year * 100 + dt.dt.month).astype("Int64")
assert d.ym.notna().mean() > 0.99, f"ym parse failed: {d.ym.notna().mean():.3f}"
print(f"  ym range {int(d.ym.min())} to {int(d.ym.max())}")
print(f"raw rows {len(d):,}   cols {list(d.columns)}")

KEY = ["pool_id", "seq_num"]
line("0. IS (pool_id, seq_num) A STABLE LOAN KEY?")
g = d.groupby(KEY)
sz = g.size()
print(f"  distinct (pool,seq) keys        : {len(sz):>12,}")
print(f"  keys appearing more than once   : {int((sz > 1).sum()):>12,}")
rep = d[d.set_index(KEY).index.isin(sz[sz > 1].index)].copy()
if "interest_rate" in rep.columns:
    v = rep.groupby(KEY).interest_rate.nunique()
    print(f"  repeat keys with a CONSTANT note rate: {int((v == 1).sum()):,} of {len(v):,} "
          f"({(v == 1).mean()*100:.1f}%)  <- stability check")
if "credit_score" in rep.columns:
    v2 = rep.groupby(KEY).credit_score.nunique()
    print(f"  repeat keys with a constant credit score: {(v2 <= 1).mean()*100:.1f}%")

line("1. DID THE SERVICER CHANGE BETWEEN EPISODES?")
rep = rep.sort_values(KEY + ["ym"])
agg = rep.groupby(KEY).agg(n=("ym", "size"), i0=("issuer_id", "first"),
                           i1=("issuer_id", "last"), y0=("ym", "first"),
                           y1=("ym", "last"), nis=("issuer_id", "nunique"),
                           upb=("upb", "last"))
agg["moved"] = (agg.nis > 1).astype(int)
print(f"  loans with 2+ threshold episodes : {len(agg):>12,}")
print(f"  of those, servicer changed       : {int(agg.moved.sum()):>12,} "
      f"({agg.moved.mean()*100:.2f}%)")

itype = pd.read_csv(os.path.join(OUT, "issuer_counts.csv")) if os.path.exists(
    os.path.join(OUT, "issuer_counts.csv")) else None
pan = pd.read_parquet(os.path.join(OUT, "panel.parquet"),
                      columns=["issuer_id", "itype_ext"]).drop_duplicates()
m = pan.set_index("issuer_id").itype_ext
agg["e0"] = agg.i0.map(m)
agg["e1"] = agg.i1.map(m)

line("2. DID TRANSFERS RISE AFTER THE SHOCK?")
agg["post"] = (agg.y1 >= 202003).astype(int)
agg["yr1"] = agg.y1 // 100
t = agg.groupby("yr1").moved.agg(["size", "sum", "mean"])
t.columns = ["episodes", "moved", "rate"]
t["rate"] = (t.rate * 100).round(2)
print("  by calendar year of the SECOND episode:")
print(t.to_string())
t2 = agg.groupby("post").moved.agg(["size", "sum", "mean"])
t2.columns = ["episodes", "moved", "rate"]
t2["rate"] = (t2.rate * 100).round(2)
t2.index = ["before Mar 2020", "Mar 2020 onward"][:len(t2)]
print("\n  pooled:")
print(t2.to_string())

line("3. WHERE DID NONBANK-SERVICED LOANS GO WHEN THEY MOVED?")
mv = agg[(agg.moved == 1) & (agg.e0 == "nonbank")]
print(f"  nonbank-originated repeat loans that moved: {len(mv):,}")
if len(mv):
    print(mv.e1.value_counts(dropna=False).to_string())
    pm = mv[mv.post == 1]
    print(f"\n  of the post-shock ones ({len(pm):,}):")
    print(pm.e1.value_counts(dropna=False).to_string())
    nb_dep = int((pm.e1 == "depository").sum())
    share = nb_dep / max(len(agg[(agg.e0 == 'nonbank') & (agg.post == 1)]), 1)
    print(f"\n  nonbank -> depository, post-shock: {nb_dep:,} loans, "
          f"${pm[pm.e1=='depository'].upb.sum()/1e9:.2f}bn, "
          f"{share*100:.2f}% of post-shock nonbank repeat episodes")

res = {"repeat_loans": int(len(agg)), "moved": int(agg.moved.sum()),
       "moved_pct": round(float(agg.moved.mean() * 100), 2),
       "rate_pre": round(float(agg[agg.post == 0].moved.mean() * 100), 2),
       "rate_post": round(float(agg[agg.post == 1].moved.mean() * 100), 2)}
if len(mv):
    pm = mv[mv.post == 1]
    res["nb_to_dep_post"] = int((pm.e1 == "depository").sum())
    res["nb_to_dep_post_bn"] = round(float(pm[pm.e1 == "depository"].upb.sum() / 1e9), 3)
json.dump(res, open(os.path.join(OUT, "results_transfers.json"), "w"), indent=1)
agg.reset_index().to_parquet(os.path.join(OUT, "repeat_loans.parquet"), index=False)
print("\nwrote results_transfers.json")
