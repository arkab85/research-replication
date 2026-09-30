"""Audit the immobilisation quantities against the reviewer's three objections.

(1) Is the $94.9bn a gross cohort sum rather than excess retention?
(2) Are loans double counted across repeat delinquency episodes?
(3) Can the data test servicing TRANSFERS directly, rather than by the flow argument?
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "disposition.parquet"))
print("columns:", list(d.columns))
print(f"rows {len(d):,}")


def line(m):
    print("\n" + "=" * 90); print(m); print("=" * 90, flush=True)

d = d[d.ext.isin(["depository", "nonbank"])].copy()
d["frozen"] = (d.res == "still in pool").astype(float)
d["bought"] = (d.res == "repurchase").astype(float)

# a loan key: Ginnie Mae sequence numbers are unique within pool
key = ["pool_id", "seq_num"] if "seq_num" in d.columns else None
print("loan key:", key)

line("1. IS THE HEADLINE A GROSS COHORT SUM?")
win = d.ym.between(202003, 202009) & (d.ext == "nonbank")
s2 = d[win & (d.frozen == 1)]
print(f"  rows classified frozen, nonbank, Mar-Sep 2020 : {len(s2):>10,}")
print(f"  gross UPB of those rows                       : ${s2.upb.sum()/1e9:>9.1f}bn   <- the headline")
if key:
    u = s2.drop_duplicates(key)
    print(f"  distinct loans among them                     : {len(u):>10,}")
    print(f"  UPB keeping one row per loan (latest)         : "
          f"${s2.sort_values('ym').drop_duplicates(key, keep='last').upb.sum()/1e9:>9.1f}bn")
    dup = len(s2) - len(u)
    print(f"  duplicate rows (same loan, repeat episode)    : {dup:>10,} "
          f"({dup/len(s2)*100:.1f}%)")

line("2. EXCESS RETENTION RELATIVE TO 2019 BEHAVIOUR")
base = d[d.ym.between(201901, 201912) & (d.ext == "nonbank")]
post = d[win]
r19, r20 = base.frozen.mean(), post.frozen.mean()
print(f"  nonbank share left in pool, 2019              : {r19*100:>9.1f}%")
print(f"  nonbank share left in pool, Mar-Sep 2020      : {r20*100:>9.1f}%")
print(f"  excess retention rate                         : {(r20-r19)*100:>+9.1f}pp")
tot_upb = post.upb.sum() / 1e9
print(f"  UPB reaching the threshold, nonbank, Mar-Sep  : ${tot_upb:>9.1f}bn")
print(f"  EXCESS UPB left in pool = (r20-r19) x total   : ${(r20-r19)*tot_upb:>9.1f}bn")
# UPB-weighted version, which is the right one for a dollar claim
w19 = np.average(base.frozen, weights=base.upb)
w20 = np.average(post.frozen, weights=post.upb)
print(f"  UPB-weighted 2019 / 2020 frozen share         : {w19*100:.1f}% / {w20*100:.1f}%")
print(f"  EXCESS UPB, upb-weighted                      : ${(w20-w19)*tot_upb:>9.1f}bn")
if key:
    pu = post.sort_values("ym").drop_duplicates(key, keep="last")
    bu = base.sort_values("ym").drop_duplicates(key, keep="last")
    w19u = np.average(bu.frozen, weights=bu.upb)
    w20u = np.average(pu.frozen, weights=pu.upb)
    print(f"  deduplicated, upb-weighted excess             : "
          f"${(w20u-w19u)*pu.upb.sum()/1e9:>9.1f}bn")

line("3. CAN WE TEST SERVICING TRANSFERS DIRECTLY?")
if key:
    rep = d[d.duplicated(key, keep=False)].sort_values(key + ["ym"])
    nloans = rep.drop_duplicates(key).shape[0]
    print(f"  loans appearing more than once                : {nloans:>10,}")
    g = rep.groupby(key).issuer_id
    changed = g.nunique() > 1
    print(f"  of those, loans whose issuer_id CHANGED       : {int(changed.sum()):>10,} "
          f"({changed.mean()*100:.2f}%)")
    # did transfers rise after the shock?
    first = rep.groupby(key).agg(i0=("issuer_id", "first"), i1=("issuer_id", "last"),
                                 y0=("ym", "first"), y1=("ym", "last"),
                                 e0=("ext", "first"), e1=("ext", "last"))
    first["moved"] = (first.i0 != first.i1).astype(int)
    first["span_post"] = (first.y1 >= 202003).astype(int)
    print("\n  transfer rate among repeat loans, by whether the second episode is post-shock:")
    print(first.groupby("span_post").moved.agg(["size", "mean"]).round(4).to_string())
    print("\n  where nonbank-serviced loans went, when they moved:")
    mv = first[(first.moved == 1) & (first.e0 == "nonbank")]
    print(mv.e1.value_counts().to_string())
    print(f"\n  nonbank->depository transfers, post-shock episodes: "
          f"{int(((mv.e1=='depository') & (mv.span_post==1)).sum()):,}")
    res_t = {"repeat_loans": int(nloans), "changed_issuer": int(changed.sum()),
             "changed_pct": round(float(changed.mean()*100), 2),
             "nb_to_dep": int((mv.e1 == "depository").sum())}
else:
    res_t = {}
    print("  no loan key available")

json.dump({"gross_bn": round(float(s2.upb.sum()/1e9), 1),
           "excess_bn": round(float((w20-w19)*tot_upb), 1),
           "r19": round(float(r19), 4), "r20": round(float(r20), 4),
           **res_t},
          open(os.path.join(OUT, "results_audit_quantities.json"), "w"), indent=1)
print("\nwrote results_audit_quantities.json")
