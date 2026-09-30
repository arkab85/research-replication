"""Among Dashboard rows carrying a Seller BPO and a Servicer BPO together: date gaps and value gaps (loan-level, deduplicated)."""
import csv, os, json, math
from datetime import date
csv.field_size_limit(10**9)
OUT = os.path.join(os.path.dirname(__file__), "out")
def nn(v):
    v = (v or "").strip(); return "" if v.upper() in ("NULL", "NA", "") else v
def d(s):
    try: return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
    except Exception: return None
pairs = {}; n = 0
with open(r"<DATA>/panel\Dashboard_Dec20_2020.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        n += 1
        ht, nt = nn(row.get("HighValueType")), nn(row.get("NewValueType"))
        if {ht, nt} != {"Seller BPO", "Servicer BPO"}: continue
        key = (nn(row.get("LoanID")), nn(row.get("HighValueDate"))[:10], nn(row.get("NewValueDate"))[:10])
        if key in pairs: continue
        try: hv, nv = float(nn(row.get("HighValue"))), float(nn(row.get("NewValue")))
        except ValueError: continue
        pairs[key] = (ht, hv, nv)
gaps, logdiff, loans = [], [], set()
for (lid, hd, nd), (ht, hv, nv) in pairs.items():
    a, b = d(hd), d(nd)
    if not (a and b and hv > 0 and nv > 0): continue
    loans.add(lid); gaps.append(abs((a - b).days))
    seller, servicer = (hv, nv) if ht == "Seller BPO" else (nv, hv)
    logdiff.append(math.log(servicer / seller))
gaps.sort(); ld = sorted(logdiff); q = lambda a, p: a[int(p * (len(a) - 1))] if a else None
res = {"rows_scanned": n, "distinct_pairs": len(gaps), "loans": len(loans), "same_date": sum(g == 0 for g in gaps), "within_30d": sum(g <= 30 for g in gaps),
       "within_90d": sum(g <= 90 for g in gaps), "gap_p10": q(gaps, .1), "gap_p50": q(gaps, .5), "gap_p90": q(gaps, .9),
       "logdiff_p10": q(ld, .1), "logdiff_p50": q(ld, .5), "logdiff_p90": q(ld, .9), "abs_logdiff_p50": q(sorted(abs(x) for x in ld), .5)}
json.dump(res, open(os.path.join(OUT, "jf_pairs.json"), "w"), indent=1); print(res)
