"""How far before the recorded boarding date do pre-boarding 'Servicer Note' records fall?"""
import csv, os, json
from collections import Counter
from datetime import date
import pandas as pd
csv.field_size_limit(10**9)
OUT = os.path.join(os.path.dirname(__file__), "out")
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv", usecols=["LoanID", "ServicingTransferDate"], dtype=str, na_values=["NULL"]).dropna()
board = cv.groupby("LoanID")["ServicingTransferDate"].min().str.slice(0, 10).to_dict()
def d(s): return date(int(s[:4]), int(s[5:7]), int(s[8:10]))
gaps = Counter(); n_sn_pre = 0; loans = set(); n_sn_post = 0; n_post = 0; n_pre = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        b = board.get((row.get("ParentId") or "").strip())
        if not b: continue
        v = (row.get("CreatedAt") or "")[:10]; sn = (row.get("Type") or "").strip() == "Servicer Note"
        if v >= b:
            n_post += 1; n_sn_post += sn; continue
        n_pre += 1
        if sn:
            n_sn_pre += 1; loans.add(row["ParentId"].strip())
            try: g = (d(b) - d(v)).days
            except Exception: continue
            gaps["1-3" if g <= 3 else "4-7" if g <= 7 else "8-30" if g <= 30 else "31-90" if g <= 90 else ">90"] += 1
res = {"pre_total": n_pre, "pre_servicer_notes": n_sn_pre, "pre_sn_loans": len(loans), "gaps": dict(gaps), "post_total": n_post, "post_servicer_notes": int(n_sn_post)}
json.dump(res, open(os.path.join(OUT, "preboard_sn.json"), "w"), indent=1); print(res)
