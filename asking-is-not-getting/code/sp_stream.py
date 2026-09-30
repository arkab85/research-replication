"""Short paper, one pass over the note log. No note text is stored or printed.
(1) Text measures in windows of 2,5,10,30,60 days after each hardship inquiry (first inquiry 1 Mar-30 Jun 2020).
(2) Event-time note and marker counts around pre-pandemic loan modifications (second setting)."""
import csv, os, re
from collections import defaultdict
import pandas as pd, numpy as np
csv.field_size_limit(10**9)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
D = pd.read_parquet(os.path.join(OUT, "inq_timing.parquet")); D = D[(D.inq >= "2020-03-01") & (D.inq <= "2020-06-30")]
inq = {k: v.normalize() for k, v in D.inq.items()}; WINS = [2, 5, 10, 30, 60]
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv", usecols=["LoanID", "ServicingTransferDate", "ModificationDate"], dtype=str, na_values=["NULL"])
for c in ["ServicingTransferDate", "ModificationDate"]: cv[c] = pd.to_datetime(cv[c], errors="coerce")
g = cv.groupby("LoanID").agg(board=("ServicingTransferDate", "min"), mod=("ModificationDate", "max")).dropna()
g = g[(g["mod"] >= "2018-01-01") & (g["mod"] <= "2019-08-31") & (g.board <= g["mod"] - pd.Timedelta(days=365))]   # a year of history before, six months after, all pre-pandemic
mod = {k: v for k, v in g["mod"].items()}; print("inquirers:", len(inq), "| loans with a qualifying modification:", len(mod), flush=True)
JOB = re.compile(r"unemploy|laid[- ]?off|lay[- ]?off|furlough|lost (his|her|their|my|the)? ?job|job loss|out of work|not working|no work|business (is |was |has )?(been )?clos|reduc(ed|tion) (in |of )?(hours|income|pay|work)|hours (were |was |have been |got )?(cut|reduced)|loss of income|income (was |is |has been )?(reduced|cut|lost)|curtailment", re.I)
MK = re.compile(r"unemploy|laid off|lay ?off|lost (his|her|their)? ?job|job loss|furlough|reduced hours|curtail|\bill(ness)?\b|hospital|medical|surgery|\bsick\b|disab|cancer|divorce|separat(ed|ion)|death|passed away|deceased|funeral|custody", re.I)
W = {p: {w: [0, 0, 0] for w in WINS} for p in inq}          # notes, jobloss notes, marker notes
E = defaultdict(lambda: defaultdict(lambda: [0, 0]))        # loan -> event month -> [notes, marker notes]
n = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        n += 1
        if (row.get("Type") or "").strip() != "Servicer Note": continue
        pid = (row.get("ParentId") or "").strip(); a, b = pid in inq, pid in mod
        if not (a or b): continue
        try: d = pd.Timestamp((row.get("CreatedAt") or "")[:10])
        except Exception: continue
        t = row.get("Text") or ""; mk = None
        if a:
            k = (d - inq[pid]).days
            if -1 <= k <= 60:
                jb = bool(JOB.search(t)); mk = bool(MK.search(t))
                for w in WINS:
                    if k <= w: x = W[pid][w]; x[0] += 1; x[1] += jb; x[2] += mk
        if b:
            m = int(np.floor((d - mod[pid]).days / 30.44))
            if -12 <= m <= 5:
                if mk is None: mk = bool(MK.search(t))
                x = E[pid][m]; x[0] += 1; x[1] += mk
rows = [{"LoanID": p, "w": w, "notes": v[0], "job": v[1], "marker": v[2]} for p, d in W.items() for w, v in d.items()]
pd.DataFrame(rows).to_parquet(os.path.join(OUT, "sp_windows.parquet"))
ev = [{"LoanID": p, "m": m, "notes": v[0], "marker": v[1]} for p, d in E.items() for m, v in d.items()]
pd.DataFrame(ev).to_parquet(os.path.join(OUT, "sp_modevent.parquet")); pd.Series(mod).rename("mod").to_frame().to_parquet(os.path.join(OUT, "sp_modloans.parquet"))
print("rows", n, "| window rows", len(rows), "| event rows", len(ev), "| DONE", flush=True)
