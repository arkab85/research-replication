"""journal boarding-depth table.
For every loan with a ServicingTransferDate (boarding into CMS): months of CMS narrative
accumulated by 2020-03-01, count of Servicer Notes in [boarding, 2020-03), illustrative hardship
markers in those notes, and resolution outcomes from the April-2021 snapshot.
Markers are MY simple regexes (labelled illustrative) - not the paper's prespecified ones."""
import os, re, csv, json, time
from collections import defaultdict
import numpy as np, pandas as pd
csv.field_size_limit(10**9)
OUT = os.path.join(os.path.dirname(__file__), "out"); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print("[%5.0fs]" % (time.time()-T0), *a, flush=True)

MARK = {
 "employment": re.compile(r"unemploy|laid off|lay ?off|lost (his|her|their)? ?job|job loss|furlough|reduced hours|curtail", re.I),
 "health":     re.compile(r"\bill(ness)?\b|hospital|medical|surgery|\bsick\b|covid|disab|cancer", re.I),
 "family":     re.compile(r"divorce|separat(ed|ion)|death|passed away|deceased|funeral|custody", re.I),
 "property":   re.compile(r"hurricane|flood|\bfire\b|storm|tornado|damage|roof|repair", re.I),
}
CUT = pd.Timestamp("2020-03-01")

log("snapshot from COVID file")
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv",
                 usecols=["LoanID", "ServicingTransferDate", "DispositionPath", "ModFLag", "FCFlag",
                          "BKFLag", "REOFlag", "FB_Agreement_Date", "FiservDelqStatusMBA", "LoanStatusCurrent",
                          "ModificationDate", "DateFCLReferred", "AmortType"],
                 dtype=str, na_values=["NULL"])
for c in ["ServicingTransferDate", "FB_Agreement_Date", "ModificationDate", "DateFCLReferred"]:
    cv[c] = pd.to_datetime(cv[c], errors="coerce")
snap = cv.groupby("LoanID").agg(board=("ServicingTransferDath" if False else "ServicingTransferDate", "min"),
                                disp=("DispositionPath", "last"), mod=("ModFLag", "last"), fc=("FCFlag", "last"),
                                bk=("BKFLag", "last"), fb=("FB_Agreement_Date", "min"),
                                dq=("FiservDelqStatusMBA", "last"), moddate=("ModificationDate", "max"),
                                fcdate=("DateFCLReferred", "max"))
snap = snap.dropna(subset=["board"])
log("boarded loans:", len(snap))

# Feb-2020 delinquency from IB_OB (pre-COVID baseline)
ib = pd.read_csv(r"<DATA>/panel\IB_OB_Flags_Full_Nov3.csv",
                 usecols=["LoanId", "Month t", "LoanStatus"], dtype=str)
ib = ib[ib["Month t"] == "2/29/2020"].drop_duplicates("LoanId").set_index("LoanId")["LoanStatus"]
snap["status_feb20"] = snap.index.map(ib)

v_panel = set(pd.read_csv(r"<DATA>", usecols=["LoanID"], dtype=str)["LoanID"])
snap["v_panel"] = snap.index.isin(v_panel)

# ---------- stream the 20M-row note log (cached) ----------
CACHE = os.path.join(OUT, "v_loan_level.parquet")
if os.path.exists(CACHE):
    log("loading cached loan-level counts")
    cached = pd.read_parquet(CACHE)
    for c in cached.columns: snap[c] = cached[c].reindex(snap.index)
    STREAM = False
else:
    STREAM = True
log("streaming Messages_Dec20 (20M rows)" if STREAM else "skip stream")
board = snap["board"].dt.strftime("%Y-%m-%d").to_dict() if STREAM else {}
notes = defaultdict(int); marked = defaultdict(int)
bykind = {k: defaultdict(int) for k in MARK}
notes_post = defaultdict(int)   # notes 2020-03..2020-12 (for reference)
n = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f)
    rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        n += 1
        if n % 4_000_000 == 0: log("  ...", n)
        pid = (row.get("ParentId") or "").strip()
        b = board.get(pid)
        if not b: continue
        if (row.get("Type") or "").strip() != "Servicer Note": continue
        d = (row.get("CreatedAt") or "")[:10]
        if d < b: continue
        if d >= "2020-03-01":
            notes_post[pid] += 1; continue
        notes[pid] += 1
        t = row.get("Text") or ""
        hit = False
        for k, rx in MARK.items():
            if rx.search(t):
                bykind[k][pid] += 1; hit = True
        if hit: marked[pid] += 1
log("done streaming", n)

if STREAM:
    snap["notes_pre"] = snap.index.map(notes).fillna(0).astype(int)
    snap["marked_pre"] = snap.index.map(marked).fillna(0).astype(int)
    snap["notes_post"] = snap.index.map(notes_post).fillna(0).astype(int)
    for k in MARK: snap["mk_" + k] = snap.index.map(bykind[k]).fillna(0).astype(int)
    snap[["notes_pre", "marked_pre", "notes_post"] + ["mk_" + k for k in MARK]].to_parquet(CACHE)
    log("cached loan-level counts")
snap["months_hist"] = ((CUT - snap["board"]).dt.days / 30.44).clip(lower=0).round(1)
snap["cohort"] = snap["board"].dt.to_period("Q").astype(str)
snap.loc[snap["board"] >= CUT, "cohort"] = "boarded after Mar-2020"
snap["fb_any"] = snap["fb"].notna()
snap["mod_any"] = snap["mod"].eq("Y")
snap["mod_post"] = snap["moddate"] >= CUT
snap["fc_any"] = snap["fc"].eq("Y")
snap["fc_post"] = snap["fcdate"] >= CUT
snap["performing"] = snap["disp"].eq("Performing")
snap["dq_feb20"] = snap["status_feb20"].fillna("").str.contains("DQ|FC|BK")

def agg(g):
    return pd.Series({
        "loans": len(g), "months_hist_med": g["months_hist"].median(),
        "notes_med": g["notes_pre"].median(), "notes_mean": g["notes_pre"].mean(),
        "any_marker_pct": 100 * (g["marked_pre"] > 0).mean(),
        "emp_pct": 100 * (g["mk_employment"] > 0).mean(), "health_pct": 100 * (g["mk_health"] > 0).mean(),
        "family_pct": 100 * (g["mk_family"] > 0).mean(), "property_pct": 100 * (g["mk_property"] > 0).mean(),
        "dq_feb20_pct": 100 * g["dq_feb20"].mean(),
        "fb_pct": 100 * g["fb_any"].mean(), "mod_post_pct": 100 * g["mod_post"].mean(),
        "fc_post_pct": 100 * g["fc_post"].mean(), "performing_pct": 100 * g["performing"].mean(),
    })

pre = snap[snap["board"] < CUT].copy()
def gapply(df, keys):
    # keep grouping columns visible to agg()
    return df.groupby(keys, observed=True)[df.columns.tolist()].apply(agg).round(1)
coh = gapply(pre, "cohort")
coh.to_csv(os.path.join(OUT, "v_cohort.csv"))
# depth terciles within Feb-2020 delinquency status
pre["depth_tercile"] = pd.qcut(pre["months_hist"], 3, labels=["short", "mid", "long"])
dep = gapply(pre, ["dq_feb20", "depth_tercile"])
dep.to_csv(os.path.join(OUT, "depth_analysis_by_status.csv"))
# depth terciles within Feb-2020 status, restricted to loans with >=1 note (drops the no-notes mass)
dep_notes = gapply(pre[pre["notes_pre"] > 0], ["dq_feb20", "depth_tercile"])
dep_notes.to_csv(os.path.join(OUT, "depth_analysis_by_status_withnotes.csv"))
panel = gapply(pre[pre["v_panel"]], "cohort")
panel.to_csv(os.path.join(OUT, "v_cohort_panel4251.csv"))
overall = agg(pre).round(2)

out = {"cohort": coh.reset_index().to_dict("records"),
       "depth_by_status": dep.reset_index().to_dict("records"),
       "depth_by_status_withnotes": dep_notes.reset_index().to_dict("records"),
       "panel4251": panel.reset_index().to_dict("records"),
       "overall": overall.to_dict(),
       "n_boarded_pre": int(len(pre)), "n_boarded_post": int((snap["board"] >= CUT).sum()),
       "hist_quantiles": pre["months_hist"].quantile([.1, .25, .5, .75, .9]).round(1).to_dict(),
       "notes_quantiles": pre["notes_pre"].quantile([.1, .25, .5, .75, .9]).to_dict(),
       "zero_notes_pct": float(100 * (pre["notes_pre"] == 0).mean()),
       "corr_hist_notes": float(pre[["months_hist", "notes_pre"]].corr().iloc[0, 1])}
with open(os.path.join(OUT, "v_results.json"), "w") as f: json.dump(out, f, indent=1, default=float)
print(coh.to_string()); print(); print(dep.to_string()); print(); print(overall.to_string())
log("DONE")
