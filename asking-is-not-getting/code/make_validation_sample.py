"""VALIDATION KIT (author runs this; it writes note text, so keep the output on the secure drive).
Draws a blinded, stratified random sample of servicer notes for hand coding of the four keyword measures used in the paper.
Usage:  python make_validation_sample.py  <output_folder_on_secure_drive>
Writes: to_code.csv  (note_id, text, and four empty columns for the human coder: docs, ineligible, repayment, jobloss -> enter 1/0)
        key_DO_NOT_OPEN_UNTIL_CODED.csv  (note_id and the machine flags)
Then run score_validation.py on the same folder."""
import csv, os, re, sys, random
import pandas as pd
csv.field_size_limit(10**9); random.seed(20200407)
dst = sys.argv[1] if len(sys.argv) > 1 else None
if not dst: sys.exit("Give an output folder on the secure drive.")
os.makedirs(dst, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
cand = [os.path.join(HERE, "out", "inq_timing.parquet"), os.path.join(HERE, "..", "results", "inq_timing.parquet")]
D = pd.read_parquet(next(p for p in cand if os.path.exists(p))); D = D[(D.Gov == 0) & (D.inq >= "2020-03-16") & (D.inq <= "2020-05-31")]
lo = {k: (v - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for k, v in D.inq.items()}; hi = {k: (v + pd.Timedelta(days=60)).strftime("%Y-%m-%d") for k, v in D.inq.items()}
RX = {"docs": r"\brma\b|request for mortgage assistance|financial (package|docs|documents|information)|hardship (letter|affidavit|application)|proof of income|pay ?stubs?|bank statements?|tax returns?|4506|(docs?|documents?|documentation) (are |is |was |were )?(needed|required|requested|missing|not received)|loss mit(igation)? (package|application)|workout package|complete (the )?(package|application)",
      "ineligible": r"\bden(y|ied|ial)\b|not eligible|ineligible|does ?n[o']t qualify|not qualify|declin(e|ed)|not approved|unable to (offer|approve|assist)",
      "repayment": r"repayment plan|\brpp?\b|reinstate|promise to pay|\bptp\b|payment arrangement",
      "jobloss": r"unemploy|laid[- ]?off|lay[- ]?off|furlough|lost (his|her|their|my|the)? ?job|job loss|out of work|not working|no work|business (is |was |has )?(been )?clos|reduc(ed|tion) (in |of )?(hours|income|pay|work)|hours (were |was |have been |got )?(cut|reduced)|loss of income|income (was |is |has been )?(reduced|cut|lost)|curtailment"}
RX = {k: re.compile(v, re.I) for k, v in RX.items()}
pool = {(k, f): [] for k in RX for f in (0, 1)}; n = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        pid = (row.get("ParentId") or "").strip()
        if pid not in lo or (row.get("Type") or "").strip() != "Servicer Note": continue
        v = (row.get("CreatedAt") or "")[:10]
        if not (lo[pid] <= v <= hi[pid]): continue
        t = row.get("Text") or ""; n += 1; flags = {k: int(bool(rx.search(t))) for k, rx in RX.items()}
        for k in RX:                                   # reservoir sample, 60 per (pattern, flag) cell
            cell = pool[(k, flags[k])]
            if len(cell) < 60: cell.append((row.get("Id"), t, flags))
            elif random.random() < 60 / n: cell[random.randrange(60)] = (row.get("Id"), t, flags)
seen, rows = set(), []
for cell in pool.values():
    for nid, t, fl in cell:
        if nid not in seen: seen.add(nid); rows.append((nid, t, fl))
random.shuffle(rows)
with open(os.path.join(dst, "to_code.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["note_id", "text", "docs", "ineligible", "repayment", "jobloss"]); [w.writerow([nid, t, "", "", "", ""]) for nid, t, _ in rows]
with open(os.path.join(dst, "key_DO_NOT_OPEN_UNTIL_CODED.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["note_id"] + list(RX)); [w.writerow([nid] + [fl[k] for k in RX]) for nid, _, fl in rows]
print(f"{len(rows)} notes written to {dst}. Code the four columns 1/0 without opening the key, ideally with a second coder, then run score_validation.py.")
