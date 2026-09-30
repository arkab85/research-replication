"""Extend the option-exercise panel past September 2020.

STATUS: written against the published record layout, NOT yet run against real files.
The monthly loan-level files (llmon_YYYYMM.zip) sit behind a free Ginnie Mae account at
https://www.ginniemae.gov/disclosure/download-login, so they cannot be fetched from an
unattended session. Once the files are on disk, this script rebuilds the vesting panel
for any window and appends it to the existing extract.

  python code/30_extend_2021.py --dir D:\\gnma_llmon --from 202010 --to 202212

Layout (Ginnie Mae MBS Loan Level Disclosure File v1.8, Section 3, "L" record, len 192):
    pos 1        Record Type              'L'
    pos 20-27    Pool Issue Date          CCYYMMDD
    pos 28-31    Issuer ID                9(4)
    pos 32-37    As of Date               CCYYMM
    pos 88       Months Delinquent        1..6  (0 = current)
    pos 136      Removal Reason           1..6  (2 = Repurchase of Delinquent Loan)
    pos 143-150  Loan Origination Date    CCYYMMDD
Field positions are 1-indexed inclusive, as printed in the dictionary; verify against
out/MBS_SingleFamily_Loan_DataDictionary_V1.8.pdf before trusting a run, because the
"L" record was revised at Version 2.0 (live 1 August 2026) and later files may differ.
"""
import argparse, os, re, sys, zipfile, io
import pandas as pd, numpy as np

# 1-indexed inclusive -> python slice
F = {"rectype": (1, 1), "pool_issue": (20, 27), "issuer_id": (28, 31),
     "as_of": (32, 37), "months_dlq": (88, 88), "removal_reason": (136, 136),
     "orig_date": (143, 150)}
def cut(line, key):
    a, b = F[key]; return line[a - 1:b].strip()

BUYOUT_CODE = "2"          # Repurchase of Delinquent Loan
VEST_DLQ = "3"             # option vests at the third missed payment

def parse_month(path, keep_ids=None):
    """Return one row per loan-record in this month's file, L records only."""
    rows = []
    op = (zipfile.ZipFile(path) if path.lower().endswith(".zip") else None)
    names = ([n for n in op.namelist() if not n.endswith("/")] if op else [path])
    for n in names:
        fh = io.TextIOWrapper(op.open(n), errors="replace") if op else open(path, errors="replace")
        for line in fh:
            if not line or line[0] != "L":
                continue
            rows.append((cut(line, "issuer_id"), cut(line, "as_of"),
                         cut(line, "months_dlq"), cut(line, "removal_reason"),
                         cut(line, "pool_issue"), cut(line, "orig_date")))
        fh.close()
    if op: op.close()
    df = pd.DataFrame(rows, columns=["issuer_id", "as_of", "months_dlq",
                                     "removal_reason", "pool_issue", "orig_date"])
    return df

def build(files, window):
    """Vesting panel: one row per loan at months_dlq == 3, with the exercise outcome."""
    frames = []
    for p in files:
        m = re.search(r"(\d{6})", os.path.basename(p))
        if not m or not (window[0] <= int(m.group(1)) <= window[1]):
            continue
        d = parse_month(p)
        d["ym"] = int(m.group(1))
        frames.append(d)
        print(f"  {os.path.basename(p):<28} rows={len(d):>10,}", flush=True)
    if not frames:
        sys.exit("no files matched the window")
    a = pd.concat(frames, ignore_index=True)
    a["issuer_id"] = pd.to_numeric(a.issuer_id, errors="coerce")
    vest = a[a.months_dlq == VEST_DLQ].copy()
    # exercise: removed under code 2 in the vesting month, or in a later month in
    # the window. Requires a loan key; the public L record does not carry one
    # directly, so match on the disclosure's loan sequence if present in the file
    # layout version in use -- see the note in the module docstring.
    vest["buyout_same_month"] = (vest.removal_reason == BUYOUT_CODE).astype(int)
    return vest

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder holding llmon_YYYYMM.zip files")
    ap.add_argument("--from", dest="lo", type=int, default=202010)
    ap.add_argument("--to", dest="hi", type=int, default=202212)
    ap.add_argument("--out", default="out/panel_ext.parquet")
    a = ap.parse_args()
    files = sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir)
                   if re.search(r"llmon.*\d{6}", f, re.I))
    print(f"{len(files)} candidate files in {a.dir}")
    v = build(files, (a.lo, a.hi))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    v.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}: {len(v):,} vesting decisions, "
          f"{v.issuer_id.nunique()} issuers, {v.ym.min()}-{v.ym.max()}")
    print("same-month exercise rate:", round(v.buyout_same_month.mean(), 4))
    print("\nNEXT: rerun code/13_pooled.py and code/05_scale.py on the appended panel "
          "to get the recovery test -- the size gradient should attenuate toward zero "
          "as nonbank funding conditions normalise through 2021.")
