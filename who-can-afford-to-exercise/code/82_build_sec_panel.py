"""Build a public quarterly panel of unexercised Ginnie Mae buyout options.

A Ginnie Mae issuer recognises, under ASC 860-50, the unpaid balance of loans it has the
unilateral right to repurchase but has not repurchased. That balance IS the stock of
vested, unexercised options -- the paper's dependent variable, in dollars, on a public
balance sheet, quarterly, long after the loan-level extract stops.

The SEC Financial Statement Data Sets carry every tag including company extensions, so
this also discovers issuers rather than requiring a list. Downloads one quarter at a
time and keeps only the matching rows.
"""
import urllib.request, io, zipfile, os, sys, time, json
import pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA}

TAGS = {"LoansEligibleForRepurchases", "LiabilityForLoansEligibleForRepurchase",
        "LoansEligibleForRepurchaseFromAgency", "LoansEligibleForRepurchase",
        "IncreaseInUSGovernmentGuaranteedLoansEligibleForRepurchase",
        "MortgageLoansEligibleForRepurchase",
        "GinnieMaeLoansEligibleForRepurchase"}
LIQ = {"CashAndCashEquivalentsAtCarryingValue", "Assets", "StockholdersEquity",
       "ServicingAssetAtFairValueAmount"}
WANT = TAGS | LIQ

QUARTERS = [f"{y}q{q}" for y in range(2018, 2026) for q in (1, 2, 3, 4)]
QUARTERS = [q for q in QUARTERS if q <= "2025q2"]

rows, subs = [], []
for qtr in QUARTERS:
    u = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{qtr}.zip"
    try:
        b = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=300).read()
    except Exception as e:
        print(f"  {qtr}: download failed ({type(e).__name__})", flush=True); continue
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
        sub = pd.read_csv(z.open("sub.txt"), sep="\t", low_memory=False,
                          usecols=["adsh", "cik", "name", "form", "period", "fy", "fp"])
        num = pd.read_csv(z.open("num.txt"), sep="\t", low_memory=False,
                          usecols=["adsh", "tag", "version", "ddate", "qtrs",
                                   "uom", "value"])
        n = num[num.tag.isin(WANT) & (num.uom == "USD")]
        n = n.merge(sub, on="adsh", how="left")
        n["qtr"] = qtr
        rows.append(n)
        subs.append(sub.assign(qtr=qtr))
        tgt = n[n.tag.isin(TAGS)]
        print(f"  {qtr}: {len(b)/1e6:5.1f} MB  rows={len(n):>6,}  "
              f"buyout-tag rows={len(tgt):>5,}  filers={tgt.cik.nunique():>3}", flush=True)
    except Exception as e:
        print(f"  {qtr}: parse failed ({type(e).__name__}: {str(e)[:60]})", flush=True)
    time.sleep(0.4)

if not rows:
    sys.exit("nothing collected")
d = pd.concat(rows, ignore_index=True)
d.to_parquet(os.path.join(OUT, "sec_panel_raw.parquet"), index=False)
print(f"\ncollected {len(d):,} fact-rows")

t = d[d.tag.isin(TAGS)]
print(f"\nfilers reporting a loans-eligible-for-repurchase tag: {t.cik.nunique()}")
who = (t.groupby(["cik", "name"]).agg(n=("value", "size"),
                                      first=("ddate", "min"), last=("ddate", "max"))
       .sort_values("n", ascending=False))
print(who.head(40).to_string())
who.to_csv(os.path.join(OUT, "sec_panel_filers.csv"))
print("\nwrote sec_panel_raw.parquet, sec_panel_filers.csv")
