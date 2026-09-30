"""The recovery test, from public filings.

'Loans eligible for repurchase' is the unpaid balance of Ginnie Mae loans on which the
issuer's buyout option has vested and which it has NOT repurchased. It is the paper's
immobilisation measure, in dollars, quarterly, and it keeps being reported long after
the loan-level extract stops.

Two things to establish:
  1. does the public series track the loan-level measure where they overlap?
  2. does it come back down after 2020, which a pre-existing trend cannot produce?
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "sec_panel_raw.parquet"))
TAGS = {"LoansEligibleForRepurchases", "LoansEligibleForRepurchase",
        "LiabilityForLoansEligibleForRepurchase", "LoansEligibleForRepurchaseFromAgency"}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

d["date"] = pd.to_datetime(d.ddate.astype(str), format="%Y%m%d", errors="coerce")
d = d.dropna(subset=["date"])
# point-in-time balances only
bal = d[(d.qtrs == 0)].copy()

line("1. THE SERIES: loans eligible for repurchase, by filer")
for cik, nm in [(1745916, "PennyMac Financial (nonbank issuer)"),
                (875357, "BOK Financial (depository issuer)")]:
    x = bal[(bal.cik == cik) & bal.tag.isin(TAGS)]
    if x.empty: print(f"\n  {nm}: none"); continue
    s = (x.sort_values("date").groupby("date").value.max() / 1e9)
    s = s[~s.index.duplicated()]
    print(f"\n  {nm}  ({len(s)} quarters, {s.index.min():%Y-%m} to {s.index.max():%Y-%m})")
    yr = s.groupby(s.index.year).max()
    for y, v in yr.items():
        bar = "#" * int(min(60, v / max(yr.max(), 1e-9) * 55))
        print(f"    {y}  ${v:7.2f}bn  {bar}")
    s.to_csv(os.path.join(OUT, f"eligible_{cik}.csv"))

line("2. PEAK AND RECOVERY")
res = {}
for cik, nm in [(1745916, "PennyMac"), (875357, "BOK Financial")]:
    x = bal[(bal.cik == cik) & bal.tag.isin(TAGS)]
    if x.empty: continue
    s = x.sort_values("date").groupby("date").value.max() / 1e9
    pre = s[(s.index >= "2019-01-01") & (s.index <= "2020-02-29")]
    peak = s[(s.index >= "2020-03-01") & (s.index <= "2021-06-30")]
    post = s[(s.index >= "2022-01-01")]
    if len(pre) and len(peak):
        print(f"\n  {nm}")
        print(f"    pre-shock mean (2019-Feb 2020) : ${pre.mean():6.2f}bn")
        print(f"    peak 2020-H1 2021              : ${peak.max():6.2f}bn   "
              f"({peak.max()/pre.mean():.1f}x pre-shock)")
        if len(post):
            print(f"    2022 onward mean               : ${post.mean():6.2f}bn   "
                  f"({post.mean()/pre.mean():.1f}x pre-shock)")
            print(f"    latest ({post.index.max():%Y-%m})            : ${post.iloc[-1]:6.2f}bn")
        res[nm] = {"pre": round(float(pre.mean()), 2),
                   "peak": round(float(peak.max()), 2),
                   "post": round(float(post.mean()), 2) if len(post) else None,
                   "peak_mult": round(float(peak.max() / pre.mean()), 2),
                   "post_mult": round(float(post.mean() / pre.mean()), 2) if len(post) else None}

line("3. DOES THE PUBLIC SERIES MATCH THE LOAN-LEVEL MEASURE IN THE OVERLAP?")
try:
    p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
    p = p.dropna(subset=["buyout"])
    pm = p[p.issuer_id == 4094].copy()          # PennyMac Loan Services
    pm["q"] = pd.PeriodIndex(pd.to_datetime(pm.ym.astype(str), format="%Y%m"), freq="Q")
    ll = pm.groupby("q").agg(not_exercised=("buyout", lambda s: (1 - s).sum()),
                             vested=("buyout", "size"))
    ll["upb_not_ex"] = pm[pm.buyout == 0].groupby("q").upb.sum() / 1e9
    x = bal[(bal.cik == 1745916) & bal.tag.isin(TAGS)]
    s = x.sort_values("date").groupby("date").value.max() / 1e9
    s.index = pd.PeriodIndex(s.index, freq="Q")
    j = ll.join(s.rename("sec_eligible"), how="inner")
    print(j.round(2).to_string())
    if len(j) > 3:
        c = j[["upb_not_ex", "sec_eligible"]].corr().iloc[0, 1]
        print(f"\n  correlation, loan-level unexercised UPB vs SEC 'loans eligible': {c:+.3f}")
        res["overlap_corr"] = round(float(c), 3)
except Exception as e:
    print("  overlap failed:", type(e).__name__, str(e)[:120])

json.dump(res, open(os.path.join(OUT, "results_recovery.json"), "w"), indent=1, default=str)
print("\nwrote results_recovery.json")
