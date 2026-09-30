"""What happens to a loan after it is bought out?

The Ginnie Mae disclosure loses sight of a loan the moment it leaves the pool, which is
why the paper stops at the assignment. One purchaser's portfolio records pick the loan up
on the other side: a daily count of bought-out loans by servicing status, Nov 2017 to
Mar 2021, plus a monthly count of repooled and modified loans.

That covers the pandemic, so it can say what the counterfactual for an immobilised loan
looked like -- for the loans that did get out.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, DATA
warnings.filterwarnings("ignore")

res = {}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

# ---------------------------------------------------------------- status panel
s = pd.read_csv(r"F:\Ekhoni_Lagbe_Na\LiqStatus_count_Rocktop_March21.csv")
s.columns = ["n", "rundate", "status"]
s["date"] = pd.to_datetime(s.rundate, errors="coerce")
s = s.dropna(subset=["date"])
s["ym"] = s.date.dt.year * 100 + s.date.dt.month
print(f"rows {len(s):,};  {s.date.min():%Y-%m-%d} to {s.date.max():%Y-%m-%d}")
print(f"statuses: {sorted(s.status.dropna().unique())}")

# take the last observation in each month (a stock, not a flow)
last = s.sort_values("date").groupby(["ym", "status"], as_index=False).last()
piv = last.pivot(index="ym", columns="status", values="n").fillna(0)
piv["TOTAL"] = piv.sum(axis=1)
LIQ = [c for c in piv.columns if str(c).startswith("LIQ")]
ACT = [c for c in piv.columns if str(c).startswith("Active")]
piv["liq_share"] = piv[LIQ].sum(axis=1) / piv.TOTAL * 100
piv["current_share"] = piv.get("Active - Current", 0) / piv.TOTAL * 100
print(f"\nmonths: {piv.index.min()} to {piv.index.max()}")

line("1. PORTFOLIO COMPOSITION OF BOUGHT-OUT LOANS, SELECTED MONTHS")
show = [201801, 201901, 201912, 202003, 202006, 202009, 202012, 202103]
cols = [c for c in piv.columns if c not in ("TOTAL",)]
sub = piv.loc[[m for m in show if m in piv.index]]
pct = (sub[[c for c in cols if c not in ("liq_share", "current_share")]]
       .div(sub.TOTAL, axis=0) * 100).round(1)
pct["TOTAL_loans"] = sub.TOTAL.astype(int)
print(pct.to_string())

line("2. CURES AND CLAIMS THROUGH THE PANDEMIC")
for c in ["Active - Current", "LIQ - FHA Claim", "LIQ - Third Party Sale",
          "LIQ - FC Sale", "LIQ - Repurchase", "Active - 120+ Days DQ"]:
    if c in piv.columns:
        v = piv[c]
        pre = v.loc[[m for m in v.index if 201901 <= m <= 202002]].mean()
        post = v.loc[[m for m in v.index if 202003 <= m <= 202103]].mean()
        print(f"   {c:<24} mean stock  2019-Feb20 {pre:8,.0f}   Mar20-Mar21 {post:8,.0f}"
              f"   {100*(post/pre-1) if pre else float('nan'):+7.1f}%")
        res[c] = {"pre": round(float(pre), 0), "post": round(float(post), 0)}

line("3. CURE-AND-REDELIVER: the repool channel")
r = pd.read_csv(DATA / "Repool_EBO_Rocktop_March_21.csv")
r.columns = ["n", "rundate"]
r["date"] = pd.to_datetime(r.rundate, errors="coerce")
r = r.dropna(subset=["date"]); r["ym"] = r.date.dt.year * 100 + r.date.dt.month
rl = r.sort_values("date").groupby("ym").last().n
print("   cumulative repooled loans, selected months:")
for m in [201712, 201812, 201912, 202003, 202006, 202009, 202012, 202103]:
    if m in rl.index: print(f"      {m}: {rl[m]:,}")
res["repool_series"] = {int(k): int(v) for k, v in rl.items()}

x = pd.read_excel(DATA / "RT_EBO_Repool_LoanMod.xlsx")
print(f"\n   EBO / Repool / LoanMod monthly file: {x.YearMonth.min()} to {x.YearMonth.max()}")
x["repool_rate"] = x.Repool / x.EBO * 100
x["mod_rate"] = x.LoanMod_Stock / x.EBO * 100
print(x[["YearMonth", "EBO", "Repool", "repool_rate", "LoanMod_Stock", "mod_rate"]]
      .round(1).to_string(index=False))

line("4. WHAT THIS BOUNDS")
if "LIQ - FHA Claim" in piv.columns and "Active - Current" in piv.columns:
    late = piv.loc[piv.index >= 202006]
    print(f"   Among bought-out loans held by this purchaser in mid-2020 onward:")
    print(f"     performing (current) share : {late.current_share.mean():.1f}%")
    print(f"     liquidated share           : {late.liq_share.mean():.1f}%")
    res["late_current"] = round(float(late.current_share.mean()), 1)
    res["late_liq"] = round(float(late.liq_share.mean()), 1)
print("\n   A loan left in the pool by a constrained issuer did not get this treatment.")
print("   One purchaser is not the market, and these are stocks rather than matched")
print("   outcomes, so this bounds the counterfactual -- it does not identify it.")

piv.to_csv(os.path.join(OUT, "postbuyout_status.csv"))
json.dump(res, open(os.path.join(OUT, "results_postbuyout.json"), "w"),
          indent=1, default=str)
print("\nwrote postbuyout_status.csv, results_postbuyout.json")
