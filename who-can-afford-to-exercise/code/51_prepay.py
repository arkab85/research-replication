"""The prepayment channel: issuer funding structure as a property of the security.

A buyout is an involuntary prepayment at par. In 2020 seasoned Ginnie Mae pools traded
well above par, so every buyout returned principal to the investor at 100 on a bond worth
more than 100. When nonbank issuers stopped exercising, investors holding the pools those
issuers serviced stopped receiving those par prepayments.

Because multi-issuer pools mix issuers, the SAME security can have a high or low share of
its delinquent principal serviced by constrained issuers. That share is disclosed. So the
question is whether it predicted realised buyout-driven prepayment -- which would make
issuer funding structure a characteristic of the bond, not just of the servicer.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT, EXTRACT as DATA, ISSUER_TYPES
warnings.filterwarnings("ignore")

res = {}
def line(m): print("\n" + "=" * 88); print(m); print("=" * 88, flush=True)

USE = ["as_of_date", "pool_id", "issuer_id", "bEBO", "upb", "interest_rate",
       "issue_type", "agency", "state", "loan_age"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d = d[d.ym.between(201901, 202009)].copy()
d["buyout"] = d.bEBO.astype(str).str.upper().eq("TRUE").astype(float)
d["coupon"] = pd.to_numeric(d.interest_rate, errors="coerce")
d["upb"] = pd.to_numeric(d.upb, errors="coerce")
base = pd.read_csv(ISSUER_TYPES)
BM = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
tmap = {k: BM.get(v) for k, v in zip(base.IssuerID, base.bank_type)}
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
tmap = {**tmap, **iss.ext.to_dict()}
d["ext"] = d.issuer_id.map(tmap)
d = d.dropna(subset=["ext", "upb", "coupon"])
d["nb"] = (d.ext == "nonbank").astype(float)

line("1. POOL-MONTH AGGREGATES")
g = d.groupby(["pool_id", "ym"]).apply(lambda x: pd.Series({
    "vest_upb": x.upb.sum(),
    "buy_upb": (x.upb * x.buyout).sum(),
    "nb_share": (x.upb * x.nb).sum() / x.upb.sum(),
    "n": len(x), "coupon": np.average(x.coupon, weights=x.upb),
    "n_iss": x.issuer_id.nunique()}), include_groups=False).reset_index()
g = g[g.n >= 10].copy()
g["buy_rate"] = g.buy_upb / g.vest_upb
g["post"] = (g.ym >= 202003).astype(float)
g["nb_post"] = g.nb_share * g.post
g["cb"] = pd.cut(g.coupon, [0, 3.5, 4, 4.5, 5, 5.5, 99], labels=False)
g["cb_ym"] = g.cb.astype(str) + "_" + g.ym.astype(str)
print(f"   pool-months: {len(g):,}   pools: {g.pool_id.nunique():,}")
print(f"   mean nonbank share of delinquent principal: {g.nb_share.mean():.3f}")
print(f"   pools with a mixed issuer base (n_iss>1): "
      f"{(g.n_iss>1).mean()*100:.1f}% of pool-months")
print("\n   buyout-weighted prepayment rate by nonbank share quartile and period:")
g["q"] = pd.qcut(g.nb_share, 4, labels=["Q1 low nb", "Q2", "Q3", "Q4 high nb"],
                 duplicates="drop")
print((g.pivot_table(index="q", columns="post", values="buy_rate", observed=True)
       * 100).round(1).to_string())

line("2. DOES THE ISSUER MIX PREDICT REALISED BUYOUT PREPAYMENT?")
SP = [("pool + month",                 "pool_id + ym"),
      ("pool + coupon-bin x month",    "pool_id + cb_ym"),
      ("pool + month, mixed pools",    "pool_id + ym")]
for lab, fe in SP:
    x = g if "mixed" not in lab else g[g.n_iss > 1]
    m = pf.feols(f"buy_rate ~ nb_post | {fe}", data=x, weights="vest_upb",
                 vcov={"CRV1": "pool_id"})
    c, se, p = m.coef()["nb_post"], m.se()["nb_post"], m.pvalue()["nb_post"]
    st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    print(f"   {lab:<32} {c:>8.4f} ({se:.4f}) {st:<3} N={int(m._N):>8,}")
    res[lab] = {"coef": round(float(c), 4), "se": round(float(se), 4), "p": float(p)}

line("3. IMPLIED PREPAYMENT DIFFERENCE, ALL-NONBANK vs ALL-DEPOSITORY POOL")
m = pf.feols("buy_rate ~ nb_post | pool_id + cb_ym", data=g, weights="vest_upb",
             vcov={"CRV1": "pool_id"})
b = float(m.coef()["nb_post"])
print(f"   moving a pool from 0% to 100% nonbank-serviced delinquent principal")
print(f"   changes buyout-driven prepayment of that principal by {b*100:+.1f} pp "
      f"after March 2020")
res["implied"] = round(b, 4)

line("4. WHAT IT WAS WORTH: par prepayment out of a premium bond")
# realised buyout rate on delinquent principal, by type, Mar-Sep 2020
sub = d[d.ym >= 202003]
for t in ["depository", "nonbank"]:
    x = sub[sub.ext == t]
    r = (x.upb * x.buyout).sum() / x.upb.sum()
    print(f"   {t:<11} buyout-driven prepayment of delinquent principal: {r*100:.1f}%")
    res[f"rate_{t}"] = round(float(r * 100), 1)
gap = res["rate_depository"] - res["rate_nonbank"]
print(f"   difference: {gap:.1f} pp of delinquent principal")
print("\n   value transfer per $100 of delinquent principal, at an assumed premium of:")
for prem in [3, 5, 8]:
    print(f"     {prem} points:  ${gap/100*prem:5.2f} per $100  "
          f"(depository-serviced principal loses this much more to par takeouts)")
res["gap_pp"] = round(float(gap), 1)
res["value_5pt"] = round(float(gap / 100 * 5), 2)

line("5. DISPERSION: same coupon, same month, different issuer mix")
sub2 = g[(g.ym >= 202003)]
disp = sub2.groupby("cb").agg(
    n=("buy_rate", "size"),
    sd_buy=("buy_rate", "std"),
    sd_nb=("nb_share", "std"),
    corr=("buy_rate", lambda s: np.nan))
for cb in sub2.cb.dropna().unique():
    x = sub2[sub2.cb == cb]
    if len(x) > 30:
        disp.loc[cb, "corr"] = x.buy_rate.corr(x.nb_share)
LBL = {0: "<3.5", 1: "3.5-4", 2: "4-4.5", 3: "4.5-5", 4: "5-5.5", 5: "5.5+"}
disp.index = [LBL.get(i, i) for i in disp.index]
print(disp.round(3).to_string())
print("\n   corr = correlation across pools, within coupon bin, between the nonbank")
print("   share of delinquent principal and realised buyout prepayment, Mar-Sep 2020")
res["dispersion"] = disp.round(3).to_dict()

g.to_parquet(os.path.join(OUT, "pool_month.parquet"), index=False)
json.dump(res, open(os.path.join(OUT, "results_prepay.json"), "w"), indent=1, default=str)
print("\nwrote results_prepay.json, pool_month.parquet")
