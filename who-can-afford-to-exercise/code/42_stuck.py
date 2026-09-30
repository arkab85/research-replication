"""The option is tied to the owner and cannot be sold.

Standard financing-constraint results are about what the constrained FIRM does. This
asks what happens to the ASSET. A well-capitalised issuer would happily exercise these
options, but the Ginnie Mae buyout right belongs to the issuer of record and cannot be
transferred without moving the servicing itself. So the test is:

  (a) did the assets move to someone who could act?  (servicing transfers)
  (b) were the frozen options valuable ones?          (coupon of frozen vs bought)
  (c) how long did they stay frozen, and what did the advance obligation cost?
  (d) does the accumulated stock of frozen loans predict lower later exercise?
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "disposition.parquet"))
d = d[d.ext.isin(["depository", "nonbank"])].copy()
d["frozen"] = (d.res == "still in pool").astype(float)
d["bought"] = (d.res == "repurchase").astype(float)
d["coupon"] = pd.to_numeric(d.interest_rate, errors="coerce")
d["post"] = (d.ym >= 202003).astype(int)
W = d.ym.between(201901, 202009)
res = {}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

# ---------------------------------------------------------------- (a)
line("A. DID THE ASSETS MOVE TO SOMEONE WHO COULD ACT?")
cnt = d[W].groupby(["issuer_id", "ext", "ym"]).size().rename("n").reset_index()
pre = cnt[cnt.ym <= 202002].groupby(["issuer_id", "ext"]).n.mean()
post = cnt[cnt.ym >= 202003].groupby(["issuer_id", "ext"]).n.mean()
tr = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).dropna().reset_index()
tr = tr[tr.pre >= 25]
tr["ratio"] = tr.post / tr.pre
print(tr.groupby("ext").ratio.describe()[["count", "mean", "50%", "min", "max"]].round(2))
print(f"\n  nonbank issuers whose monthly vesting flow FELL: "
      f"{int(((tr.ratio < 1) & (tr.ext=='nonbank')).sum())} of "
      f"{int((tr.ext=='nonbank').sum())}")
print("  -> the options stayed where they were. No reallocation to unconstrained hands.")
res["transfers"] = {"nb_declining": int(((tr.ratio < 1) & (tr.ext == 'nonbank')).sum()),
                    "nb_total": int((tr.ext == 'nonbank').sum()),
                    "nb_median_ratio": round(float(tr[tr.ext=='nonbank'].ratio.median()), 2)}

# ---------------------------------------------------------------- (b)
line("B. WERE THE FROZEN OPTIONS THE VALUABLE ONES?")
s = d[W & (d.ext == "nonbank")].dropna(subset=["coupon"])
for per, lab in [(s.ym <= 202002, "2019-Feb 2020"), (s.ym >= 202003, "Mar-Sep 2020")]:
    x = s[per]
    print(f"\n  {lab}:  mean coupon of loans BOUGHT {x[x.bought==1].coupon.mean():.3f}"
          f"   FROZEN {x[x.frozen==1].coupon.mean():.3f}"
          f"   gap {x[x.bought==1].coupon.mean()-x[x.frozen==1].coupon.mean():+.3f}")
B = [0, 3.5, 4, 4.5, 5, 5.5, 99]; L = ["<3.5", "3.5-4", "4-4.5", "4.5-5", "5-5.5", "5.5+"]
s["cb"] = pd.cut(s.coupon, B, labels=L, right=False)
fz = s.pivot_table(index="cb", columns=s.ym >= 202003, values="frozen", observed=True) * 100
fz.columns = ["2019-Feb20", "Mar-Sep20"]; fz["change"] = (fz.iloc[:, 1] - fz.iloc[:, 0]).round(1)
print("\n  nonbank FROZEN share by coupon bin (%):"); print(fz.round(1).to_string())
print("  -> the high-coupon options, the ones most worth exercising, froze the most.")
res["frozen_by_coupon"] = fz.round(1).to_dict()

# ---------------------------------------------------------------- (c)
line("C. HOW LONG FROZEN, AND WHAT DID THE ADVANCE OBLIGATION COST?")
s2 = d[d.ym.between(202003, 202009) & (d.ext == "nonbank") & (d.frozen == 1)]
mo_left = (202009 // 100 * 12 + 202009 % 100) - (s2.ym // 100 * 12 + s2.ym % 100)
pi = s2.upb * (s2.coupon / 100) / 12                 # interest component of the advance
princ = s2.upb / 360                                  # rough scheduled principal
adv_m = (pi + princ)
print(f"  frozen nonbank loans Mar-Sep 2020: {len(s2):,}   UPB ${s2.upb.sum()/1e9:.1f}bn")
print(f"  mean monthly P&I advance per frozen loan: ${adv_m.mean():,.0f}")
print(f"  observed frozen-months in window: {mo_left.sum():,.0f}")
print(f"  implied advances on frozen loans through Sep 2020: "
      f"${(adv_m * mo_left).sum()/1e9:.2f}bn")
res["advance_cost"] = {"n": int(len(s2)), "upb_bn": round(float(s2.upb.sum()/1e9), 1),
                       "adv_per_loan_month": round(float(adv_m.mean()), 0),
                       "advances_bn": round(float((adv_m*mo_left).sum()/1e9), 2)}

# ---------------------------------------------------------------- (d)
line("D. DOES THE ACCUMULATED FROZEN STOCK PREDICT LOWER LATER EXERCISE?")
im = (d[W].groupby(["issuer_id", "ext", "ym"])
      .agg(bought=("bought", "mean"), frozen=("frozen", "mean"),
           n=("bought", "size"), upb=("upb", "sum")).reset_index())
im = im[im.n >= 25].sort_values(["issuer_id", "ym"])
im["t"] = im.ym // 100 * 12 + im.ym % 100
# stock of loans frozen in the previous 6 months, relative to the issuer's own flow
im["frz_n"] = im.frozen * im.n
g = im.groupby("issuer_id")
im["stock"] = g.frz_n.transform(lambda x: x.shift(1).rolling(6, min_periods=2).sum())
im["flow"] = g.n.transform(lambda x: x.shift(1).rolling(6, min_periods=2).sum())
im["stock_ratio"] = im.stock / im.flow.clip(lower=1)
im["log_stock"] = np.log1p(im.stock)
sp = im.dropna(subset=["stock_ratio", "log_stock"]).copy()
sp["post"] = (sp.ym >= 202003).astype(int)
for t in ["nonbank", "depository"]:
    x = sp[sp.ext == t].copy()
    x["ls_post"] = x.log_stock * x.post
    m = pf.feols("bought ~ log_stock + ls_post | issuer_id + ym", data=x,
                 weights="n", vcov={"CRV1": "issuer_id"})
    print(f"\n  {t}  (issuer-months={len(x)}, issuers={x.issuer_id.nunique()})")
    for k in ["log_stock", "ls_post"]:
        st = "***" if m.pvalue()[k] < .01 else "**" if m.pvalue()[k] < .05 else "*" if m.pvalue()[k] < .1 else ""
        print(f"     {k:<10} {m.coef()[k]:+.4f} ({m.se()[k]:.4f}) p={m.pvalue()[k]:.3f} {st}")
    res[f"stock_{t}"] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                             float(m.pvalue()[k])] for k in ["log_stock", "ls_post"]}

json.dump(res, open(os.path.join(OUT, "results_stuck.json"), "w"), indent=1, default=str)
im.to_parquet(os.path.join(OUT, "issuer_month.parquet"), index=False)
print("\nwrote results_stuck.json")
