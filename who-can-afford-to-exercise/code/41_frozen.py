"""Is the 'frozen in pool' result real, or is it right-censoring?

Loans vesting late in the extract have less follow-up, so they are mechanically more
likely to be coded 'not yet removed'. Three checks:
  (1) hold the follow-up horizon fixed
  (2) compare types within the same vesting month (censoring is common to both)
  (3) the spiral test: does an issuer's stock of frozen loans predict its SUBSEQUENT
      exercise, within issuer? That is what makes this a mechanism rather than an
      accounting identity.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "disposition.parquet"))
d = d[d.ext.isin(["depository", "nonbank"])].copy()
d["frozen"] = (d.res == "still in pool").astype(float)
d["bought"] = (d.res == "repurchase").astype(float)
END = 202009
def mdiff(a, b):
    return (b // 100 - a // 100) * 12 + (b % 100 - a % 100)
d["horizon"] = mdiff(d.ym, END)
res = {}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

line("1. FROZEN SHARE BY VESTING MONTH AND TYPE (censoring is common to both types)")
g = (d[d.ym.between(201901, 202009)].groupby(["ym", "ext"])
     .agg(frozen=("frozen", "mean"), bought=("bought", "mean"), n=("frozen", "size")))
p = (g.frozen.unstack() * 100).round(1)
b = (g.bought.unstack() * 100).round(1)
print("frozen %:"); print(p.to_string())
print("\nrepurchase %:"); print(b.to_string())
print("\nfrozen gap (nonbank - depository), pp:")
print((p.nonbank - p.depository).round(1).to_string())

line("2. FIXED FOLLOW-UP HORIZON")
for h in [3, 6, 9]:
    s = d[(d.horizon >= h) & d.ym.between(201901, 202009)]
    pre = s[s.ym <= 202002]; post = s[s.ym >= 202003]
    if len(post) == 0: continue
    print(f"\n-- at least {h} months of follow-up  (post window ends "
          f"{post.ym.max()}) --")
    for t in ["depository", "nonbank"]:
        a = pre[pre.ext == t].frozen.mean() * 100
        bb = post[post.ext == t].frozen.mean() * 100
        a2 = pre[pre.ext == t].bought.mean() * 100
        b2 = post[post.ext == t].bought.mean() * 100
        print(f"   {t:<11} frozen {a:5.1f} -> {bb:5.1f} ({bb-a:+5.1f})   "
              f"repurchase {a2:5.1f} -> {b2:5.1f} ({b2-a2:+5.1f})   "
              f"sum of changes {(bb-a)+(b2-a2):+5.1f}")
        res[f"h{h}_{t}"] = {"frozen_pre": round(a, 1), "frozen_post": round(bb, 1),
                            "bought_pre": round(a2, 1), "bought_post": round(b2, 1)}

line("3. ONE-FOR-ONE? regression of frozen on bought, within issuer x month")
im = (d[d.ym.between(201901, 202009)].groupby(["issuer_id", "ym", "ext"])
      .agg(frozen=("frozen", "mean"), bought=("bought", "mean"),
           lossmit=("res", lambda s: (s == "loss mitigation").mean()),
           payoff=("res", lambda s: (s == "payoff").mean()),
           n=("frozen", "size"), upb=("upb", "sum")).reset_index())
im = im[im.n >= 25]
im["post"] = (im.ym >= 202003).astype(int)
for t in ["nonbank", "depository"]:
    s = im[im.ext == t]
    m = pf.feols("frozen ~ bought | issuer_id + ym", data=s, weights="n",
                 vcov={"CRV1": "issuer_id"})
    print(f"   {t:<11} d(frozen)/d(bought) = {m.coef()['bought']:+.3f} "
          f"({m.se()['bought']:.3f})   issuer-months={len(s)}")
    res[f"onefor_{t}"] = [round(float(m.coef()['bought']), 3),
                          round(float(m.se()['bought']), 3)]

line("4. THE SPIRAL: does the stock of frozen loans predict LATER exercise?")
im = im.sort_values(["issuer_id", "ym"])
im["cum_frozen"] = (im.groupby("issuer_id")
                    .apply(lambda g: (g.frozen * g.n).cumsum().shift(1),
                           include_groups=False).reset_index(level=0, drop=True))
im["cum_n"] = (im.groupby("issuer_id")
               .apply(lambda g: g.n.cumsum().shift(1), include_groups=False)
               .reset_index(level=0, drop=True))
im["frozen_stock"] = im.cum_frozen / im.cum_n.clip(lower=1)
im["log_frozen_upb"] = np.log1p(im.groupby("issuer_id").upb.transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).sum()) * im.frozen.shift(1).fillna(0))
sp = im.dropna(subset=["frozen_stock"])
for t in ["nonbank", "depository"]:
    s = sp[sp.ext == t].copy()
    s["fs_post"] = s.frozen_stock * s.post
    m = pf.feols("bought ~ frozen_stock + fs_post | issuer_id + ym", data=s,
                 weights="n", vcov={"CRV1": "issuer_id"})
    print(f"\n   {t}  (issuer-months={len(s)}, issuers={s.issuer_id.nunique()})")
    for k in ["frozen_stock", "fs_post"]:
        print(f"      {k:<14} {m.coef()[k]:+.4f} ({m.se()[k]:.4f}) p={m.pvalue()[k]:.3f}")
    res[f"spiral_{t}"] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                              float(m.pvalue()[k])] for k in ["frozen_stock", "fs_post"]}

line("5. HOW MUCH WAS FROZEN? dollar magnitudes, Mar-Sep 2020")
s = d[d.ym.between(202003, 202009)]
for t in ["depository", "nonbank"]:
    x = s[(s.ext == t) & (s.frozen == 1)]
    print(f"   {t:<11} {len(x):>8,} loans frozen   ${x.upb.sum()/1e9:8.1f}bn UPB   "
          f"mean ${x.upb.mean():,.0f}")
    res[f"frozen_{t}"] = {"n": int(len(x)), "upb_bn": round(float(x.upb.sum()/1e9), 1)}
# counterfactual: nonbank frozen at the pre-period rate
pre_rate = d[(d.ext == "nonbank") & d.ym.between(201901, 202002)].frozen.mean()
nb = s[s.ext == "nonbank"]
excess = (nb.frozen.mean() - pre_rate) * len(nb)
print(f"\n   excess nonbank loans frozen vs the 2019 rate: {excess:,.0f}")
print(f"   at mean UPB, ${excess * nb[nb.frozen==1].upb.mean()/1e9:,.1f}bn of delinquent "
      f"collateral held in pools that 2019 behaviour would have removed")
res["excess_frozen_n"] = int(excess)
res["excess_frozen_upb_bn"] = round(float(excess * nb[nb.frozen == 1].upb.mean() / 1e9), 1)

json.dump(res, open(os.path.join(OUT, "results_frozen.json"), "w"), indent=1, default=str)
im.to_parquet(os.path.join(OUT, "issuer_month.parquet"), index=False)
print("\nwrote results_frozen.json, issuer_month.parquet")
