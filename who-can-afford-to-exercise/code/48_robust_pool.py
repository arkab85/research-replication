"""Robustness for the within-pool design.

The obvious objection: loans serviced by nonbanks inside these pools are worse credits
(FICO 652 vs 671, CLTV 0.78 vs 0.70, younger). If the sensitivity of buyout to credit
quality changed in 2020, that alone could produce the result. So absorb it: let every
credit-quality cell have its own time path.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 88); print(m); print("=" * 88, flush=True)

d = pd.read_parquet(os.path.join(OUT, "mixed_pools.parquet"))
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL: d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=CTRL + ["buyout"]).copy()
d["nonbank"] = (d.ext == "nonbank").astype(float)
d["post"] = (d.ym >= 202003).astype(float)
d["nb_post"] = d.nonbank * d.post
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)
d["fb_"] = pd.cut(d.fico, [0, 600, 640, 680, 720, 999], labels=False)
d["lb_"] = pd.cut(d.cltv, [-9, .6, .8, .9, 1.0, 99], labels=False)
d["cb_"] = pd.cut(d.coupon, [0, 3.5, 4, 4.5, 5, 5.5, 99], labels=False)
d["ab_"] = pd.cut(d.age, [-1, 12, 24, 48, 96, 9999], labels=False)
d["credit_cell"] = (d.fb_.astype(str) + "_" + d.lb_.astype(str) + "_" +
                    d.cb_.astype(str) + "_" + d.ab_.astype(str))
d["cell_month"] = d.credit_cell + "_" + d.ym.astype(str)
d["state_month"] = d.state.astype(str) + "_" + d.ym.astype(str)

def keep(x):
    k = x.groupby("pm").nonbank.transform(lambda s: s.nunique())
    return x[k > 1]

FLAT = keep(d[d.ym >= 201907].copy())
FULL = keep(d.copy())

line("1. THE MAIN SPECIFICATION, AND THEN ABSORBING CREDIT-QUALITY TIME PATHS")
SPECS = [
    ("issuer + pool x month",                       "issuer_id + pm", CTRL),
    ("  + credit-cell x month",                     "issuer_id + pm + cell_month", CTRL),
    ("  + state x month",                           "issuer_id + pm + state_month", CTRL),
    ("  + credit-cell x month + state x month",     "issuer_id + pm + cell_month + state_month", CTRL),
]
for nm, dat in [("FLAT WINDOW (Jul 2019-Sep 2020)", FLAT), ("FULL WINDOW", FULL)]:
    print(f"\n--- {nm} ---")
    for lab, fe, cs in SPECS:
        try:
            m = pf.feols(f"buyout ~ nb_post + {' + '.join(cs)} | {fe}",
                         data=dat, vcov={"CRV1": "issuer_id"})
            c, se, p = m.coef()["nb_post"], m.se()["nb_post"], m.pvalue()["nb_post"]
            st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
            print(f"   {lab:<44} {c:>8.4f} ({se:.4f}) {st:<3} N={int(m._N):>9,}")
            res[f"{nm}|{lab}"] = {"coef": round(float(c), 4), "se": round(float(se), 4),
                                  "p": float(p), "n": int(m._N)}
        except Exception as e:
            print(f"   {lab:<44} failed: {type(e).__name__}")

line("2. GINNIE MAE II MULTI-ISSUER POOLS ONLY (issue_type M)")
for nm, dat in [("FLAT", FLAT), ("FULL", FULL)]:
    x = keep(dat[dat.issue_type == "M"].copy())
    if len(x) < 1000: continue
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
                 data=x, vcov={"CRV1": "issuer_id"})
    print(f"   {nm}: {m.coef()['nb_post']:+.4f} ({m.se()['nb_post']:.4f}) "
          f"p={m.pvalue()['nb_post']:.4f}  N={int(m._N):,}  pools={x.pool_id.nunique():,}")
    res[f"typeM_{nm}"] = [round(float(m.coef()['nb_post']), 4),
                          round(float(m.se()['nb_post']), 4), float(m.pvalue()['nb_post'])]

line("3. DROP THE LARGEST ISSUER; DROP FHA-ONLY; WINSORISE")
big = FLAT.issuer_id.value_counts().index[0]
tests = [("excluding the largest issuer", keep(FLAT[FLAT.issuer_id != big].copy())),
         ("FHA loans only", keep(FLAT[FLAT.agency == "F"].copy())),
         ("cells with >=10 loans of each type",
          keep(FLAT.groupby("pm").filter(
              lambda g: (g.nonbank == 1).sum() >= 10 and (g.nonbank == 0).sum() >= 10)))]
for lab, x in tests:
    if len(x) < 1000: print(f"   {lab}: too few"); continue
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
                 data=x, vcov={"CRV1": "issuer_id"})
    print(f"   {lab:<38} {m.coef()['nb_post']:>8.4f} ({m.se()['nb_post']:.4f}) "
          f"p={m.pvalue()['nb_post']:.3f}  N={int(m._N):>8,}")
    res[lab] = [round(float(m.coef()['nb_post']), 4), round(float(m.se()['nb_post']), 4),
                float(m.pvalue()['nb_post'])]

line("4. THE FREEZE, WITHIN POOL x MONTH")
d["frozen"] = (d.removal_in_n_mth_code == 0).astype(float)
F = keep(d[d.ym >= 201907].copy())
for y, lab in [("frozen", "not removed from the pool"), ("buyout", "repurchased")]:
    m = pf.feols(f"{y} ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
                 data=F, vcov={"CRV1": "issuer_id"})
    print(f"   {lab:<30} {m.coef()['nb_post']:>8.4f} ({m.se()['nb_post']:.4f}) "
          f"p={m.pvalue()['nb_post']:.4f}")
    res[f"freeze_{y}"] = [round(float(m.coef()['nb_post']), 4),
                          round(float(m.se()['nb_post']), 4), float(m.pvalue()['nb_post'])]

json.dump(res, open(os.path.join(OUT, "results_robust_pool.json"), "w"),
          indent=1, default=str)
print("\nwrote results_robust_pool.json")
