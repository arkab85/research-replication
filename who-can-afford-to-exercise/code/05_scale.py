"""Is the within-nonbank scale gradient a capacity result or a floor effect?

Many mid-sized nonbanks never exercise the option at all (2019 rate ~0) and therefore
cannot decline. If the scale gradient is mechanical, it disappears once the pre-period
level is held fixed or once never-exercisers are dropped.
"""
import pandas as pd, numpy as np, os, json, warnings
import statsmodels.api as sm
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
res = {}
def line(m): print("\n" + "=" * 78); print(m); print("=" * 78)

nb = iss[iss.ext == "nonbank"].copy()
dp = iss[iss.ext == "depository"].copy()
nb["log_ratio"] = np.log((nb.ebo20 + .01) / (nb.ebo19 + .01))

line("HOW MANY NONBANKS COULD FALL AT ALL?")
print(f"nonbank issuers: {len(nb)};  with 2019 exercise rate < 2%: "
      f"{(nb.ebo19 < .02).sum()};  >= 10%: {(nb.ebo19 >= .10).sum()}")
print(nb[["name", "n19", "ebo19", "ebo20", "d_ebo", "S"]]
      .sort_values("ebo19").round(3).to_string())

def xs(d, xs_, label, wt=True, y="d_ebo"):
    d = d.dropna(subset=xs_ + [y])
    X = sm.add_constant(d[xs_].astype(float))
    w = (d.n19 + d.n20) if wt else None
    m = (sm.WLS(d[y], X, weights=w) if wt else sm.OLS(d[y], X)).fit(cov_type="HC1")
    print(f"\n-- {label}  (n={len(d)}, y={y}) --")
    for k in xs_:
        print(f"   {k:<8} {m.params[k]:>8.4f} ({m.bse[k]:.4f})  p={m.pvalues[k]:.3f}")
    print(f"   R2={m.rsquared:.3f}")
    return {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4),
                float(m.pvalues[k])] for k in xs_} | {"n": int(len(d))}

line("A. NONBANKS: is the scale gradient robust to the pre-period level?")
A = {}
A["raw"]        = xs(nb, ["S"], "scale only")
A["ctrl_ebo19"] = xs(nb, ["S", "ebo19"], "scale, controlling 2019 exercise rate")
A["actives"]    = xs(nb[nb.ebo19 >= .10], ["S"], "scale, issuers exercising >=10% in 2019")
A["actives_c"]  = xs(nb[nb.ebo19 >= .10], ["S", "ebo19"], "  + control for 2019 rate")
A["logratio"]   = xs(nb[nb.ebo19 >= .10], ["S"], "proportional decline, active issuers",
                     y="log_ratio")
A["unweighted"] = xs(nb[nb.ebo19 >= .10], ["S"], "active issuers, unweighted", wt=False)
res["A_nonbank"] = A

line("B. DEPOSITORIES: the same gradient, as a placebo")
Bp = {}
Bp["raw"]     = xs(dp, ["S"], "scale only")
Bp["actives"] = xs(dp[dp.ebo19 >= .10], ["S"], "issuers exercising >=10% in 2019")
Bp["ctrl"]    = xs(dp[dp.ebo19 >= .10], ["S", "ebo19"], "  + control for 2019 rate")
res["B_depository"] = Bp

line("C. LOAN LEVEL, ACTIVE ISSUERS ONLY (2019 exercise rate >= 10%)")
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
CTRL = ["coupon", "fico", "cltv", "age"]
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext
act = iss[iss.ebo19 >= .10]
p2 = p.merge(iss[["S", "ebo19", "ext"]].drop(columns="ext"),
             left_on="issuer_id", right_index=True, how="inner")
p2["active"] = p2.issuer_id.isin(act.index)

C = {}
for t in ["nonbank", "depository"]:
    for sel, nm in [(p2.ext == t, "all"), ((p2.ext == t) & p2.active, "active only")]:
        d = p2[sel].copy()
        if d.issuer_id.nunique() < 5: continue
        mu, sd = d.S.mean(), d.S.std()
        d["S_z"] = (d.S - mu) / sd
        d["xp"] = d.S_z * d.post
        m = pf.feols(f"buyout ~ xp + {' + '.join(CTRL)} | issuer_id + state_month",
                     data=d, vcov={"CRV1": "issuer_id"})
        c, se, pv = m.coef()["xp"], m.se()["xp"], m.pvalue()["xp"]
        print(f"   {t:<11} {nm:<12} scale x Post = {c:>8.4f} ({se:.4f}) p={pv:.3f}  "
              f"N={len(d):>8,} G={d.issuer_id.nunique()}")
        C[f"{t}_{nm}"] = {"coef": round(float(c), 4), "se": round(float(se), 4),
                          "p": float(pv), "n": int(len(d)),
                          "g": int(d.issuer_id.nunique())}
res["C_loan_level"] = C

line("D. UPB AT RISK: the dollar size of the 2019 delinquent book")
for d, t in [(nb, "nonbank"), (dp, "depository")]:
    d = d.copy(); d["logU"] = np.log(d.upb19.clip(lower=1))
    a = d[d.ebo19 >= .10]
    res[f"D_{t}"] = xs(a, ["logU"], f"{t}: log 2019 delinquent UPB, active issuers")

with open(os.path.join(OUT, "results_scale.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("\nwrote results_scale.json")
