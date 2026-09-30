"""Direct issuer-capacity tests, run *within* issuer type.

Three predetermined issuer-level measures of how hard the advance shock hit, and of
how much balance sheet the issuer had to meet it with:

  Z_j  shift-share advance-burden exposure. Shares = issuer j's 2019 distribution of
       vested options across loan-characteristic cells; shocks = leave-one-out national
       growth in newly vested options in each cell, 2019 -> Mar-Sep 2020.
  S_j  scale: log of the issuer's 2019 option volume (facility access proxy).
  B_j  realized advance burden: growth in the issuer's own monthly vested-option count.

The capacity mechanism predicts that exercise falls more, within nonbanks, where the
shock was larger relative to the balance sheet available to absorb it, and that no such
gradient appears among depositories.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

res = {}
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
CTRL = ["coupon", "fico", "cltv", "age"]
p = p.dropna(subset=CTRL + ["buyout"]).copy()
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext

def line(m): print("\n" + "=" * 78); print(m); print("=" * 78)

# ---------------------------------------------------------------- cells
p["cb"] = pd.cut(p.coupon, [0, 3.5, 4, 4.5, 5, 5.5, 99], labels=False, right=False)
p["fb_"] = pd.cut(p.fico, [0, 600, 640, 680, 720, 999], labels=False, right=False)
p["lb"] = pd.cut(p.cltv, [-9, .6, .8, .9, 1.0, 99], labels=False, right=False)
p["ab"] = pd.cut(p.age, [-1, 12, 24, 48, 96, 9999], labels=False, right=False)
p["cell"] = (p.agency.astype(str) + "_" + p.cb.astype(str) + "_" + p.fb_.astype(str)
             + "_" + p.lb.astype(str) + "_" + p.ab.astype(str))

PRE  = p.ym.between(201901, 201912)
POST = p.ym.between(202003, 202009)
n_pre_m, n_post_m = 12, 7

# ---------------------------------------------------------------- Z_j (shift-share)
cnt_pre  = p[PRE ].groupby(["issuer_id", "cell"]).size().rename("n_pre").reset_index()
cnt_post = p[POST].groupby(["issuer_id", "cell"]).size().rename("n_post").reset_index()
cc = cnt_pre.merge(cnt_post, on=["issuer_id", "cell"], how="outer").fillna(0)
tot = cc.groupby("cell")[["n_pre", "n_post"]].sum().rename(columns=lambda c: c + "_all")
cc = cc.join(tot, on="cell")
# leave-one-out cell growth in monthly vested-option flow
cc["loo_pre"]  = (cc.n_pre_all  - cc.n_pre)  / n_pre_m
cc["loo_post"] = (cc.n_post_all - cc.n_post) / n_post_m
cc = cc[cc.loo_pre >= 5]                       # cells with a usable leave-one-out base
cc["g"] = cc.loo_post / cc.loo_pre
cc["w"] = cc.n_pre / cc.groupby("issuer_id").n_pre.transform("sum")
Z = (cc.assign(wz=cc.w * cc.g).groupby("issuer_id")
       .agg(Z=("wz", "sum"), wsum=("w", "sum"), ncell=("cell", "size")))
Z = Z[Z.wsum > 0.80]
Z["Z"] = Z.Z / Z.wsum

# ---------------------------------------------------------------- S_j, B_j
vol = pd.DataFrame({
    "n19": p[PRE ].groupby("issuer_id").size(),
    "n20": p[POST].groupby("issuer_id").size(),
    "upb19": p[PRE].groupby("issuer_id").upb.sum(),
}).fillna(0)
vol["S"] = np.log(vol.n19.clip(lower=1))
vol["B"] = (vol.n20 / n_post_m) / (vol.n19 / n_pre_m).clip(lower=.5)

iss = (p.groupby("issuer_id")
         .agg(ext=("ext", "first"),
              ebo19=("buyout", lambda s: np.nan),
              ))
iss["ebo19"] = p[PRE ].groupby("issuer_id").buyout.mean()
iss["ebo20"] = p[POST].groupby("issuer_id").buyout.mean()
iss["fb20"]  = p[POST].groupby("issuer_id").forbear.mean()
iss = iss.join(vol[["n19", "n20", "S", "B", "upb19"]]).join(Z[["Z"]])
iss["d_ebo"] = iss.ebo20 - iss.ebo19
iss = iss[(iss.n19 >= 300) & (iss.n20 >= 300)]
names = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"]
iss["name"] = iss.index.map(names)
iss.to_csv(os.path.join(OUT, "issuer_level.csv"))

line("ISSUER-LEVEL PANEL (>=300 decisions in each period)")
print(iss.ext.value_counts())
print("\nmeans by type:")
print(iss.groupby("ext")[["ebo19", "ebo20", "d_ebo", "Z", "B", "S", "fb20"]].mean().round(3))
print("\nZ dispersion within type:")
print(iss.groupby("ext").Z.describe()[["count", "mean", "std", "min", "max"]].round(3))

line("TOP NONBANK ISSUERS: exposure and retrenchment")
nb = iss[iss.ext == "nonbank"].sort_values("n19", ascending=False)
print(nb[["name", "n19", "n20", "ebo19", "ebo20", "d_ebo", "Z", "B"]]
      .head(22).round(3).to_string())

# ---------------------------------------------------------------- cross-sectional
line("A. ISSUER-LEVEL CROSS SECTION: change in exercise rate on exposure")
import statsmodels.api as sm
def xs(d, xs_, label, wt=True):
    d = d.dropna(subset=xs_ + ["d_ebo"])
    X = sm.add_constant(d[xs_].astype(float))
    w = d.n19 + d.n20 if wt else None
    m = (sm.WLS(d.d_ebo, X, weights=w) if wt else sm.OLS(d.d_ebo, X)).fit(
        cov_type="HC1")
    print(f"\n-- {label} (n={len(d)}) --")
    for k in xs_:
        print(f"   {k:<6} {m.params[k]:>8.4f} ({m.bse[k]:.4f})  p={m.pvalues[k]:.3f}")
    print(f"   R2={m.rsquared:.3f}")
    return {k: [round(float(m.params[k]), 4), round(float(m.bse[k]), 4),
                float(m.pvalues[k])] for k in xs_} | {"n": int(len(d)),
                                                      "r2": round(float(m.rsquared), 3)}
A = {}
for t in ["nonbank", "depository"]:
    d = iss[iss.ext == t]
    A[f"{t}_Z"]   = xs(d, ["Z"], f"{t}: d_ebo on Z")
    A[f"{t}_ZS"]  = xs(d, ["Z", "S"], f"{t}: d_ebo on Z, scale")
    A[f"{t}_B"]   = xs(d, ["B"], f"{t}: d_ebo on realized burden B")
res["A_cross_section"] = A

# ---------------------------------------------------------------- loan level
line("B. LOAN-LEVEL: Exposure x Post, within issuer type")
p2 = p.merge(iss[["Z", "S", "B"]], left_on="issuer_id", right_index=True, how="inner")
for c in ["Z", "S", "B"]:
    for t in ["nonbank", "depository"]:
        mu = p2.loc[p2.ext == t, c].mean(); sd = p2.loc[p2.ext == t, c].std()
        p2.loc[p2.ext == t, c + "_z"] = (p2.loc[p2.ext == t, c] - mu) / sd

def ll(d, x, label):
    d = d.copy(); d["xp"] = d[x] * d.post
    m = pf.feols(f"buyout ~ xp + {' + '.join(CTRL)} | issuer_id + state_month",
                 data=d, vcov={"CRV1": "issuer_id"})
    c, se, pv = m.coef()["xp"], m.se()["xp"], m.pvalue()["xp"]
    print(f"   {label:<52} {c:>8.4f} ({se:.4f}) p={pv:.3f} "
          f"N={len(d):>9,} G={d.issuer_id.nunique()}")
    return {"coef": round(float(c), 4), "se": round(float(se), 4), "p": float(pv),
            "n": int(len(d)), "g": int(d.issuer_id.nunique())}

B = {}
print("\n  [ coefficient on (exposure z-score) x Post ]")
for t in ["nonbank", "depository"]:
    d = p2[p2.ext == t]
    for c, nm in [("Z_z", "shift-share exposure Z"), ("B_z", "realized burden B"),
                  ("S_z", "scale (log 2019 volume)")]:
        B[f"{t}_{c}"] = ll(d, c, f"{t}: {nm} x Post")
res["B_loan_level"] = B

line("C. WITHIN-NONBANK TRIPLE DIFFERENCE: exposure x post x coupon")
d = p2[p2.ext == "nonbank"].copy()
d["xp"] = d.Z_z * d.post
d["xpc"] = d.Z_z * d.post * d.coupon
d["pc"] = d.post * d.coupon
d["xc"] = d.Z_z * d.coupon
m = pf.feols("buyout ~ xp + xpc + pc + xc + coupon + fico + cltv + age | issuer_id + state_month",
             data=d, vcov={"CRV1": "issuer_id"})
for k in ["xp", "xpc", "pc", "xc"]:
    print(f"   {k:<6} {m.coef()[k]:>8.4f} ({m.se()[k]:.4f}) p={m.pvalue()[k]:.3f}")
res["C_within_nb_ddd"] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                              float(m.pvalue()[k])] for k in ["xp", "xpc", "pc", "xc"]}

with open(os.path.join(OUT, "results_capacity.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("\nwrote results_capacity.json")
