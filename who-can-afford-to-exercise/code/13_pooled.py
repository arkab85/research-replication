"""Pooled size-gradient contrast and the activity-threshold sensitivity table."""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

CTRL = ["coupon", "fico", "cltv", "age"]
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext
p = p.merge(iss[["S", "ebo19"]], left_on="issuer_id", right_index=True, how="inner")
res = {}

print("=" * 78); print("POOLED CONTRAST"); print("=" * 78)
B = {}
for thr, lab in [(.10, "active issuers (2019 rate >= 10%)"), (.00, "all issuers")]:
    d = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= thr)].copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"] = d.S_z * d.post; d["nb_post"] = d.nb * d.post
    d["S_nb_post"] = d.S_z * d.nb * d.post
    m = pf.feols(f"buyout ~ S_post + nb_post + S_nb_post + {' + '.join(CTRL)}"
                 f" | issuer_id + state_month", data=d, vcov={"CRV1": "issuer_id"})
    print(f"\n-- {lab}: N={len(d):,} G={d.issuer_id.nunique()}")
    for k in ["S_post", "nb_post", "S_nb_post"]:
        print(f"   {k:<11} {m.coef()[k]:>8.4f} ({m.se()[k]:.4f}) p={m.pvalue()[k]:.4f}")
    print(f"   implied nonbank gradient = {m.coef()['S_post']+m.coef()['S_nb_post']:.4f}")
    B[lab] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                  float(m.pvalue()[k])] for k in ["S_post", "nb_post", "S_nb_post"]}
    B[lab]["n"] = int(len(d)); B[lab]["g"] = int(d.issuer_id.nunique())
res["B_pooled"] = B

print("\n" + "=" * 78); print("THRESHOLD SENSITIVITY"); print("=" * 78)
A = {}
for thr in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    row = {}
    for t in ["nonbank", "depository"]:
        d = p[(p.ext == t) & (p.ebo19 >= thr)].copy()
        g = d.issuer_id.nunique()
        if g < 5: row[t] = None; continue
        d["S_z"] = (d.S - d.S.mean()) / d.S.std(); d["xp"] = d.S_z * d.post
        m = pf.feols(f"buyout ~ xp + {' + '.join(CTRL)} | issuer_id + state_month",
                     data=d, vcov={"CRV1": "issuer_id"})
        row[t] = (round(float(m.coef()["xp"]), 4), round(float(m.se()["xp"]), 4),
                  float(m.pvalue()["xp"]), g)
    f = lambda v: f"{v[0]:>8.3f} ({v[1]:.3f}) p={v[2]:.3f} G={v[3]:>2}" if v else " " * 30
    print(f"  thr>={thr:.2f}   nonbank {f(row.get('nonbank'))}"
          f"    depository {f(row.get('depository'))}")
    A[f"thr_{thr}"] = row
res["A_threshold"] = A

json.dump(res, open(os.path.join(OUT, "results_stress.json"), "w"), indent=1, default=str)
print("\nwrote results_stress.json")
