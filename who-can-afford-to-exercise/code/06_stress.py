"""Stress-test the size gradient: threshold sensitivity, pooled contrast, wild bootstrap,
plus the remaining tests the paper listed as not yet run."""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
import statsmodels.api as sm
from config import OUT
warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260922)

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 78); print(m); print("=" * 78)

iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext
p = p.merge(iss[["S", "ebo19", "upb19"]], left_on="issuer_id", right_index=True, how="inner")

# ------------------------------------------------ threshold sensitivity
line("A. SENSITIVITY TO THE 'ACTIVE ISSUER' THRESHOLD")
A = {}
for thr in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    row = {}
    for t in ["nonbank", "depository"]:
        d = p[(p.ext == t) & (p.ebo19 >= thr)].copy()
        g = d.issuer_id.nunique()
        if g < 5: row[t] = None; continue
        d["S_z"] = (d.S - d.S.mean()) / d.S.std()
        d["xp"] = d.S_z * d.post
        m = pf.feols(f"buyout ~ xp + {' + '.join(CTRL)} | issuer_id + state_month",
                     data=d, vcov={"CRV1": "issuer_id"})
        row[t] = (round(float(m.coef()["xp"]), 4), round(float(m.se()["xp"]), 4),
                  float(m.pvalue()["xp"]), g)
    nbv, dpv = row.get("nonbank"), row.get("depository")
    f = lambda v: f"{v[0]:>8.3f} ({v[1]:.3f}) p={v[2]:.3f} G={v[3]:>2}" if v else " " * 30
    print(f"  thr>={thr:.2f}   nonbank {f(nbv)}    depository {f(dpv)}")
    A[f"thr_{thr}"] = row
res["A_threshold"] = A

# ------------------------------------------------ pooled contrast
line("B. POOLED CONTRAST: is the nonbank gradient different from the depository gradient?")
B = {}
for thr, lab in [(0.10, "active issuers (2019 rate >= 10%)"), (0.00, "all issuers")]:
    d = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= thr)].copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"]     = d.S_z * d.post
    d["nb_post"]    = d.nb * d.post
    d["S_nb"]       = d.S_z * d.nb
    d["S_nb_post"]  = d.S_z * d.nb * d.post
    m = pf.feols(f"buyout ~ S_post + nb_post + S_nb_post + {' + '.join(CTRL)}"
                 f" | issuer_id + state_month", data=d, vcov={"CRV1": "issuer_id"})
    print(f"\n-- {lab}:  N={len(d):,}  G={d.issuer_id.nunique()} --")
    for k in ["S_post", "nb_post", "S_nb_post"]:
        st = "***" if m.pvalue()[k] < .01 else "**" if m.pvalue()[k] < .05 else "*" if m.pvalue()[k] < .1 else ""
        print(f"   {k:<11} {m.coef()[k]:>8.4f} ({m.se()[k]:.4f}) p={m.pvalue()[k]:.3f} {st}")
    print(f"   implied nonbank gradient = {m.coef()['S_post'] + m.coef()['S_nb_post']:.4f}")
    B[lab] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                  float(m.pvalue()[k])] for k in ["S_post", "nb_post", "S_nb_post"]} | {
              "n": int(len(d)), "g": int(d.issuer_id.nunique())}
res["B_pooled"] = B

# ------------------------------------------------ wild cluster bootstrap
line("C. WILD CLUSTER BOOTSTRAP (Rademacher, 9999 reps) on the size gradients")
def wcb(d, xcol, reps=9999):
    d = d.copy()
    d["xp"] = d[xcol] * d.post
    fml = f"buyout ~ xp + {' + '.join(CTRL)} | issuer_id + state_month"
    m = pf.feols(fml, data=d, vcov={"CRV1": "issuer_id"})
    t0 = float(m.coef()["xp"] / m.se()["xp"])
    # restricted model (impose H0: coefficient on xp is zero)
    m0 = pf.feols(f"buyout ~ {' + '.join(CTRL)} | issuer_id + state_month",
                  data=d, vcov="iid")
    yhat0 = d.buyout.values - m0.resid()
    u0 = m0.resid()
    cl = pd.factorize(d.issuer_id)[0]
    G = cl.max() + 1
    ts = np.empty(reps)
    for b in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)[cl]
        d["_y"] = yhat0 + u0 * w
        mb = pf.feols(f"_y ~ xp + {' + '.join(CTRL)} | issuer_id + state_month",
                      data=d, vcov={"CRV1": "issuer_id"})
        ts[b] = float(mb.coef()["xp"] / mb.se()["xp"])
    return t0, float((np.abs(ts) >= abs(t0)).mean())

C = {}
for t, thr in [("nonbank", .10), ("depository", .10)]:
    d = p[(p.ext == t) & (p.ebo19 >= thr)].copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    t0, pb = wcb(d, "S_z", reps=1999)
    print(f"   {t:<11} active issuers: t={t0:>7.3f}   bootstrap p = {pb:.4f}   "
          f"G={d.issuer_id.nunique()}")
    C[t] = {"t": round(t0, 3), "boot_p": round(pb, 4), "g": int(d.issuer_id.nunique())}
res["C_bootstrap"] = C

# ------------------------------------------------ servicing transfers
line("D. SERVICING TRANSFERS: per-issuer monthly decision counts (test not run in the draft)")
cnt = (p.groupby(["issuer_id", "ym"]).size().rename("n").reset_index()
        .merge(iss[["ext"]], left_on="issuer_id", right_index=True))
pre = cnt[cnt.ym.between(201901, 201912)].groupby(["issuer_id", "ext"]).n.mean()
post = cnt[cnt.ym.between(202003, 202009)].groupby(["issuer_id", "ext"]).n.mean()
tr = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).reset_index()
tr["ratio"] = tr.post / tr.pre
print(tr.groupby("ext").ratio.describe()[["count", "mean", "50%", "min", "max"]].round(2))
print(f"\nissuers whose monthly decision flow FELL after March 2020: "
      f"{(tr.ratio < 1).sum()} of {len(tr)}")
print(tr[tr.ratio < 1].merge(iss[['name']], left_on='issuer_id', right_index=True)
      [["name", "ext", "pre", "post", "ratio"]].round(2).to_string(index=False))
res["D_transfers"] = {"n_declining": int((tr.ratio < 1).sum()), "n_issuers": int(len(tr)),
                      "median_ratio": tr.groupby("ext").ratio.median().round(2).to_dict()}

# ------------------------------------------------ reweighting
line("E. REWEIGHTING: nonbanks reweighted to the depository covariate distribution")
d = p[p.ext.isin(["nonbank", "depository"])].copy()
d["nb"] = (d.ext == "nonbank").astype(float)
X = sm.add_constant(d[CTRL].astype(float))
ps = sm.Logit(d.nb, X).fit(disp=0).predict(X)
d["w"] = np.where(d.nb == 1, (1 - ps) / ps.clip(1e-6, 1 - 1e-6), 1.0)
d["w"] = d.w.clip(upper=d.w.quantile(.99))
d["nb_post"] = d.nb * d.post
for lab, wcol in [("unweighted", None), ("IPW to depository covariates", "w")]:
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + state_month",
                 data=d, vcov={"CRV1": "issuer_id"}, weights=wcol)
    print(f"   {lab:<32} {m.coef()['nb_post']:>8.4f} ({m.se()['nb_post']:.4f})")
    res.setdefault("E_reweight", {})[lab] = [round(float(m.coef()['nb_post']), 4),
                                             round(float(m.se()['nb_post']), 4)]

with open(os.path.join(OUT, "results_stress.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("\nwrote results_stress.json")
