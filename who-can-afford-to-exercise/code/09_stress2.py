"""Corrected stress tests + a high-frequency within-issuer capacity test.

The capacity model says an issuer can fund at most K_jt repurchases in month t. If K is
sticky in the short run, then the *share* of vested options exercised falls when the
issuer's own flow of newly vested options rises. That is a within-issuer, monthly test
of the same mechanism the cross-issuer size gradient probes, with three orders of
magnitude more variation.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
import statsmodels.api as sm
from config import OUT
warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260922)

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 78, flush=True); print(m); print("=" * 78, flush=True)

iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext
p = p.merge(iss[["S", "ebo19"]], left_on="issuer_id", right_index=True, how="inner")

def drop_singletons(d):
    while True:
        n0 = len(d)
        d = d[d.groupby("issuer_id").issuer_id.transform("size") > 1]
        d = d[d.groupby("state_month").state_month.transform("size") > 1]
        if len(d) == n0: return d

# ---------------------------------------------------------------- nonparametric
line("A. THE SIMPLEST STATEMENT OF THE RESULT")
a = iss[(iss.ebo19 >= .10) & iss.ext.isin(["nonbank", "depository"])]
for t in ["depository", "nonbank"]:
    d = a[a.ext == t]
    print(f"  {t:<11}: {(d.d_ebo > 0).sum():>2} of {len(d):>2} active issuers raised their "
          f"exercise rate;  median change {d.d_ebo.median()*100:+.1f} pp")
res["A_counts"] = {t: {"up": int((a[a.ext == t].d_ebo > 0).sum()),
                       "n": int((a.ext == t).sum()),
                       "median_change_pp": round(float(a[a.ext == t].d_ebo.median() * 100), 1)}
                   for t in ["depository", "nonbank"]}

# ---------------------------------------------------------------- bootstrap
line("B. WILD CLUSTER BOOTSTRAP ON THE POOLED SIZE-GRADIENT CONTRAST")
def wcb(d, target, rhs, reps=999):
    d = drop_singletons(d.copy())
    fml_u = f"buyout ~ {' + '.join(rhs)} | issuer_id + state_month"
    m = pf.feols(fml_u, data=d, vcov={"CRV1": "issuer_id"})
    t0 = float(m.coef()[target] / m.se()[target])
    r0 = [c for c in rhs if c != target]
    m0 = pf.feols(f"buyout ~ {' + '.join(r0)} | issuer_id + state_month", data=d, vcov="iid")
    u0 = np.asarray(m0.resid()).ravel()
    assert len(u0) == len(d), (len(u0), len(d))
    yhat0 = d.buyout.values - u0
    cl = pd.factorize(d.issuer_id)[0]; G = cl.max() + 1
    ts = np.empty(reps)
    for b in range(reps):
        d["_y"] = yhat0 + u0 * rng.choice([-1.0, 1.0], size=G)[cl]
        mb = pf.feols(f"_y ~ {' + '.join(rhs)} | issuer_id + state_month",
                      data=d, vcov={"CRV1": "issuer_id"})
        ts[b] = float(mb.coef()[target] / mb.se()[target])
    return t0, float((np.abs(ts) >= abs(t0)).mean()), G

B = {}
for thr, lab in [(.10, "active issuers"), (.00, "all issuers")]:
    d = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= thr)].copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"] = d.S_z * d.post
    d["nb_post"] = d.nb * d.post
    d["S_nb_post"] = d.S_z * d.nb * d.post
    rhs = ["S_post", "nb_post", "S_nb_post"] + CTRL
    t0, pb, G = wcb(d, "S_nb_post", rhs, reps=999)
    print(f"  {lab:<16} S_nb_post  t={t0:>7.3f}   wild-bootstrap p={pb:.4f}   G={G}")
    B[lab] = {"t": round(t0, 3), "boot_p": round(pb, 4), "g": int(G)}
res["B_bootstrap"] = B

line("C. LEAVE-ONE-ISSUER-OUT on the pooled contrast (active issuers)")
d0 = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= .10)].copy()
vals = []
for drop in [None] + sorted(d0.issuer_id.unique()):
    d = d0 if drop is None else d0[d0.issuer_id != drop]
    d = d.copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"] = d.S_z * d.post; d["nb_post"] = d.nb * d.post
    d["S_nb_post"] = d.S_z * d.nb * d.post
    m = pf.feols(f"buyout ~ S_post + nb_post + S_nb_post + {' + '.join(CTRL)}"
                 f" | issuer_id + state_month", data=d, vcov={"CRV1": "issuer_id"})
    vals.append((drop, float(m.coef()["S_nb_post"]), float(m.se()["S_nb_post"]),
                 float(m.pvalue()["S_nb_post"])))
lo = pd.DataFrame(vals, columns=["dropped", "coef", "se", "p"])
lo["name"] = lo.dropped.map(iss["name"])
print(f"  full sample: {lo.coef[0]:+.4f} ({lo.se[0]:.4f})")
print(f"  leave-one-out range: [{lo.coef[1:].min():+.4f}, {lo.coef[1:].max():+.4f}]; "
      f"max p = {lo.p[1:].max():.4f}; all negative = {bool((lo.coef[1:] < 0).all())}")
print(lo.iloc[1:].reindex(lo.iloc[1:].coef.abs().sort_values().index)
      .head(4)[["name", "coef", "se", "p"]].round(4).to_string(index=False))
lo.to_csv(os.path.join(OUT, "leave_one_out.csv"), index=False)
res["C_loo"] = {"full": round(lo.coef[0], 4), "min": round(lo.coef[1:].min(), 4),
                "max": round(lo.coef[1:].max(), 4), "max_p": round(lo.p[1:].max(), 4),
                "all_neg": bool((lo.coef[1:] < 0).all())}

# ---------------------------------------------------------------- flow test
line("D. HIGH-FREQUENCY CAPACITY TEST: own monthly flow of newly vested options")
flow = p.groupby(["issuer_id", "ym"]).size().rename("N").reset_index()
p = p.merge(flow, on=["issuer_id", "ym"], how="left")
p["logN"] = np.log(p.N)
D = {}
def flowreg(d, label, extra=()):
    d = d.copy()
    rhs = ["logN"] + list(extra) + CTRL
    m = pf.feols(f"buyout ~ {' + '.join(rhs)} | issuer_id + state_month",
                 data=d, vcov={"CRV1": "issuer_id"})
    out = {}
    for k in ["logN"] + list(extra):
        c, se, pv = m.coef()[k], m.se()[k], m.pvalue()[k]
        st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
        print(f"   {label:<34} {k:<12} {c:>8.4f} ({se:.4f}) {st:<3} "
              f"N={len(d):>8,} G={d.issuer_id.nunique()}")
        out[k] = [round(float(c), 4), round(float(se), 4), float(pv)]
    return out | {"n": int(len(d)), "g": int(d.issuer_id.nunique())}

print("\n  [ elasticity of the exercise rate to the issuer's own monthly option flow ]")
for t in ["nonbank", "depository"]:
    d = p[p.ext == t]
    D[f"{t}_pre"]  = flowreg(d[d.post == 0], f"{t}, 2019 only")
    D[f"{t}_post"] = flowreg(d[d.post == 1], f"{t}, Mar-Sep 2020")
    dd = d.copy(); dd["logN_post"] = dd.logN * dd.post
    D[f"{t}_int"]  = flowreg(dd, f"{t}, pooled + interaction", extra=["logN_post"])
print("\n  [ pooled across types: is the flow elasticity type-specific? ]")
dd = p[p.ext.isin(["nonbank", "depository"])].copy()
dd["nb"] = (dd.ext == "nonbank").astype(float)
dd["logN_nb"] = dd.logN * dd.nb
dd["logN_post"] = dd.logN * dd.post
dd["logN_nb_post"] = dd.logN * dd.nb * dd.post
dd["nb_post"] = dd.nb * dd.post
D["pooled"] = flowreg(dd, "pooled", extra=["logN_nb", "logN_post", "logN_nb_post", "nb_post"])
res["D_flow"] = D

# ---------------------------------------------------------------- transfers
line("E. SERVICING TRANSFERS: per-issuer monthly decision counts")
cnt = (p.groupby(["issuer_id", "ym"]).size().rename("n").reset_index()
        .merge(iss[["ext", "name"]], left_on="issuer_id", right_index=True))
pre = cnt[cnt.ym.between(201901, 201912)].groupby(["issuer_id"]).n.mean()
post = cnt[cnt.ym.between(202003, 202009)].groupby(["issuer_id"]).n.mean()
tr = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).join(iss[["ext", "name"]])
tr["ratio"] = tr.post / tr.pre
print(tr.groupby("ext").ratio.describe()[["count", "mean", "50%", "min", "max"]].round(2))
print(f"\n  issuers whose monthly flow FELL after March 2020: {(tr.ratio < 1).sum()} "
      f"of {len(tr)}  (nonbanks: {((tr.ratio < 1) & (tr.ext == 'nonbank')).sum()} "
      f"of {(tr.ext == 'nonbank').sum()})")
res["E_transfers"] = {"n_declining": int((tr.ratio < 1).sum()), "n": int(len(tr)),
                      "median_ratio": tr.groupby("ext").ratio.median().round(2).to_dict(),
                      "nonbank_declining": int(((tr.ratio < 1) & (tr.ext == "nonbank")).sum()),
                      "nonbank_n": int((tr.ext == "nonbank").sum())}

# ---------------------------------------------------------------- reweighting
line("F. REWEIGHTING nonbanks to the depository covariate distribution")
d = p[p.ext.isin(["nonbank", "depository"])].copy()
d["nb"] = (d.ext == "nonbank").astype(float)
X = sm.add_constant(d[CTRL].astype(float))
ps = np.asarray(sm.Logit(d.nb, X).fit(disp=0).predict(X))
w = np.where(d.nb == 1, (1 - ps) / np.clip(ps, 1e-6, 1 - 1e-6), 1.0)
d["w"] = np.clip(w, None, np.quantile(w, .99))
d["nb_post"] = d.nb * d.post
F = {}
for lab, wc in [("unweighted", None), ("IPW to depository covariates", "w")]:
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + state_month",
                 data=d, vcov={"CRV1": "issuer_id"}, weights=wc)
    print(f"   {lab:<34} {m.coef()['nb_post']:>8.4f} ({m.se()['nb_post']:.4f})")
    F[lab] = [round(float(m.coef()["nb_post"]), 4), round(float(m.se()["nb_post"]), 4)]
res["F_reweight"] = F

with open(os.path.join(OUT, "results_stress2.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("\nwrote results_stress2.json", flush=True)
