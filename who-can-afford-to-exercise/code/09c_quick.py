"""Stress tests, fast version: absorb the fixed effects once (Frisch-Waugh) and run the
wild cluster bootstrap on the residualized design. Also the flow test, the servicing-
transfer test and the reweighting check."""
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

# ---------------------------------------------------------------- FWL helpers
def demean(M, groups, tol=1e-8, maxit=200):
    """Alternating projections: sweep out several sets of group means."""
    M = np.asarray(M, dtype=float).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    sizes = [c.max() + 1 for c in codes]
    for _ in range(maxit):
        delta = 0.0
        for c, k in zip(codes, sizes):
            s = np.zeros((k, M.shape[1])); n = np.bincount(c, minlength=k)[:, None]
            np.add.at(s, c, M)
            adj = (s / n)[c]
            M -= adj
            delta = max(delta, np.abs(adj).max())
        if delta < tol: break
    return M

def cluster_t(Xd, yd, cl, j):
    """OLS on residualized data with one-way cluster-robust t on column j."""
    XtX = Xd.T @ Xd
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (Xd.T @ yd)
    u = yd - Xd @ b
    G = cl.max() + 1
    meat = np.zeros_like(XtX)
    Xu = Xd * u[:, None]
    agg = np.zeros((G, Xd.shape[1]))
    np.add.at(agg, cl, Xu)
    meat = agg.T @ agg
    n, k = Xd.shape
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * XtXi @ meat @ XtXi
    return b[j], np.sqrt(V[j, j])

def prep(d, rhs):
    d = d.copy()
    while True:
        n0 = len(d)
        d = d[d.groupby("issuer_id").issuer_id.transform("size") > 1]
        d = d[d.groupby("state_month").state_month.transform("size") > 1]
        if len(d) == n0: break
    M = demean(np.column_stack([d.buyout.values] + [d[c].values for c in rhs]),
               [d.issuer_id.values, d.state_month.values])
    return d, M[:, 0], M[:, 1:], pd.factorize(d.issuer_id)[0]

def wcb(d, rhs, jname, reps=9999):
    d, yd, Xd, cl = prep(d, rhs)
    j = rhs.index(jname)
    b, se = cluster_t(Xd, yd, cl, j); t0 = b / se
    keep = [i for i in range(len(rhs)) if i != j]
    X0 = Xd[:, keep]
    b0 = np.linalg.pinv(X0.T @ X0) @ (X0.T @ yd)
    fit0 = X0 @ b0; u0 = yd - fit0
    G = cl.max() + 1
    ts = np.empty(reps)
    for r in range(reps):
        ystar = fit0 + u0 * rng.choice([-1.0, 1.0], size=G)[cl]
        bb, ss = cluster_t(Xd, ystar, cl, j)
        ts[r] = bb / ss
    return float(b), float(se), float(t0), float((np.abs(ts) >= abs(t0)).mean()), int(G), len(d)

# ---------------------------------------------------------------- A
line("A. THE SIMPLEST STATEMENT OF THE RESULT")
a = iss[(iss.ebo19 >= .10) & iss.ext.isin(["nonbank", "depository"])]
for t in ["depository", "nonbank"]:
    d = a[a.ext == t]
    print(f"  {t:<11}: {(d.d_ebo > 0).sum():>2} of {len(d):>2} active issuers raised their "
          f"exercise rate;  median change {d.d_ebo.median()*100:+.1f} pp")
from scipy.stats import fisher_exact
tab = [[int((a[a.ext == 'depository'].d_ebo > 0).sum()),
        int((a[a.ext == 'depository'].d_ebo <= 0).sum())],
       [int((a[a.ext == 'nonbank'].d_ebo > 0).sum()),
        int((a[a.ext == 'nonbank'].d_ebo <= 0).sum())]]
orr, pf_ = fisher_exact(tab)
print(f"  Fisher exact test of the 2x2: p = {pf_:.4f}")
res["A_counts"] = {t: {"up": int((a[a.ext == t].d_ebo > 0).sum()),
                       "n": int((a.ext == t).sum()),
                       "median_change_pp": round(float(a[a.ext == t].d_ebo.median()*100), 1)}
                   for t in ["depository", "nonbank"]} | {"fisher_p": round(float(pf_), 4)}


# ---------------------------------------------------------------- C
line("C. LEAVE-ONE-ISSUER-OUT on the contrast (active issuers)")
d0 = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= .10)].copy()
vals = []
for drop in [None] + sorted(d0.issuer_id.unique()):
    d = (d0 if drop is None else d0[d0.issuer_id != drop]).copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"] = d.S_z * d.post; d["nb_post"] = d.nb * d.post
    d["S_nb_post"] = d.S_z * d.nb * d.post
    rhs = ["S_post", "nb_post", "S_nb_post"] + CTRL
    dd, yd, Xd, cl = prep(d, rhs)
    b, se = cluster_t(Xd, yd, cl, rhs.index("S_nb_post"))
    from scipy.stats import t as tdist
    pv = 2 * (1 - tdist.cdf(abs(b / se), cl.max()))
    vals.append((drop, b, se, pv))
lo = pd.DataFrame(vals, columns=["dropped", "coef", "se", "p"])
lo["name"] = lo.dropped.map(iss["name"])
print(f"  full sample: {lo.coef[0]:+.4f} ({lo.se[0]:.4f})")
print(f"  leave-one-out range [{lo.coef[1:].min():+.4f}, {lo.coef[1:].max():+.4f}]; "
      f"max p={lo.p[1:].max():.4f}; all negative={bool((lo.coef[1:] < 0).all())}")
print(lo.iloc[1:].reindex(lo.iloc[1:].coef.abs().sort_values().index).head(4)
      [["name", "coef", "se", "p"]].round(4).to_string(index=False))
lo.to_csv(os.path.join(OUT, "leave_one_out.csv"), index=False)
res["C_loo"] = {"full": round(float(lo.coef[0]), 4), "min": round(float(lo.coef[1:].min()), 4),
                "max": round(float(lo.coef[1:].max()), 4),
                "max_p": round(float(lo.p[1:].max()), 4),
                "all_neg": bool((lo.coef[1:] < 0).all())}

# ---------------------------------------------------------------- D
line("D. HIGH-FREQUENCY CAPACITY TEST: the issuer's own monthly option flow")
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
        print(f"   {label:<36} {k:<13} {c:>8.4f} ({se:.4f}) {st:<3} "
              f"N={len(d):>8,} G={d.issuer_id.nunique()}", flush=True)
        out[k] = [round(float(c), 4), round(float(se), 4), float(pv)]
    return out | {"n": int(len(d)), "g": int(d.issuer_id.nunique())}

print("\n  [ exercise rate vs. the issuer's own monthly flow of newly vested options ]")
for t in ["nonbank", "depository"]:
    d = p[p.ext == t]
    D[f"{t}_pre"] = flowreg(d[d.post == 0], f"{t}, 2019 only")
    D[f"{t}_post"] = flowreg(d[d.post == 1], f"{t}, Mar-Sep 2020")
dd = p[p.ext.isin(["nonbank", "depository"])].copy()
dd["nb"] = (dd.ext == "nonbank").astype(float)
dd["logN_nb"] = dd.logN * dd.nb; dd["logN_post"] = dd.logN * dd.post
dd["logN_nb_post"] = dd.logN * dd.nb * dd.post; dd["nb_post"] = dd.nb * dd.post
print()
D["pooled"] = flowreg(dd, "pooled, all interactions",
                      extra=["logN_nb", "logN_post", "logN_nb_post", "nb_post"])
res["D_flow"] = D

# ---------------------------------------------------------------- E
line("E. SERVICING TRANSFERS: per-issuer monthly decision counts")
cnt = p.groupby(["issuer_id", "ym"]).size().rename("n").reset_index()
pre = cnt[cnt.ym.between(201901, 201912)].groupby("issuer_id").n.mean()
post = cnt[cnt.ym.between(202003, 202009)].groupby("issuer_id").n.mean()
tr = pd.concat([pre.rename("pre"), post.rename("post")], axis=1).join(iss[["ext", "name"]])
tr["ratio"] = tr.post / tr.pre
print(tr.groupby("ext").ratio.describe()[["count", "mean", "50%", "min", "max"]].round(2))
nbd = int(((tr.ratio < 1) & (tr.ext == "nonbank")).sum()); nbn = int((tr.ext == "nonbank").sum())
print(f"\n  nonbank issuers whose monthly flow FELL after March 2020: {nbd} of {nbn}")
res["E_transfers"] = {"nonbank_declining": nbd, "nonbank_n": nbn,
                      "median_ratio": tr.groupby("ext").ratio.median().round(2).to_dict()}

# ---------------------------------------------------------------- F
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

json.dump(res, open(os.path.join(OUT, "results_quick.json"), "w"), indent=1, default=str)
print("\nwrote results_quick.json", flush=True)

