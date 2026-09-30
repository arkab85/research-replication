"""Wild cluster bootstrap on the size-gradient contrast, vectorised."""
import pandas as pd, numpy as np, os, json, warnings, time
from config import OUT
warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260922)

CTRL = ["coupon", "fico", "cltv", "age"]
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
p["ext"] = p.itype_ext
p = p.merge(iss[["S", "ebo19"]], left_on="issuer_id", right_index=True, how="inner")

def demean(M, groups, tol=1e-9, maxit=400):
    M = np.asarray(M, float).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    sizes = [c.max() + 1 for c in codes]
    for _ in range(maxit):
        delta = 0.0
        for c, k in zip(codes, sizes):
            n = np.bincount(c, minlength=k).astype(float)[:, None]
            s = np.column_stack([np.bincount(c, weights=M[:, j], minlength=k)
                                 for j in range(M.shape[1])])
            adj = (s / n)[c]
            M -= adj
            delta = max(delta, float(np.abs(adj).max()))
        if delta < tol: break
    return M

def cl_se(Xd, yd, cl, G, XtXi, j):
    b = XtXi @ (Xd.T @ yd)
    u = yd - Xd @ b
    Xu = Xd * u[:, None]
    agg = np.column_stack([np.bincount(cl, weights=Xu[:, q], minlength=G)
                           for q in range(Xd.shape[1])])
    n, k = Xd.shape
    c = (G / (G - 1)) * ((n - 1) / (n - k))
    V = c * (XtXi @ (agg.T @ agg) @ XtXi)
    return b[j], float(np.sqrt(V[j, j]))

res = {}
REPS = 9999
for thr, lab in [(.10, "active issuers"), (.00, "all issuers")]:
    t0w = time.time()
    d = p[p.ext.isin(["nonbank", "depository"]) & (p.ebo19 >= thr)].copy()
    d["S_z"] = (d.S - d.S.mean()) / d.S.std()
    d["nb"] = (d.ext == "nonbank").astype(float)
    d["S_post"] = d.S_z * d.post; d["nb_post"] = d.nb * d.post
    d["S_nb_post"] = d.S_z * d.nb * d.post
    rhs = ["S_post", "nb_post", "S_nb_post"] + CTRL
    while True:
        n0 = len(d)
        d = d[d.groupby("issuer_id").issuer_id.transform("size") > 1]
        d = d[d.groupby("state_month").state_month.transform("size") > 1]
        if len(d) == n0: break
    M = demean(np.column_stack([d.buyout.values] + [d[c].values for c in rhs]),
               [d.issuer_id.values, d.state_month.values])
    yd, Xd = M[:, 0], M[:, 1:]
    cl = pd.factorize(d.issuer_id)[0]; G = cl.max() + 1
    j = rhs.index("S_nb_post")
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    b, se = cl_se(Xd, yd, cl, G, XtXi, j); t0 = b / se
    keep = [q for q in range(len(rhs)) if q != j]
    X0 = Xd[:, keep]
    fit0 = X0 @ (np.linalg.pinv(X0.T @ X0) @ (X0.T @ yd))
    u0 = yd - fit0
    ts = np.empty(REPS)
    for r in range(REPS):
        ystar = fit0 + u0 * rng.choice([-1.0, 1.0], size=G)[cl]
        bb, ss = cl_se(Xd, ystar, cl, G, XtXi, j)
        ts[r] = bb / ss
    pb = float((np.abs(ts) >= abs(t0)).mean())
    print(f"  {lab:<16} pi3={b:+.4f} (se {se:.4f})  t={t0:+.3f}  "
          f"bootstrap p={pb:.4f}  G={G}  N={len(d):,}  [{time.time()-t0w:.0f}s]",
          flush=True)
    row = {"coef": round(float(b), 4), "se": round(float(se), 4),
           "t": round(float(t0), 3), "boot_p": round(pb, 4), "g": int(G), "n": int(len(d))}
    for k in ["S_post", "nb_post"]:
        bb, ss = cl_se(Xd, yd, cl, G, XtXi, rhs.index(k))
        row[k] = [round(float(bb), 4), round(float(ss), 4)]
    res[lab] = row

json.dump({"B_bootstrap": res}, open(os.path.join(OUT, "results_boot.json"), "w"), indent=1)
print("wrote results_boot.json")
