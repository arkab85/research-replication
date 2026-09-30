"""Directional certificates on the Tuebingen cause-effect pairs (ground truth A -> B).

For each pair: random split into training and evaluation halves (evaluation capped
at 1,000 and training at 3,000 observations), standardization by training moments,
cubic B-spline dictionaries with interior knots at training quantiles, unit Gaussian
kernels, and the joint Gaussian multiplier of Theorem 4 (499 draws, iid weights).
A certificate for A -> B requires L > 0; a certificate for B -> A requires U < 0.
"""
import numpy as np, pandas as pd, json, sys, time
from scipy.interpolate import BSpline

def make_dict(z_train, n_int=3, deg=3):
    lo, hi = np.min(z_train), np.max(z_train)
    inner = np.quantile(z_train, np.linspace(0, 1, n_int + 2)[1:-1])
    t = np.r_[[lo] * (deg + 1), inner, [hi] * (deg + 1)]
    nb = len(t) - deg - 1
    def p(z):
        zc = np.clip(z, lo, hi)
        B = BSpline.design_matrix(zc, t, deg).toarray()
        return B[:, 1:]                      # drop one column; intercept added separately
    return p

def ols(w, P):
    X = np.column_stack([np.ones(len(w)), P])
    Q = X.T @ X
    Qi = np.linalg.pinv(Q)
    th = Qi @ X.T @ w
    e = w - X @ th
    h = np.einsum("ij,jk,ik->i", X, Qi, X)
    sc = (X * (e / np.clip(1 - h, 1e-3, None))[:, None]) @ Qi
    return th, sc

def parts(e, z, P, s=1.0):
    n = len(e); H = np.eye(n) - 1.0 / n
    d = e[:, None] - e[None, :]
    K = np.exp(-d ** 2 / (2 * s * s))
    L = np.exp(-(z[:, None] - z[None, :]) ** 2 / (2 * s * s))
    Lc = H @ L @ H; Kc = H @ K @ H
    hs = float(np.sum(K * Lc)) / n ** 2
    Ac = H @ (Kc * Lc) @ H
    GJ = P.T @ (((1 / s ** 2) - d ** 2 / s ** 4) * K * Lc) @ P / n ** 2
    Dm = d * K / s ** 2
    B = -(H @ ((Dm - Dm.mean(0, keepdims=True)) * Lc) @ P) / n
    return hs, Ac, GJ, B

def certify(Xtr, Ytr, Xev, Yev, draws=499, seed=0):
    pX, pY = make_dict(Xtr), make_dict(Ytr)
    thf, scf = ols(Ytr, pX(Xtr)); thb, scb = ols(Xtr, pY(Ytr))
    S = np.column_stack([scf[:, 1:], scb[:, 1:]]); V = S.T @ S
    PXe, PYe = pX(Xev), pY(Yev)
    ef = Yev - PXe @ thf[1:]; eb = Xev - PYe @ thb[1:]
    res = [parts(ef, Xev, PXe), parts(eb, Yev, PYe)]
    n = len(Xev); k = PXe.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((draws, n))
    evals, evecs = np.linalg.eigh(V)
    vv = rng.standard_normal((draws, V.shape[0])) @ (evecs * np.sqrt(np.clip(evals, 0, None))).T
    sf, sj = [], []
    for r, (hs, Ac, GJ, B) in enumerate(res):
        v = vv[:, r * k:(r + 1) * k]
        qw = np.einsum("di,ij,dj->d", W, Ac, W) / n ** 2
        qv = np.einsum("di,ij,dj->d", v, GJ, v)
        qc = 2 * np.einsum("di,ij,dj->d", W, B, v) / n
        sf.append(np.sqrt(np.maximum(qw, 0))); sj.append(np.sqrt(np.maximum(qw + qv + qc, 0)))
    q0 = float(np.quantile(np.maximum(*sf), 0.95)); qJ = float(np.quantile(np.maximum(*sj), 0.95))
    hf, hb = res[0][0], res[1][0]
    out = dict(Hf=hf, Hb=hb, D=hb - hf, q0=q0, qJ=qJ)
    for q, tag in ((q0, "fixed"), (qJ, "joint")):
        nf, nb_ = np.sqrt(max(hf, 0)), np.sqrt(max(hb, 0))
        out[tag] = (max(nb_ - q, 0) ** 2 - min(1, nf + q) ** 2, min(1, nb_ + q) ** 2 - max(nf - q, 0) ** 2)
    return out

if __name__ == "__main__":
    pairs = pd.read_csv("Tuebingen_pairs.csv")
    done = set()
    try:
        for line in open("bench.jsonl"): done.add(json.loads(line)["pair"])
    except FileNotFoundError:
        pass
    f = open("bench.jsonl", "a")
    for _, row in pairs.iterrows():
        pid = row["SampleID"].strip()
        if pid in done: continue
        t0 = time.time(); print("start", pid, flush=True)
        A = np.array(row["A"].split(), float); B = np.array(row["B"].split(), float)
        ok = np.isfinite(A) & np.isfinite(B); A, B = A[ok], B[ok]
        N = len(A)
        rng = np.random.default_rng(int(pid.replace("pair", "")))
        idx = rng.permutation(N)
        n_ev = min(N // 2, 1000); n_tr = min(N - n_ev, 5000)
        ev, tr = idx[:n_ev], idx[n_ev:n_ev + n_tr]
        mA, sA, mB, sB = A[tr].mean(), A[tr].std(ddof=1), B[tr].mean(), B[tr].std(ddof=1)
        rec = dict(pair=pid, N=N, n_ev=int(n_ev), n_tr=int(n_tr))
        if sA == 0 or sB == 0 or n_ev < 50:
            rec["skip"] = "degenerate"
        else:
            Xtr, Ytr, Xev, Yev = (A[tr] - mA) / sA, (B[tr] - mB) / sB, (A[ev] - mA) / sA, (B[ev] - mB) / sB
            rec["rho"] = float(np.corrcoef(Xtr, Ytr)[0, 1])
            rec["distinct_A"] = int(len(np.unique(A))); rec["distinct_B"] = int(len(np.unique(B)))
            try:
                rec.update(certify(Xtr, Ytr, Xev, Yev, seed=1))
            except Exception as ex:
                rec["skip"] = f"error: {ex}"
        rec["sec"] = round(time.time() - t0, 1)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(pid, rec.get("D"), rec.get("joint"), rec.get("skip"), rec["sec"], flush=True)
