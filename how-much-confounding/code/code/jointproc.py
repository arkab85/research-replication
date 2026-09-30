"""Joint generated-residual inference for the directional contrast (Theorem 4).

Forward: W_f = Y regressed on p(X); backward: W_b = X regressed on p(Y), with
p(z) = (z, z^2) plus an intercept, fitted on a training sample.  Gaussian
kernels with bandwidth s on residuals and regressors (standardized units).
The Gaussian multiplier combines evaluation tensor features and the
estimated slopes as in eq. (jointboot); Gram-matrix formulas follow
Section OA.1 of the Supplement.
"""
import numpy as np


def dictionary(z):
    return np.column_stack([z, z ** 2])


def ols_fit(w, z, hc="HC3", hac_lags=None):
    P = np.column_stack([np.ones_like(z), dictionary(z)])
    Q = P.T @ P
    theta = np.linalg.solve(Q, P.T @ w)
    e = w - P @ theta
    Qi = np.linalg.inv(Q)
    if hc == "HC3":
        h = np.einsum("ij,jk,ik->i", P, Qi, P)
        e_adj = e / (1 - h)
    else:
        e_adj = e
    scores = (P * e_adj[:, None]) @ Qi          # rows: influence of each obs on theta (times m)
    return theta, scores


def slope_cov(scores_list, hac_lags=None):
    """Joint covariance of the stacked slope estimates (intercepts dropped)."""
    S = np.column_stack([s[:, 1:] for s in scores_list])
    V = S.T @ S
    if hac_lags:
        m = S.shape[0]
        for l in range(1, hac_lags + 1):
            G = S[l:].T @ S[:-l]
            V += (1 - l / (hac_lags + 1)) * (G + G.T)
    return V


def gram(x, s=1.0):
    d = x[:, None] - x[None, :]
    return np.exp(-d ** 2 / (2 * s ** 2)), d


def direction_parts(e, z, P, s=1.0):
    n = len(e)
    H = np.eye(n) - 1.0 / n
    K, d = gram(e, s)
    L, _ = gram(z, s)
    Lc = H @ L @ H
    Kc = H @ K @ H
    hsic = float(np.sum(K * Lc)) / n ** 2
    Ac = H @ (Kc * Lc) @ H
    GJ = P.T @ (((1 / s ** 2) - d ** 2 / s ** 4) * K * Lc) @ P / n ** 2
    D = d * K / s ** 2
    Dc = D - D.mean(0, keepdims=True)
    B = -(H @ (Dc * Lc) @ P) / n
    return dict(hsic=hsic, norm=np.sqrt(max(hsic, 0.0)), Ac=Ac, GJ=GJ, B=B)


def analyse(Xtr, Ytr, Xev, Yev, draws=1999, block=1, alpha=0.05, s=1.0, eta=(0.0,),
            hc="HC3", hac_lags=None, seed=0):
    """Return point components, radii (fixed and joint) and intervals for each eta."""
    th_f, sc_f = ols_fit(Ytr, Xtr, hc)
    th_b, sc_b = ols_fit(Xtr, Ytr, hc)
    V = slope_cov([sc_f, sc_b], hac_lags)
    e_f = Yev - dictionary(Xev) @ th_f[1:]
    e_b = Xev - dictionary(Yev) @ th_b[1:]
    parts = {"f": direction_parts(e_f, Xev, dictionary(Xev), s),
             "b": direction_parts(e_b, Yev, dictionary(Yev), s)}
    n = len(Xev)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    g = rng.standard_normal((draws, nb))
    Wm = np.repeat(g, block, axis=1)[:, :n]
    Lchol = np.linalg.cholesky(V + 1e-14 * np.eye(V.shape[0]))
    vv = rng.standard_normal((draws, V.shape[0])) @ Lchol.T
    stats_fixed, stats_joint = [], []
    for r, sl in (("f", slice(0, 2)), ("b", slice(2, 4))):
        p = parts[r]
        q_w = np.einsum("di,ij,dj->d", Wm, p["Ac"], Wm) / n ** 2
        v = vv[:, sl]
        q_v = np.einsum("di,ij,dj->d", v, p["GJ"], v)
        q_c = 2 * np.einsum("di,ij,dj->d", Wm, p["B"], v) / n
        stats_fixed.append(np.sqrt(np.maximum(q_w, 0)))
        stats_joint.append(np.sqrt(np.maximum(q_w + q_v + q_c, 0)))
    q0 = float(np.quantile(np.maximum(*stats_fixed), 1 - alpha))
    qJ = float(np.quantile(np.maximum(*stats_joint), 1 - alpha))
    out = dict(Hf=parts["f"]["hsic"], Hb=parts["b"]["hsic"], D=parts["b"]["hsic"] - parts["f"]["hsic"],
               q0=q0, qJ=qJ, n=n, m=len(Xtr), theta_f=th_f, theta_b=th_b)
    for q, tag in ((q0, "fixed"), (qJ, "joint")):
        for et in eta:
            a = 2 * et / s
            lb = max(parts["b"]["norm"] - q - a, 0) ** 2
            ub = min(1.0, parts["b"]["norm"] + q + a) ** 2
            lf = max(parts["f"]["norm"] - q - a, 0) ** 2
            uf = min(1.0, parts["f"]["norm"] + q + a) ** 2
            out[(tag, et)] = (lb - uf, ub - lf)
    return out


def analyse_multi(pairs_tr, pairs_ev, draws=1999, block=1, alpha=0.05, s=1.0, hc="HC3", seed=0, hac_lags=None):
    """Simultaneous event over several (X, Y) pairs: 2 operators per pair, shared multipliers."""
    fits, parts, scores = [], [], []
    for (Xtr, Ytr), (Xev, Yev) in zip(pairs_tr, pairs_ev):
        th_f, sc_f = ols_fit(Ytr, Xtr, hc); th_b, sc_b = ols_fit(Xtr, Ytr, hc)
        scores += [sc_f, sc_b]
        e_f = Yev - dictionary(Xev) @ th_f[1:]; e_b = Xev - dictionary(Yev) @ th_b[1:]
        parts += [direction_parts(e_f, Xev, dictionary(Xev), s), direction_parts(e_b, Yev, dictionary(Yev), s)]
    V = slope_cov(scores, hac_lags)
    n = len(pairs_ev[0][0])
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); g = rng.standard_normal((draws, nb))
    Wm = np.repeat(g, block, axis=1)[:, :n]
    vv = rng.standard_normal((draws, V.shape[0])) @ np.linalg.cholesky(V + 1e-14 * np.eye(V.shape[0])).T
    sf, sj = [], []
    for k, p in enumerate(parts):
        v = vv[:, 2 * k:2 * k + 2]
        qw = np.einsum("di,ij,dj->d", Wm, p["Ac"], Wm) / n ** 2
        qv = np.einsum("di,ij,dj->d", v, p["GJ"], v)
        qc = 2 * np.einsum("di,ij,dj->d", Wm, p["B"], v) / n
        sf.append(np.sqrt(np.maximum(qw, 0))); sj.append(np.sqrt(np.maximum(qw + qv + qc, 0)))
    q0 = float(np.quantile(np.max(sf, 0), 1 - alpha)); qJ = float(np.quantile(np.max(sj, 0), 1 - alpha))
    res = []
    for k in range(len(pairs_ev)):
        pf, pb = parts[2 * k], parts[2 * k + 1]
        r = dict(Hf=pf["hsic"], Hb=pb["hsic"], D=pb["hsic"] - pf["hsic"])
        for q, tag in ((q0, "fixed"), (qJ, "joint")):
            r[tag] = (max(pb["norm"] - q, 0) ** 2 - min(1, pf["norm"] + q) ** 2,
                      min(1, pb["norm"] + q) ** 2 - max(pf["norm"] - q, 0) ** 2)
        res.append(r)
    return res, q0, qJ
