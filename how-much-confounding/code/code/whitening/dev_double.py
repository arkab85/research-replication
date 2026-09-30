"""DEVELOPMENT study, part 2: iterated (double) bootstrap calibration.
Degrees-of-freedom corrected covariance U'U/(n-K) in both worlds; residuals are
rescaled by sqrt(n/(n-K)) before resampling so that the bootstrap-world truth
equals the point estimate.  Outer B1=999, inner B2=199 draws.
Seeds: SeedSequence(7002xx) (development family).
"""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from wcore import *
from designs import simulate, A_ROT
from dev_whitening import DEV, logchol, jac_logchol

LEVEL = .975; B1 = 999; B2 = 199; CH = 25


def stats_from_resid(U, K, ell, boot):
    """l, V_l (Cholesky), g, V_g (log-Cholesky) from residual arrays (...,n,2)."""
    n = U.shape[-2]
    Uc = U - U.mean(axis=-2, keepdims=True)
    S = np.einsum('...ti,...tj->...ij', Uc, Uc) / (n - K)
    l = chol2(S)
    X = xi(U)
    Vs = (lrv_bartlett(X, 1) if ell == 1 else (lrv_blocks(X, ell) if boot else lrv_bartlett(X, ell)))
    J = jac_chol2(l)
    Vl = J @ Vs @ np.swapaxes(J, -1, -2)
    D = jac_logchol(l) @ J
    Vg = D @ Vs @ np.swapaxes(D, -1, -2)
    return l, Vl, logchol(l), Vg


def Lmat(l):
    M = np.zeros(l.shape[:-1] + (2, 2)); M[..., 0, 0] = l[..., 0]; M[..., 1, 0] = l[..., 1]; M[..., 1, 1] = l[..., 2]
    return M


def one(task):
    design, n, ell, seed = task
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(design, n, rng)
    l0 = np.array([L0[0, 0], L0[1, 0], L0[1, 1]]); p = 1; K = 1 + 2 * p
    Bh, U = var_fit(y, p)
    l, Vl, g, Vg = stats_from_resid(U, K, ell, False)
    Ur = U * np.sqrt(n / (n - K))
    draw = (lambda m, r: iid_indices(n, m, r)) if ell == 1 else (lambda m, r: mbb_indices_batch(n, m, ell, r))
    Ys = regen_batch(Bh, y[:p], Ur[draw(B1, rng)])
    Bs, Us = refit_batch_coef(Ys, p)
    ls, Vls, gs, Vgs = stats_from_resid(Us, K, ell, True)
    T2s = n * quad(ls - l, Vls); T3s = n * quad(gs - g, Vgs)
    Fs = np.linalg.norm(np.linalg.solve(Lmat(l), Lmat(ls) - Lmat(l)), axis=(1, 2))
    u2 = np.empty(B1); u3 = np.empty(B1); u0 = np.empty(B1)
    Urs = Us * np.sqrt(n / (n - K))
    for c0 in range(0, B1, CH):
        bb = np.arange(c0, min(c0 + CH, B1)); m = len(bb)
        idx = draw(m * B2, rng).reshape(m, B2, n)
        Uin = Urs[bb][np.arange(m)[:, None, None], idx]            # (m,B2,n,2)
        Bin = np.repeat(Bs[bb], B2, axis=0)
        y0 = np.repeat(Ys[bb, :p], B2, axis=0)
        Yin = regen_batch_coef(Bin, y0, Uin.reshape(m * B2, n, 2))
        _, Uin2 = refit_batch_coef(Yin, p)
        li, Vli, gi, Vgi = stats_from_resid(Uin2.reshape(m, B2, n, 2), K, ell, True)
        T2i = n * quad(li - ls[bb][:, None], Vli); T3i = n * quad(gi - gs[bb][:, None], Vgi)
        Fi = np.linalg.norm(np.linalg.solve(Lmat(ls[bb])[:, None], Lmat(li) - Lmat(ls[bb])[:, None]), axis=(2, 3))
        u2[bb] = np.mean(T2i <= T2s[bb][:, None], axis=1)
        u3[bb] = np.mean(T3i <= T3s[bb][:, None], axis=1)
        u0[bb] = np.mean(Fi <= Fs[bb][:, None], axis=1)
    T2 = n * quad(l - l0, Vl); T3 = n * quad(g - logchol(l0), Vg)
    F0 = np.linalg.norm(np.linalg.solve(Lmat(l), L0 - Lmat(l)))
    res = dict(seed=int(seed))
    for key, Ts, T, u in (('F', Fs, F0, u0), ('W2', T2s, T2, u2), ('W3', T3s, T3, u3)):
        c1 = order_quantile(Ts, LEVEL)
        beta = order_quantile(u, LEVEL)
        c2 = order_quantile(Ts, min(beta, B1 / (B1 + 1)))
        res[key + '_single'] = bool(T <= c1); res[key + '_double'] = bool(T <= c2)
        res[key + '_beta'] = float(beta); res[key + '_c1'] = float(c1); res[key + '_c2'] = float(c2)
    return res


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    names = sys.argv[2:] if len(sys.argv) > 2 else list(DEV)
    out = {}
    t = time.time()
    for name in names:
        d = DEV[name]; di = list(DEV).index(name)
        ell = 1 if d['vol'][0] in ('iid2', 'none') else 10
        seeds = np.random.SeedSequence(700200 + 10 * di).generate_state(reps)
        with ProcessPoolExecutor(2) as ex:
            rows = list(ex.map(one, [(d, 300, ell, int(s)) for s in seeds], chunksize=4))
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'seed'}
        out[name] = summ
        print(name, json.dumps({k: round(v, 3) for k, v in summ.items()}), round(time.time() - t), flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(__file__), 'dev_double_summary.json'), 'w'), indent=2)
