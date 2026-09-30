"""DEVELOPMENT study (not a confirmatory result): compare candidate confidence
regions for the Cholesky whitening factor.  Seeds come from SeedSequence(7001xx);
the confirmatory study uses a disjoint seed family and additional designs.

Candidates (all at the Bonferroni level .975 used for the whitening event):
 W0  percentile root ||L^-1(L*-L)||_F (earlier procedure), B draws
 W1  asymptotic Wald ellipsoid in vech(L), chi2_3 critical value
 W2  studentised bootstrap-t Wald ellipsoid in vech(L)
 W3  studentised bootstrap-t Wald in log-Cholesky coordinates (log L11, L21, log L22)
 W4  studentised bootstrap-t Wald in vech(Sigma)
 W2n as W2 but with Bartlett studentisation also in the bootstrap world (MBB only)
"""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from wcore import (var_fit, regen_batch, refit_batch, iid_indices, mbb_indices_batch, cov2,
                   chol2, jac_chol2, xi, lrv_bartlett, lrv_blocks, quad, order_quantile, vech2)
from designs import simulate, A_ROT

LEVEL = .975
B = 999


def logchol(l):
    return np.stack([np.log(l[..., 0]), l[..., 1], np.log(l[..., 2])], axis=-1)


def jac_logchol(l):
    D = np.zeros(l.shape[:-1] + (3, 3))
    D[..., 0, 0] = 1 / l[..., 0]; D[..., 1, 1] = 1; D[..., 2, 2] = 1 / l[..., 2]
    return D


def one(task):
    design, n, ell, seed = task
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(design, n, rng)
    l0 = np.array([L0[0, 0], L0[1, 0], L0[1, 1]]); s0 = vech2(L0 @ L0.T)
    Bh, U = var_fit(y, 1)
    S = cov2(U); l = chol2(S); s = vech2(S)
    X = xi(U)
    Vs = lrv_bartlett(X, ell)
    J = jac_chol2(l); Vl = J @ Vs @ J.T
    Dg = jac_logchol(l) @ J; Vg = Dg @ Vs @ Dg.T
    # bootstrap
    idx = iid_indices(n, B, rng) if ell == 1 else mbb_indices_batch(n, B, ell, rng)
    Ys = regen_batch(Bh, y[:1], U[idx])
    Us = refit_batch(Ys, 1)
    Ss = cov2(Us); ls = chol2(Ss); ss = vech2(Ss)
    Xs = xi(Us)
    Vss = lrv_bartlett(Xs, 1) if ell == 1 else lrv_blocks(Xs, ell)
    Vssn = lrv_bartlett(Xs, ell)
    Js = jac_chol2(ls)
    Vls = Js @ Vss @ np.swapaxes(Js, -1, -2)
    Vlsn = Js @ Vssn @ np.swapaxes(Js, -1, -2)
    Dgs = jac_logchol(ls) @ Js; Vgs = Dgs @ Vss @ np.swapaxes(Dgs, -1, -2)
    Lm = np.array([[l[0], 0], [l[1], l[2]]])
    Lsm = np.zeros((B, 2, 2)); Lsm[:, 0, 0] = ls[:, 0]; Lsm[:, 1, 0] = ls[:, 1]; Lsm[:, 1, 1] = ls[:, 2]
    froot = np.linalg.norm(np.linalg.solve(Lm, Lsm - Lm), axis=(1, 2))
    r0 = order_quantile(froot, LEVEL)
    T2s = n * quad(ls - l, Vls); c2 = order_quantile(T2s, LEVEL)
    T2ns = n * quad(ls - l, Vlsn); c2n = order_quantile(T2ns, LEVEL)
    T3s = n * quad(logchol(ls) - logchol(l), Vgs); c3 = order_quantile(T3s, LEVEL)
    T4s = n * quad(ss - s, Vss); c4 = order_quantile(T4s, LEVEL)
    t0 = np.linalg.norm(np.linalg.solve(Lm, L0 - Lm))
    T = n * quad(l - l0, Vl); T3 = n * quad(logchol(l) - logchol(l0), Vg); T4 = n * quad(s - s0, Vs)
    c1 = chi2.ppf(LEVEL, 3)
    # exact linear projection half-width of entries of L R(theta), averaged over theta and k
    th = np.linspace(0, np.pi / 2, 31)
    hw = {'W0': [], 'W1': [], 'W2': []}
    for t in th:
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        for k in range(2):
            a1 = np.array([R[0, k], 0, 0]); a2 = np.array([0, R[0, k], R[1, k]])
            # W0 ball: (L H R)_{1k} = L11 H11 R1k ; (L H R)_{2k} coefficients (L21 R1k, L22 R1k, L22 R2k)
            b1 = r0 * abs(l[0] * R[0, k]); b2 = r0 * np.linalg.norm([l[1] * R[0, k], l[2] * R[0, k], l[2] * R[1, k]])
            hw['W0'].append(b1 + b2)
            for key, c in (('W1', c1), ('W2', c2)):
                hw[key].append(np.sqrt(c * (a1 @ Vl @ a1) / n) + np.sqrt(c * (a2 @ Vl @ a2) / n))
    return dict(seed=int(seed), n=n, ell=ell, W0=bool(t0 <= r0), W1=bool(T <= c1), W2=bool(T <= c2),
                W2n=bool(T <= c2n), W3=bool(T3 <= c3), W4=bool(T4 <= c4),
                c2=float(c2), c3=float(c3), c4=float(c4), r0=float(r0),
                hw0=float(np.mean(hw['W0'])), hw1=float(np.mean(hw['W1'])), hw2=float(np.mean(hw['W2'])))


DEV = {
    'iid_d0_exp': dict(A=A_ROT, eta='exp', vol=('iid2', 0.0)),
    'iid_d4_exp': dict(A=A_ROT, eta='exp', vol=('iid2', 0.4)),
    'markov_d4_exp': dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, 0.95)),
}

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = {}
    t = time.time()
    for di, (name, d) in enumerate(DEV.items()):
        for n in (300,):
            ell = 1 if d['vol'][0] in ('iid2', 'none') else 10
            seeds = np.random.SeedSequence(700100 + 10 * di).generate_state(reps)
            with ProcessPoolExecutor(2) as ex:
                rows = list(ex.map(one, [(d, n, ell, int(s)) for s in seeds], chunksize=8))
            summ = {k: float(np.mean([r[k] for r in rows])) for k in ('W0', 'W1', 'W2', 'W2n', 'W3', 'W4', 'hw0', 'hw1', 'hw2', 'c2', 'c3', 'c4', 'r0')}
            out[f'{name}_n{n}'] = summ
            print(name, n, json.dumps({k: round(v, 3) for k, v in summ.items()}), round(time.time() - t), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'dev_whitening_summary.json'), 'w'), indent=2)
