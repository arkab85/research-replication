"""General-d tools for the four-variable monetary-policy application.

Rotations are Q in SO(d); candidate shocks are e = z Q (rows of z are whitened innovations), i.e.
e_t = Q' z_t.  Kernel operators are pairwise, unit bandwidth unless stated.
"""
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from acore import op_inner_cond, quantile_bins


def whiten(u):
    L = np.linalg.cholesky(np.cov(u.T))
    return np.linalg.solve(L, u.T).T, L


def skew(theta, d):
    A = np.zeros((d, d)); iu = np.triu_indices(d, 1)
    A[iu] = theta; return A - A.T


def rotation(theta, d):
    return expm(skew(theta, d))


def pair_sq_norms(E, bins=None, k=1, bw=1.0):
    """Squared operator norms for all pairs i<j of the columns of E (optionally regime-conditioned)."""
    n, d = E.shape
    b = np.zeros(n, int) if bins is None else bins
    kk = 1 if bins is None else k
    return np.array([op_inner_cond(E[:, [i, j]], b, E[:, [i, j]], b, kk, bw) for i, j in itertools.combinations(range(d), 2)])


def pair_devs(Eb, bb, E, b, k, orig, bw=1.0):
    """||C*_ij - C_ij|| for all pairs, given original squared norms."""
    d = E.shape[1]; out = []
    for m, (i, j) in enumerate(itertools.combinations(range(d), 2)):
        v = op_inner_cond(Eb[:, [i, j]], bb, Eb[:, [i, j]], bb, k, bw) + orig[m] - 2 * op_inner_cond(Eb[:, [i, j]], bb, E[:, [i, j]], b, k, bw)
        out.append(np.sqrt(max(v, 0.)))
    return np.array(out)


def ordering_rotations(L, d):
    """Q_pi for every recursive ordering pi: e = L_pi^{-1} P u = L_pi^{-1} P L z, so Q_pi = (L_pi^{-1} P L)'.
    Columns of e are returned in the original variable order of the ordering's shocks (shock k is the
    one attached to variable pi[k])."""
    S = L @ L.T; out = {}
    for pi in itertools.permutations(range(d)):
        P = np.eye(d)[list(pi)]
        Lp = np.linalg.cholesky(P @ S @ P.T)
        M = np.linalg.solve(Lp, P @ L)
        out[pi] = M.T
    return out


def min_dependence_rotation(z, starts=20, seed=0, objective='sum'):
    """Rotation minimising the sum (or max) of pairwise squared operator norms (an ICA-type estimate)."""
    n, d = z.shape; p = d * (d - 1) // 2
    rng = np.random.default_rng(seed)
    f = (lambda t: pair_sq_norms(z @ rotation(t, d)).sum()) if objective == 'sum' else (lambda t: pair_sq_norms(z @ rotation(t, d)).max())
    best = None
    for s in range(starts):
        t0 = rng.uniform(-np.pi, np.pi, p) if s else np.zeros(p)
        r = minimize(f, t0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return rotation(best.x, d), best.fun


def joint_diag(z, bins, k, starts=20, seed=0):
    """Rotation that jointly diagonalises the regime covariances of z (heteroskedasticity identification)."""
    n, d = z.shape; p = d * (d - 1) // 2
    pb = np.bincount(bins, minlength=k) / n
    Ms = [np.cov(z[bins == j].T, bias=True) for j in range(k)]
    iu = np.triu_indices(d, 1)

    def f(t):
        R = rotation(t, d)
        return sum(w * ((R.T @ M @ R)[iu] ** 2).sum() for w, M in zip(pb, Ms))
    rng = np.random.default_rng(seed); best = None
    for s in range(starts):
        t0 = rng.uniform(-np.pi, np.pi, p) if s else np.zeros(p)
        r = minimize(f, t0, method='BFGS')
        if best is None or r.fun < best.fun:
            best = r
    R = rotation(best.x, d)
    D = np.array([np.diag(R.T @ M @ R) for M in Ms])
    return R, D, best.fun


def match_columns(R, R0):
    """Signed permutation of the columns of R that best matches R0 (maximises |trace|);
    returns the aligned R and the column-wise angles in degrees."""
    d = R.shape[1]; best = None
    for pi in itertools.permutations(range(d)):
        Rp = R[:, list(pi)]
        s = np.sign(np.sum(Rp * R0, 0)); s[s == 0] = 1
        Rp = Rp * s
        sc = np.sum(Rp * R0)
        if best is None or sc > best[0]:
            best = (sc, Rp)
    Rp = best[1]
    ang = np.degrees(np.arccos(np.clip(np.sum(Rp * R0, 0), -1, 1)))
    return Rp, ang


def coskew_moments(E):
    E = (E - E.mean(0)) / E.std(0); d = E.shape[1]
    return np.array([np.mean(E[:, i] ** 2 * E[:, j]) for i in range(d) for j in range(d) if i != j])


def isotropy_moments(z, bins, k):
    n, d = z.shape
    iu = np.triu_indices(d)
    G = np.einsum('ti,tj->tij', z, z) - (np.einsum('ti,ti->t', z, z) / d)[:, None, None] * np.eye(d)
    g = G[:, iu[0], iu[1]][:, :-1]                         # drop last diagonal (trace zero)
    return np.column_stack([(bins == j)[:, None] * g for j in range(k)])[:, :-(g.shape[1])]
