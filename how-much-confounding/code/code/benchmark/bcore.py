"""Shared tools for the exposure benchmark and the FOMC application.

Notation follows the paper: u_t are reduced-form innovations, z_t = L^{-1} u_t are
Cholesky-whitened innovations, and rotation angle theta indexes candidate shocks
e(theta) = R(theta)' z_t.  Gaussian kernels have unit bandwidth.
"""
import os, sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'whitening'))
import numpy as np
import pandas as pd
from scipy.stats import chi2
from fastop import op_inner_fast

CURV = 8 + 4 * np.sqrt(3)          # curvature constant M of the certified-cell proposition


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def whiten(u):
    L = np.linalg.cholesky(np.cov(u.T))
    return np.linalg.solve(L, u.T).T, L


def boot_index(n, block, rng):
    """iid indices (block = 1) or moving-block indices."""
    if block == 1:
        return rng.integers(0, n, n)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, nb)
    return (st[:, None] + np.arange(block)).ravel()[:n]


# ------------------------------------------------------------------ operator certificate
def sq_norms(z, grid):
    return np.array([op_inner_fast(z @ rot(t), z @ rot(t)) for t in grid])


def operator_radius(u, grid, B, block, rng, level=.975, refit=None):
    """Bootstrap .975 quantile of max_theta ||C*(theta) - C_hat(theta)||.
    `refit(ub)` maps resampled raw innovations to whitened ones (default: demean and whiten)."""
    refit = refit or (lambda ub: whiten(ub - ub.mean(0))[0])
    z = refit(u)
    orig = sq_norms(z, grid)
    D = np.empty((B, len(grid)))
    n = len(u)
    for b in range(B):
        zb = refit(u[boot_index(n, block, rng)])
        for j, (t, v) in enumerate(zip(grid, orig)):
            Eb = zb @ rot(t)
            D[b, j] = np.sqrt(max(0., op_inner_fast(Eb, Eb) + v - 2 * op_inner_fast(Eb, z @ rot(t))))
    mx = np.sort(D.max(1))
    q = mx[min(int(np.ceil(level * (B + 1))), B) - 1]
    return float(q), orig, D


def cell_lower(z, grid, orig, q):
    """Certified lower bound on inf_{theta in cell} ||C(theta)||, one per grid cell."""
    low, seg = [], []
    for j in range(len(grid) - 1):
        a, b = orig[j:j + 2]
        c = op_inner_fast(z @ rot(grid[j]), z @ rot(grid[j + 1]))
        v = max(a + b - 2 * c, 0)
        t = np.clip((a - c) / v, 0, 1) if v > 1e-16 else 0
        m = np.sqrt(max(a + 2 * t * (c - a) + t * t * v, 0))
        seg.append(m)
        low.append(max(0., m - q - CURV * (grid[j + 1] - grid[j]) ** 2 / 8))
    return np.array(low), np.array(seg)


# ------------------------------------------------------------------ exposure benchmark
def bins_of(W, k):
    return pd.qcut(pd.Series(W).rank(method='first'), k, labels=False).values


def rho_proxy(u, W, k):
    """Binned proxy benchmark: rho_W = {1 - (E tau)^2 / E tau^2}^{1/2}, where tau^2 is the
    within-bin mean of z'z / dim.  Returns (rho, normalised tau^2 by bin)."""
    S = np.cov(u.T)
    m2 = np.einsum('ti,ij,tj->t', u, np.linalg.inv(S), u) / u.shape[1]
    b = bins_of(W, k)
    p = np.bincount(b, minlength=k) / len(b)
    t2 = np.array([m2[b == j].mean() for j in range(k)])
    R = (p @ np.sqrt(t2)) ** 2 / (p @ t2)
    return float(np.sqrt(max(0., 1 - R))), t2 / (p @ t2)


def bin_means(u, W, k):
    """Within-bin means of ||z||^2/d minus the last bin (implied), used for the equal-means test."""
    S = np.cov(u.T)
    m2 = np.einsum('ti,ij,tj->t', u, np.linalg.inv(S), u) / u.shape[1]
    b = bins_of(W, k)
    return np.array([m2[b == j].mean() for j in range(k - 1)])


def rho_proxy_boot(u, W, k, B, block, rng, level=.95):
    """Point estimate, basic-bootstrap lower confidence bound 2*rho_hat - q_level(rho*), and a
    Wald pretest of equal bin means of ||z||^2 (null: rho_{W_k} = 0) with bootstrap covariance.
    The lower bound is set to zero unless the pretest rejects at the 5% level, because the
    bootstrap is inconsistent at the boundary.  Sigma_hat is refitted in every draw."""
    r, t2 = rho_proxy(u, W, k)
    m = bin_means(u, W, k)
    n = len(u); rb = np.empty(B); mb = np.empty((B, k - 1))
    for i in range(B):
        idx = boot_index(n, block, rng)
        ub = u[idx] - u[idx].mean(0); rb[i] = rho_proxy(ub, W[idx], k)[0]
        mb[i] = bin_means(ub, W[idx], k)
    V = np.cov(mb.T).reshape(k - 1, k - 1)
    dm = m - 1.0
    wald = float(dm @ np.linalg.solve(V, dm)); p = float(chi2.sf(wald, k - 1))
    basic = float(max(0., 2 * r - np.quantile(rb, level)))
    return dict(k=k, rho=r, boot_mean=float(rb.mean()), lower=basic if p < .05 else 0., lower_unguarded=basic,
                equal_means_wald=wald, equal_means_p=p, tau2=t2.tolist())


def shock_floors(u, W, k, angle_deg):
    """Per-shock floors {Var tau_i(W_k)}^{1/2} at a given rotation (d = 2), and the pair floor."""
    z, _ = whiten(u)
    e = z @ rot(np.deg2rad(angle_deg))
    b = bins_of(W, k); p = np.bincount(b, minlength=k) / len(b)
    out = []
    for i in range(2):
        t2 = np.array([(e[b == j, i] ** 2).mean() for j in range(k)])
        out.append(float(np.sqrt(max(0., 1 - (p @ np.sqrt(t2)) ** 2 / (p @ t2)))))
    return dict(shock_floors=out, pair_floor=float(np.sqrt(out[0] * out[1])))


def isotropy_moments(u, W, k):
    z, _ = whiten(u)
    b = bins_of(W, k)
    g = np.column_stack([(b == j)[:, None] * np.c_[z[:, 0] ** 2 - z[:, 1] ** 2, 2 * z[:, 0] * z[:, 1]]
                         for j in range(k)])
    return g[:, :-2]          # the last bin is implied by whitening


def isotropy_test(u, W, k, B, block, rng):
    """Wald test of E[z z' | bin] proportional to I.  The covariance of the moment vector is
    estimated by the (block) bootstrap; df = 2(k-1)."""
    n = len(u)
    m = isotropy_moments(u, W, k).mean(0)
    mb = np.array([isotropy_moments(u[idx] - u[idx].mean(0), W[idx], k).mean(0)
                   for idx in (boot_index(n, block, rng) for _ in range(B))])
    V = np.cov(mb.T)
    stat = float(m @ np.linalg.solve(V, m))
    return dict(k=k, wald=stat, df=len(m), p=float(chi2.sf(stat, len(m))))


def het_angle(u, W, k, grid=None):
    """Heteroskedasticity-identified rotation: the angle that jointly diagonalises the
    bin-conditional covariances of z (bivariate).  Returns degrees in [0, 90)."""
    z, L = whiten(u)
    b = bins_of(W, k)
    p = np.bincount(b, minlength=k) / len(b)
    Ms = [np.cov(z[b == j].T, bias=True) for j in range(k)]
    th = np.linspace(0, np.pi / 2, 9001)[:-1] if grid is None else grid
    c, s = np.cos(th), np.sin(th)
    obj = 0
    for pj, M in zip(p, Ms):
        off = (c * c - s * s) * M[0, 1] + c * s * (M[1, 1] - M[0, 0])
        obj = obj + pj * off ** 2
    t = th[int(np.argmin(obj))]
    D = [np.diag(rot(t).T @ M @ rot(t)) for M in Ms]
    return float(np.rad2deg(t)), L, np.array(D)
