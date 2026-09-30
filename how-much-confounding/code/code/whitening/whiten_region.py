"""Studentised (bootstrap-t) confidence region for the whitening factor, and its
exact linear projection into impact coefficients.  This is the procedure frozen
in PRESPECIFICATION.md before the confirmatory calibration study.

Procedure (bivariate block `cols` of a k-variable VAR(p) with intercept):
 1. OLS residuals U (n x k), K = 1 + k p regressors per equation.
 2. Sigma_hat = U_b'U_b/(n-K) for the block, l_hat = vech(chol Sigma_hat).
 3. V_l = J V_s J', where V_s is the Bartlett long-run covariance of
    xi_t = vech(u_bt u_bt') with bandwidth ell (ell = 1: sample covariance)
    and J is the derivative of the Cholesky map.
 4. Residual bootstrap: resample sqrt(n/(n-K)) U (iid if ell = 1, moving
    blocks of length ell otherwise), regenerate the VAR from the fitted
    coefficients and observed initial values, refit, and compute
    T* = n (l* - l_hat)' V*^{-1} (l* - l_hat), where V* uses the
    Goetze--Kuensch block studentisation (non-overlapping consecutive blocks
    of the bootstrap residual series; ell = 1: sample covariance).
 5. c = ceil(level (B+1))-th order statistic of T*.
 6. Region E = {l : n (l - l_hat)' V_l^{-1} (l - l_hat) <= c}.
"""
import numpy as np
from wcore import (var_fit, regen_batch, refit_batch, iid_indices, mbb_indices_batch, chol2,
                   jac_chol2, xi, lrv_bartlett, lrv_blocks, quad, order_quantile)

LEVEL = .975


def _block_stats(Ub, K, ell, boot):
    n = Ub.shape[-2]
    Uc = Ub - Ub.mean(axis=-2, keepdims=True)
    S = np.einsum('...ti,...tj->...ij', Uc, Uc) / (n - K)
    l = chol2(S)
    X = xi(Ub)
    if ell == 1:
        Vs = lrv_bartlett(X, 1)
    else:
        Vs = lrv_blocks(X, ell) if boot else lrv_bartlett(X, ell)
    J = jac_chol2(l)
    return l, J @ Vs @ np.swapaxes(J, -1, -2)


def studentized_region(y, p, cols, ell, B, rng, level=LEVEL, chunk=250, fit=None):
    """Return a dict describing the region.  `fit` may pass (Bcoef, U) to reuse a fit."""
    Bh, U = var_fit(y, p) if fit is None else fit
    n, k = U.shape
    K = 1 + k * p
    l, Vl = _block_stats(U[:, cols], K, ell, False)
    Ur = U * np.sqrt(n / (n - K))
    T = np.empty(B)
    for c0 in range(0, B, chunk):
        m = min(chunk, B - c0)
        idx = iid_indices(n, m, rng) if ell == 1 else mbb_indices_batch(n, m, ell, rng)
        Ys = regen_batch(Bh, y[:p], Ur[idx])
        Us = refit_batch(Ys, p)
        ls, Vls = _block_stats(Us[:, :, cols], K, ell, True)
        T[c0:c0 + m] = n * quad(ls - l, Vls)
    c = float(order_quantile(T, level))
    return dict(l=l, V=Vl, c=c, n=n, T=T, K=K, ell=ell, level=level)


def contains(reg, l0):
    return bool(reg['n'] * quad(reg['l'] - np.asarray(l0), reg['V']) <= reg['c'])


def linear_range(reg, a):
    """Exact range of a'l over the ellipsoid."""
    a = np.asarray(a, float)
    w = np.sqrt(reg['c'] * (a @ reg['V'] @ a) / reg['n'])
    m = a @ reg['l']
    return m - w, m + w


def row_norm_sup(reg):
    """Upper bounds on ||L_1.|| = |L11| and ||L_2.|| = ||(L21, L22)|| over the ellipsoid."""
    l, V, c, n = reg['l'], reg['V'], reg['c'], reg['n']
    r1 = abs(l[0]) + np.sqrt(c * V[0, 0] / n)
    r2 = np.hypot(l[1], l[2]) + np.sqrt(c * np.linalg.eigvalsh(V[1:, 1:])[-1] / n)
    return np.array([r1, r2])


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def entry_bounds(reg, a, b):
    """Certified bounds on every entry of L R(theta), L in the ellipsoid, theta in [a,b].
    Returns lo, hi arrays of shape (2,2)."""
    m = (a + b) / 2
    R = rot(m)
    e = row_norm_sup(reg) * 2 * np.sin((b - a) / 4)
    lo = np.empty((2, 2)); hi = np.empty((2, 2))
    for k in range(2):
        for i, coef in ((0, [R[0, k], 0, 0]), (1, [0, R[0, k], R[1, k]])):
            x0, x1 = linear_range(reg, coef)
            lo[i, k] = x0 - e[i]; hi[i, k] = x1 + e[i]
    return lo, hi


def elasticity_cells(reg, TH):
    """Outer ranges of the production/price impact ratio for co-moving columns, per cell."""
    bounds = []
    for a, b in zip(TH[:-1], TH[1:]):
        lo, hi = entry_bounds(reg, a, b)
        intervals = []
        for k in range(2):
            xlo, xhi, ylo, yhi = lo[0, k], hi[0, k], lo[1, k], hi[1, k]
            for sign in (1, -1):
                xl, xh = (xlo, xhi) if sign == 1 else (-xhi, -xlo)
                yl, yh = (ylo, yhi) if sign == 1 else (-yhi, -ylo)
                if xh >= 0 and yh > 0:
                    intervals.append((max(0, xl) / yh, float('inf') if yl <= 0 else max(0, xh) / yl))
        bounds.append([min(v[0] for v in intervals), max(v[1] for v in intervals)] if intervals else [float('nan')] * 2)
    return np.array(bounds)
