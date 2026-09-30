"""Bounded-confounding identification sets for a bivariate structural VAR block.

Rotations theta in [0, 90) degrees index orthogonal matrices modulo signed
permutations. For each rotation the candidate shocks are e(theta) = R(theta)' z with
z the Cholesky-whitened innovations; dependence is measured by the Gaussian-kernel
cross-covariance operator (unit bandwidths).  First-stage uncertainty (VAR slopes
and the whitening matrix) is handled by a residual-based moving block bootstrap in
the spirit of Brueggemann, Jentsch and Trenkler (2016): innovations are resampled
in blocks, series are regenerated from the fitted VAR, and everything is re-estimated.
"""
import numpy as np


def var_fit(y, p):
    T, k = y.shape
    X = np.column_stack([np.ones(T - p)] + [y[p - j:T - j] for j in range(1, p + 1)])
    Y = y[p:]
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    return B, Y - X @ B


def var_regenerate(B, y0, U):
    """Recursively generate a series from coefficients B, initial p rows y0, innovations U."""
    p = y0.shape[0]; k = y0.shape[1]; n = U.shape[0]
    y = np.zeros((p + n, k)); y[:p] = y0
    A = B[1:].reshape(p, k, k)          # A[j-1] multiplies y_{t-j}
    for t in range(p, p + n):
        acc = B[0].copy()
        for j in range(1, p + 1):
            acc += y[t - j] @ A[j - 1]
        y[t] = acc + U[t - p]
    return y


def mbb_indices(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def whiten(U):
    L = np.linalg.cholesky(np.cov(U.T))
    return np.linalg.solve(L, U.T).T, L


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _center(K):
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def gram(a, b=None):
    b = a if b is None else b
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2)


def op_norms(Z, thetas):
    """||C(theta)|| (Hilbert--Schmidt norm of the kernel cross-covariance) for each rotation."""
    n = len(Z); out = []
    for t in thetas:
        E = Z @ rot(t)
        out.append(np.sqrt(max(np.sum(_center(gram(E[:, 0])) * _center(gram(E[:, 1]))), 0)) / n)
    return np.array(out)


def op_diff_norms(Zb, Zo, thetas):
    """||C_boot(theta) - C_orig(theta)|| for each rotation, via cross Gram matrices."""
    nb, no = len(Zb), len(Zo); out = []
    for t in thetas:
        R = rot(t); Eb = Zb @ R; Eo = Zo @ R
        Sbb = np.sum(_center(gram(Eb[:, 0])) * _center(gram(Eb[:, 1]))) / nb ** 2
        Soo = np.sum(_center(gram(Eo[:, 0])) * _center(gram(Eo[:, 1]))) / no ** 2
        K1 = gram(Eb[:, 0], Eo[:, 0]); K2 = gram(Eb[:, 1], Eo[:, 1])
        K1c = K1 - K1.mean(0, keepdims=True) - K1.mean(1, keepdims=True) + K1.mean()
        K2c = K2 - K2.mean(0, keepdims=True) - K2.mean(1, keepdims=True) + K2.mean()
        Sbo = np.sum(K1c * K2c) / (nb * no)
        out.append(np.sqrt(max(Sbb + Soo - 2 * Sbo, 0)))
    return np.array(out)


def coskew(Z, thetas):
    """Sample co-skewness moments (E e1^2 e2, E e1 e2^2) of standardized rotated shocks."""
    out = []
    for t in thetas:
        E = Z @ rot(t); E = (E - E.mean(0)) / E.std(0)
        out.append([np.mean(E[:, 0] ** 2 * E[:, 1]), np.mean(E[:, 0] * E[:, 1] ** 2)])
    return np.array(out)


def demand_elasticity(L, t):
    """Supply elasticity implied by rotation t: production over price response to the demand shock,
    the shock that moves production and price in the same direction."""
    A = L @ rot(t)
    el = [A[0, j] / A[1, j] for j in range(2) if A[0, j] * A[1, j] >= 0 and abs(A[1, j]) > 1e-12]
    return min(el, key=abs) if el else np.nan
