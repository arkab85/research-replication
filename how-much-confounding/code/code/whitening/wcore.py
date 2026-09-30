"""Whitening (reduced-form covariance / Cholesky) inference for the SVAR certificate.

Vectorised residual bootstrap for VAR(p) models with an intercept, and the
candidate confidence regions for the lower-triangular Cholesky factor L of the
innovation covariance of a bivariate block.

Notation: l = vech(L) = (L11, L21, L22) for a 2x2 block; s = vech(Sigma)
= (S11, S21, S22).  All regions are for the population Cholesky factor of the
covariance of the selected innovation block.
"""
import numpy as np


# ---------------------------------------------------------------- VAR tools
def lagmat(y, p):
    T = len(y)
    return np.column_stack([np.ones(T - p)] + [y[p - j:T - j] for j in range(1, p + 1)])


def var_fit(y, p):
    X = lagmat(y, p)
    B = np.linalg.lstsq(X, y[p:], rcond=None)[0]
    return B, y[p:] - X @ B


def regen_batch(B, y0, U):
    """Regenerate a batch of series.  B: (1+kp, k); y0: (p, k); U: (nb, n, k)."""
    nb, n, k = U.shape
    p = y0.shape[0]
    A = B[1:].reshape(p, k, k)
    Y = np.empty((nb, p + n, k))
    Y[:, :p] = y0
    for t in range(p, p + n):
        acc = np.broadcast_to(B[0], (nb, k)).copy()
        for j in range(1, p + 1):
            acc += Y[:, t - j] @ A[j - 1]
        Y[:, t] = acc + U[:, t - p]
    return Y


def refit_batch(Y, p):
    """Batch OLS with intercept.  Returns residuals (nb, n, k)."""
    nb, T, k = Y.shape
    n = T - p
    X = np.concatenate([np.ones((nb, n, 1))] + [Y[:, p - j:T - j] for j in range(1, p + 1)], axis=2)
    XtX = np.einsum('bti,btj->bij', X, X)
    XtY = np.einsum('bti,btj->bij', X, Y[:, p:])
    Bh = np.linalg.solve(XtX, XtY)
    return Y[:, p:] - np.einsum('bti,bij->btj', X, Bh)


def iid_indices(n, nb, rng):
    return rng.integers(0, n, size=(nb, n))


def mbb_indices_batch(n, nb, block, rng):
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(nb, k))
    return (starts[:, :, None] + np.arange(block)).reshape(nb, -1)[:, :n]


# ------------------------------------------------------- covariance pieces
def vech2(S):
    """(…,2,2) -> (…,3) = (S11, S21, S22)."""
    return np.stack([S[..., 0, 0], S[..., 1, 0], S[..., 1, 1]], axis=-1)


def chol2(S):
    L11 = np.sqrt(S[..., 0, 0])
    L21 = S[..., 1, 0] / L11
    L22 = np.sqrt(np.maximum(S[..., 1, 1] - L21 ** 2, 1e-300))
    return np.stack([L11, L21, L22], axis=-1)


def jac_chol2(l):
    """d vech(L) / d vech(Sigma) at L with vech(L)=l, shape (…,3,3)."""
    L11, L21, L22 = l[..., 0], l[..., 1], l[..., 2]
    J = np.zeros(l.shape[:-1] + (3, 3))
    J[..., 0, 0] = 1 / (2 * L11)
    J[..., 1, 0] = -L21 / (2 * L11 ** 2)
    J[..., 1, 1] = 1 / L11
    J[..., 2, 0] = L21 ** 2 / (2 * L22 * L11 ** 2)
    J[..., 2, 1] = -L21 / (L11 * L22)
    J[..., 2, 2] = 1 / (2 * L22)
    return J


def xi(U):
    """Per-observation second-moment contributions vech(u_t u_t') of demeaned U, (…,n,3)."""
    Uc = U - U.mean(axis=-2, keepdims=True)
    return np.stack([Uc[..., 0] ** 2, Uc[..., 0] * Uc[..., 1], Uc[..., 1] ** 2], axis=-1)


def lrv_bartlett(X, ell):
    """Bartlett (overlapping-block) long-run covariance of rows of X, (…,n,d) -> (…,d,d).
    ell=1 gives the ordinary (1/n) sample covariance."""
    Xc = X - X.mean(axis=-2, keepdims=True)
    n = X.shape[-2]
    V = np.einsum('...ti,...tj->...ij', Xc, Xc) / n
    for j in range(1, ell):
        G = np.einsum('...ti,...tj->...ij', Xc[..., j:, :], Xc[..., :-j, :]) / n
        V = V + (1 - j / ell) * (G + np.swapaxes(G, -1, -2))
    return V


def lrv_blocks(X, ell):
    """Goetze--Kuensch studentisation for the moving-block bootstrap: covariance of
    non-overlapping consecutive block sums of the (bootstrap-ordered) series."""
    Xc = X - X.mean(axis=-2, keepdims=True)
    n = X.shape[-2]
    k = n // ell
    S = Xc[..., :k * ell, :].reshape(Xc.shape[:-2] + (k, ell, Xc.shape[-1])).sum(axis=-2)
    return np.einsum('...ti,...tj->...ij', S, S) / (k * ell)


def cov2(U):
    Uc = U - U.mean(axis=-2, keepdims=True)
    return np.einsum('...ti,...tj->...ij', Uc, Uc) / (U.shape[-2] - 1)


def quad(d, V):
    return np.einsum('...i,...i->...', d, np.linalg.solve(V, d[..., None])[..., 0])


def order_quantile(x, level):
    """Order-statistic quantile: the ceil(level*(B+1))-th smallest of B draws."""
    x = np.sort(np.asarray(x))
    k = int(np.ceil(level * (len(x) + 1)))
    return x[min(k, len(x)) - 1]


# ------------------------------------------------ per-series coefficient tools
def refit_batch_coef(Y, p):
    """Batch OLS with intercept.  Returns (coefficients (nb,1+kp,k), residuals (nb,n,k))."""
    nb, T, k = Y.shape
    n = T - p
    X = np.concatenate([np.ones((nb, n, 1))] + [Y[:, p - j:T - j] for j in range(1, p + 1)], axis=2)
    XtX = np.matmul(np.swapaxes(X, 1, 2), X)
    XtY = np.matmul(np.swapaxes(X, 1, 2), Y[:, p:])
    Bh = np.linalg.solve(XtX, XtY)
    return Bh, Y[:, p:] - np.matmul(X, Bh)


def regen_batch_coef(Bs, y0s, U):
    """Regenerate with series-specific coefficients.  Bs: (nb,1+kp,k); y0s: (nb,p,k); U: (nb,n,k)."""
    nb, n, k = U.shape
    p = y0s.shape[1]
    A = Bs[:, 1:].reshape(nb, p, k, k)
    Y = np.empty((nb, p + n, k))
    Y[:, :p] = y0s
    for t in range(p, p + n):
        acc = Bs[:, 0].copy()
        for j in range(1, p + 1):
            acc += np.einsum('bi,bij->bj', Y[:, t - j], A[:, j - 1])
        Y[:, t] = acc + U[:, t - p]
    return Y
