"""Proxy-adjusted (conditional) operator and bandwidth-set tools.

The proxy-adjusted operator at rotation theta is
    C^{|W_k}(theta) = sum_b p_b C^{(b)}(theta),
the bin-weighted average of within-bin cross-covariance operators of the two rotated
components, where bins are k quantile bins of the proxy W.  Its sample version is
(1/n) sum_t phi~_t (x) psi~_t with features centred within bins, so that
    <C_E, C_F> = (nE nF)^{-1} sum_{t,s} K1~_{ts} K2~_{ts},
    K~ = (I - P_E) K (I - P_F)',
with P the within-bin averaging operator.  With a single bin this is the ordinary
operator of fastop.op_inner_fast (checked in __main__).
"""
import numpy as np


def gram(a, b, bw=1.0):
    d = np.subtract.outer(a, b)
    np.multiply(d, d, out=d)
    d *= -0.5 / (bw * bw)
    np.exp(d, out=d)
    return d


def _center(K, bE, kE, bF, kF):
    """(I - P_E) K (I - P_F)' for bin labels bE (rows) and bF (columns)."""
    cF = np.bincount(bF, minlength=kF).astype(float)
    GF = np.zeros((len(bF), kF)); GF[np.arange(len(bF)), bF] = 1.0
    colmean = (K @ GF) / cF                        # nE x kF: mean over columns within F-bin
    K = K - colmean[:, bF]
    cE = np.bincount(bE, minlength=kE).astype(float)
    GE = np.zeros((len(bE), kE)); GE[np.arange(len(bE)), bE] = 1.0
    rowmean = (GE.T @ K) / cE[:, None]             # kE x nF
    return K - rowmean[bE, :]


def op_inner_cond(E, bE, F, bF, k, bw=1.0):
    K1 = _center(gram(E[:, 0], F[:, 0], bw), bE, k, bF, k)
    K2 = _center(gram(E[:, 1], F[:, 1], bw), bE, k, bF, k)
    return float(np.einsum('ij,ij->', K1, K2) / (len(E) * len(F)))


def op_inner_bw(E, F, bw=1.0):
    """Ordinary (unconditional) operator inner product with bandwidth bw."""
    z = np.zeros(len(E), int); zf = np.zeros(len(F), int)
    return op_inner_cond(E, z, F, zf, 1, bw)


def quantile_bins(W, k):
    r = np.argsort(np.argsort(W, kind='stable'), kind='stable')
    return (r * k // len(W)).astype(int)


def dev(Eb, bb, E, b, k, v, bw=1.0):
    """||C*_b - C|| given the original squared norm v."""
    return np.sqrt(max(0., op_inner_cond(Eb, bb, Eb, bb, k, bw) + v - 2 * op_inner_cond(Eb, bb, E, b, k, bw)))


def qtile(x, level):
    B = len(x)
    return float(np.sort(x)[min(int(np.ceil(level * (B + 1))), B) - 1])


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'whitening'))
    from fastop import op_inner_fast
    rng = np.random.default_rng(1)
    E = rng.standard_normal((200, 2)); F = rng.standard_normal((150, 2))
    a = op_inner_fast(E, F); b = op_inner_bw(E, F)
    print('unconditional check', a, b, abs(a - b))
    # conditional version equals bin-weighted within-bin operator (direct feature check with random Fourier)
    W = rng.standard_normal(200); bE = quantile_bins(W, 3)
    s = op_inner_cond(E, bE, E, bE, 3)
    # direct: sum_{b,b'} p_b p_b' <C_b, C_b'>, <C_b,C_b'> via ordinary op inner on sub-samples? (not
    # separable), so check against brute-force explicit centring of the Gram matrices
    K1 = gram(E[:, 0], E[:, 0]); K2 = gram(E[:, 1], E[:, 1])
    P = np.zeros((200, 200))
    for j in range(3):
        idx = np.where(bE == j)[0]; P[np.ix_(idx, idx)] = 1 / len(idx)
    H = np.eye(200) - P
    s2 = np.sum((H @ K1 @ H) * (H @ K2 @ H)) / 200 ** 2
    print('conditional check', s, s2, abs(s - s2))
