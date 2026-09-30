"""Exact Gaussian-kernel operator inner products without forming centred matrices.
Algebraically identical to certified_revision.op_inner (checked in validate_whitening.py)."""
import numpy as np


def _gram(a, b):
    d = np.subtract.outer(a, b)
    np.multiply(d, d, out=d)
    d *= -0.5
    np.exp(d, out=d)
    return d


def op_inner_fast(E, F):
    K1 = _gram(E[:, 0], F[:, 0]); K2 = _gram(E[:, 1], F[:, 1])
    nE, nF = K1.shape
    r1 = K1.sum(1); r2 = K2.sum(1); c1 = K1.sum(0); c2 = K2.sum(0)
    s = np.einsum('ij,ij->', K1, K2) - r1 @ r2 / nF - c1 @ c2 / nE + r1.sum() * r2.sum() / (nE * nF)
    return s / (nE * nF)
