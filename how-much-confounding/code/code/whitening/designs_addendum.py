"""Addendum designs (weak identification and population norms for persistent designs).
Extends designs.py without modifying it (designs.py is hash-protected by PRESPECIFICATION.md)."""
import numpy as np
from designs import PHI, BURN, sigma_path, eta_draw as eta_base


def eta_draw(kind, size, rng):
    if kind == 'gauss':        # Gaussian shocks: independence does not identify the rotation
        return rng.standard_normal(size)
    if kind == 'chi40':        # weakly non-Gaussian: standardised chi-square(40), skewness .45
        return (rng.chisquare(40, size=size) - 40) / np.sqrt(80)
    return eta_base(kind, size, rng)


def simulate(design, n, rng):
    A = design['A']
    m = BURN + n + 1
    sig, rho = sigma_path(design['vol'], m, rng)
    eps = eta_draw(design['eta'], (m, 2), rng) * sig[:, None]
    u = eps @ A.T
    y = np.zeros((m, 2))
    for t in range(1, m):
        y[t] = PHI @ y[t - 1] + u[t]
    return y[BURN:], np.linalg.cholesky(A @ A.T), rho


# ---------------- population operator norm for general (eta, sigma) marginals
def phi_eta(kind, t):
    t = np.asarray(t, complex)
    ex = lambda s: np.exp(-1j * s) / (1 - 1j * s)          # demeaned unit exponential
    if kind == 'exp':
        return ex(t)
    if kind == 'mix':                                     # (Exp-1) times scale, renormalised
        k = np.sqrt(.8 + .2 * 6.25)
        return .8 * ex(t / k) + .2 * ex(2.5 * t / k)
    raise ValueError(kind)


def sigma_nodes(vol, nodes=80):
    """Nodes and weights of the marginal law of the common multiplier."""
    if vol[0] in ('iid2', 'markov2'):
        d = vol[1]
        return np.array([1 - d, 1 + d]) / np.sqrt(1 + d * d), np.array([.5, .5])
    if vol[0] == 'logsv':
        sd = vol[2]
        x, w = np.polynomial.hermite_e.hermegauss(nodes); w = w / w.sum()
        return np.exp(sd * x - sd ** 2), w
    raise ValueError(vol)


def pop_norm(eta, vol, nodes=200):
    x, w = np.polynomial.hermite_e.hermegauss(nodes); w = w / w.sum()
    sv, sw = sigma_nodes(vol)
    S, T = np.meshgrid(x, x, indexing='ij')
    p12 = sum(q * phi_eta(eta, g * S) * phi_eta(eta, g * T) for g, q in zip(sv, sw))
    p1 = sum(q * phi_eta(eta, g * x) for g, q in zip(sv, sw))
    D = np.abs(p12 - p1[:, None] * p1[None, :]) ** 2
    return float(np.sqrt(w @ D @ w))
