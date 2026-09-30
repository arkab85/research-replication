"""Population operator norm ||C(eps1, eps2)|| (unit-bandwidth Gaussian kernels) for
eps_i = sigma * eta_i with a two-point common multiplier and demeaned unit-exponential
eta, by the spectral representation
  ||C||^2 = E_{s,t ~ N(0,1) iid} |phi_12(s,t) - phi_1(s) phi_2(t)|^2
evaluated with tensor Gauss--Hermite quadrature."""
import numpy as np


def phi_exp(s):
    return np.exp(-1j * s) / (1 - 1j * s)


def pop_norm_two_point(delta, nodes=200):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    sig = np.array([1 - delta, 1 + delta]) / np.sqrt(1 + delta ** 2)
    S, T = np.meshgrid(x, x, indexing='ij')
    p12 = sum(0.5 * phi_exp(g * S) * phi_exp(g * T) for g in sig)
    p1 = sum(0.5 * phi_exp(g * x) for g in sig)
    D = np.abs(p12 - p1[:, None] * p1[None, :]) ** 2
    return float(np.sqrt(w @ D @ w))


if __name__ == '__main__':
    for d in (0.0, 0.4):
        print(d, [pop_norm_two_point(d, m) for m in (100, 150, 200, 250)])
