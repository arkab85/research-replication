"""Local (second-order) sensitivity constants for residual-HSIC components.

Bivariate Gaussian reference without history:
    response  V1 = c1*U + s1*n1,   regressor V2 = c2*U + s2*n2,
with U, n1, n2 independent N(0,1).  A no-channel perturbation adds
    delta*F(U) to V1 and delta*G(U) to V2,
where F, G are polynomial loadings orthogonal to 1 and U.  The residual is
V1 - E_P[V1 | V2]; HSIC uses Gaussian kernels with bandwidth s_eps on the
residual and s_z on the regressor.

local_matrix(...) returns M with  lim_{delta->0} H(delta)/delta^2 = c' M c
for the loading sum_n c_n psi_n, psi_n = He_n / sqrt(n!), n = 2..N+1.
exact_H(...) approximates H(delta) by deterministic quadrature; numerical integration error remains.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval
from math import factorial, sqrt, pi


def gh(n):
    x, w = hermegauss(n)
    return x, w / np.sqrt(2 * pi)          # expectation weights under N(0,1)


def basis(n):
    """psi_n = He_n/sqrt(n!) and its derivative."""
    c = np.zeros(n + 1); c[n] = 1.0 / sqrt(factorial(n))
    dc = np.zeros(n); dc[n - 1] = n / sqrt(factorial(n)) if n >= 1 else 0.0
    f = lambda u, c=c: hermeval(u, c)
    df = lambda u, dc=dc: hermeval(u, dc)
    return f, df


class Setup:
    def __init__(self, c1, s1, c2, s2, s_eps=1.0, s_z=1.0, nq=60, nf=48):
        self.c1, self.s1, self.c2, self.s2 = c1, s1, c2, s2
        self.v2 = c2 ** 2 + s2 ** 2
        self.beta = c1 * c2 / self.v2
        self.mu = c2 / self.v2
        self.tau = s2 / sqrt(self.v2)
        self.var_eps = (c1 - self.beta * c2) ** 2 + s1 ** 2 + self.beta ** 2 * s2 ** 2
        self.s_eps, self.s_z = s_eps, s_z
        xu, wu = gh(nq)
        U, N2 = np.meshgrid(xu, xu, indexing="ij")
        self.U = U.ravel(); self.N2 = N2.ravel(); self.W = np.outer(wu, wu).ravel()
        self.V2 = c2 * self.U + s2 * self.N2
        self.EPS0 = (c1 - self.beta * c2) * self.U - self.beta * s2 * self.N2   # eps without n1
        xf, wf = gh(nf)
        T, S = np.meshgrid(xf / s_eps, xf / s_z, indexing="ij")
        self.T = T.ravel(); self.S = S.ravel(); self.WTS = np.outer(wf, wf).ravel()
        self.xp, self.wp = gh(80)             # posterior quadrature

    # conditional expectation E[h(U) | V2 = z] under the reference
    def cond(self, h, z):
        u = self.mu * z[:, None] + self.tau * self.xp[None, :]
        return (h(u) * self.wp[None, :]).sum(1)

    def m1(self, F, G, z, hstep=1e-5):
        c1, beta, v2 = self.c1, self.beta, self.v2
        def k(zz):
            return self.cond(lambda u: c1 * u * G(u), zz) - beta * zz * self.cond(G, zz)
        EFG = self.cond(F, z) - beta * self.cond(G, z)
        kp = (k(z + hstep) - k(z - hstep)) / (2 * hstep)
        return EFG + (z / v2) * k(z) - kp

    def delta_prime(self, F, G):
        """First derivative in delta of phi_joint - phi_eps*phi_V2 on the (T,S) grid."""
        W = F(self.U) - self.beta * G(self.U) - self.m1(F, G, self.V2)
        Gv = G(self.U)
        T, S = self.T, self.S
        ph = np.exp(1j * (np.outer(self.EPS0, T) + np.outer(self.V2, S)))      # (q, f)
        fac1 = np.exp(-0.5 * T ** 2 * self.s1 ** 2)
        A = ((self.W * W)[:, None] * ph).sum(0) * 1j * T + ((self.W * Gv)[:, None] * ph).sum(0) * 1j * S
        A *= fac1
        phe = np.exp(1j * np.outer(self.EPS0, T))
        B = ((self.W * W)[:, None] * phe).sum(0) * 1j * T * fac1
        phz = np.exp(1j * np.outer(self.V2, S))
        C = ((self.W * Gv)[:, None] * phz).sum(0) * 1j * S
        phi_eps = np.exp(-0.5 * T ** 2 * self.var_eps)
        phi_z = np.exp(-0.5 * S ** 2 * self.v2)
        return A - B * phi_z - phi_eps * C

    def local_matrix(self, load, N=10):
        """load='resp' (F=psi_n) or 'reg' (G=psi_n); returns N x N matrix M."""
        zero = lambda u: np.zeros_like(u)
        D = []
        for n in range(2, N + 2):
            f, _ = basis(n)
            D.append(self.delta_prime(f, zero) if load == "resp" else self.delta_prime(zero, f))
        D = np.array(D)
        M = np.real((D * self.WTS[None, :]) @ np.conj(D).T)
        return 0.5 * (M + M.T)

    def exact_H(self, F, G, delta, npost=400):
        """HSIC of (V1 - E_P[V1|V2], V2) under the perturbed law, by quadrature."""
        xp = np.linspace(-12, 12, 4801); wp = np.exp(-0.5 * xp ** 2) * (xp[1] - xp[0]) / np.sqrt(2 * pi)
        c1, c2, s1, s2 = self.c1, self.c2, self.s1, self.s2
        V1 = c1 * self.U + delta * F(self.U)          # without s1*n1
        V2 = c2 * self.U + delta * G(self.U) + s2 * self.N2
        # m_P(z) = E[c1 U + delta F(U) | V2^P = z]
        ll = -0.5 * ((V2[:, None] - c2 * xp[None, :] - delta * G(xp)[None, :]) / s2) ** 2 + np.log(wp)[None, :]
        lik = np.exp(ll - ll.max(1, keepdims=True))
        num = (lik * (c1 * xp + delta * F(xp))[None, :]).sum(1)
        mP = num / lik.sum(1)
        R0 = V1 - mP
        T, S = self.T, self.S
        fac1 = np.exp(-0.5 * T ** 2 * s1 ** 2)
        joint = (self.W[:, None] * np.exp(1j * (np.outer(R0, T) + np.outer(V2, S)))).sum(0) * fac1
        mR = (self.W[:, None] * np.exp(1j * np.outer(R0, T))).sum(0) * fac1
        mZ = (self.W[:, None] * np.exp(1j * np.outer(V2, S))).sum(0)
        d = joint - mR * mZ
        return float(np.sum(self.WTS * np.abs(d) ** 2))


def directions(a, sx, b, sy, **kw):
    """Backward (X on Y) and forward (Y on X) setups for X = aU+sx*xi, Y = bU+sy*e."""
    back = Setup(a, sx, b, sy, **kw)     # response X, regressor Y
    fwd = Setup(b, sy, a, sx, **kw)      # response Y, regressor X
    return back, fwd


def kappa(a, sx, b, sy, N=10, **kw):
    """Local constants. Returns dict with vertex eigenvalues and least-favourable shapes."""
    back, fwd = directions(a, sx, b, sy, **kw)
    out = {}
    # X loading: response in backward, regressor in forward. Y loading: the reverse.
    for vert, lb, lf in (("X", "resp", "reg"), ("Y", "reg", "resp")):
        Mb = back.local_matrix(lb, N); Mf = fwd.local_matrix(lf, N)
        ev, V = np.linalg.eigh(Mb)
        c = V[:, -1]
        evd, Vd = np.linalg.eigh(Mb - Mf)
        out[vert] = dict(kb=ev[-1], shape=c, Hf_at_shape=float(c @ Mf @ c), kD=evd[-1], Mb=Mb, Mf=Mf)
    out["kappa_b"] = max(out["X"]["kb"], out["Y"]["kb"])
    return out


def K_env(a, sx, b, sy, s_eps=1.0, s_z=1.0):
    """Envelope constant K_b of Corollary 1 for the bivariate reference (backward: X on Y)."""
    v2 = b * b + sy * sy
    beta = a * b / v2
    vX = a * a + sx * sx - (a * b) ** 2 / v2
    AX = (1 + sqrt(vX / sx ** 2)) / s_eps
    AY = (abs(beta) + sqrt(vX / sy ** 2)) / s_eps + 1.0 / s_z
    return 4 * max(AX, AY) ** 2


def shape_fun(c):
    fs = [basis(n)[0] for n in range(2, len(c) + 2)]
    return lambda u: sum(ci * f(u) for ci, f in zip(c, fs))
