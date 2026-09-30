"""Data-generating designs for the whitening and SVAR calibration studies.

All designs are bivariate stable VAR(1) models with intercept zero,
coefficient matrix PHI and impact matrix A, innovations u_t = A eps_t with
eps_it = sigma_t * eta_it, E sigma_t^2 = 1, E eta = 0, E eta^2 = 1.
The population innovation covariance is A A'.
"""
import numpy as np

PHI = np.array([[.5, .1], [.2, .4]])
A_ROT = np.array([[1., .5], [.3, 1.]])      # rotation not on any grid (earlier study)
A_REC = np.array([[1., 0.], [.5, 1.]])      # first-ordered recursive structure is true
BURN = 100


def eta_draw(kind, size, rng):
    if kind == 'exp':      # skewed: demeaned unit exponential
        return rng.exponential(size=size) - 1.0
    if kind == 't5':       # symmetric heavy tails, unit variance
        return rng.standard_t(5, size=size) / np.sqrt(5 / 3)
    if kind == 'mix':      # skewed and heavier-tailed: standardised chi-square(2) with a
        # 20% scale contamination (scale 2.5), renormalised to unit variance.
        base = (rng.chisquare(2, size=size) - 2) / 2
        s = np.where(rng.random(size) < .2, 2.5, 1.0)
        return base * s / np.sqrt(.8 + .2 * 6.25)
    raise ValueError(kind)


def sigma_path(vol, m, rng):
    """Common volatility multiplier with E sigma^2 = 1.  Returns (sigma, rho)."""
    kind = vol[0]
    if kind == 'none':
        return np.ones(m), 0.0
    if kind == 'iid2':             # iid two-point, delta = vol[1]
        d = vol[1]
        s = rng.choice([-1., 1.], m)
        return (1 + d * s) / np.sqrt(1 + d * d), d / np.sqrt(1 + d * d)
    if kind == 'markov2':          # persistent two-state chain, stay probability vol[2]
        d, stay = vol[1], vol[2]
        s = np.empty(m)
        s[0] = rng.choice([-1., 1.])
        flips = rng.random(m) > stay
        for t in range(1, m):
            s[t] = -s[t - 1] if flips[t] else s[t - 1]
        return (1 + d * s) / np.sqrt(1 + d * d), d / np.sqrt(1 + d * d)
    if kind == 'logsv':            # log-normal stochastic volatility, AR(1) log-variance
        phi, sd = vol[1], vol[2]   # stationary sd of log sigma
        h = np.empty(m)
        h[0] = rng.normal(0, sd)
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), m)
        for t in range(1, m):
            h[t] = phi * h[t - 1] + e[t]
        # E exp(2h) = exp(2 sd^2);  sigma = exp(h - sd^2) has E sigma^2 = 1
        sig = np.exp(h - sd ** 2)
        rho = np.sqrt(1 - np.exp(-sd ** 2))   # Var(sigma) = 1 - (E sigma)^2, E sigma = exp(-sd^2/2)
        return sig, rho
    raise ValueError(kind)


def simulate(design, n, rng):
    """Return (y, L0, rho) with y having n+1 rows (VAR(1) leaves n residuals)."""
    A = design['A']
    m = BURN + n + 1
    sig, rho = sigma_path(design['vol'], m, rng)
    eps = eta_draw(design['eta'], (m, 2), rng) * sig[:, None]
    u = eps @ A.T
    y = np.zeros((m, 2))
    for t in range(1, m):
        y[t] = PHI @ y[t - 1] + u[t]
    L0 = np.linalg.cholesky(A @ A.T)
    return y[BURN:], L0, rho
