"""Design of flows/PROTOCOL.md. The estimator is code/engine.py, unchanged; the pooled bootstraps are core_lib.pooled_contrast, unchanged.
This file adds the flow innovation, training-block standardization, and an enumerated rotation test whose inner loop avoids
building data frames (checked against the engine to machine precision by `run_flows.py check`)."""
import os, sys
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, '1')      # before numpy loads: the matrices are small, and parts run side by side
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
import engine
from engine import build, Stage1, sieve
engine_cgram = engine.cgram


def fast_cgram(Z):
    """engine.cgram with the double centering H K H done by subtracting column and row means (O(n^2)) instead of two matrix
    products (O(n^3)). Same kernel, same bandwidth; `run_flows.py check` compares the two."""
    Z = np.asarray(Z, float); Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0; K = np.exp(-d2 / (2 * (med / 2.0) + 1e-12))
    K -= K.mean(0); K -= K.mean(1)[:, None]; return K


engine.cgram = fast_cgram                    # hsic2, dii and core_q look cgram up in the engine module, so the bootstraps use it too
from engine import hsic2, dii
import core_lib as RL
DATA = os.path.join(HERE, 'data'); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)

SETS = {
    'set1': dict(file='weekly_panel.csv', hs=(1, 2, 4, 8, 13, 26), gap=26,
                 markets=['CAD', 'EUR', 'JPY', 'CHF', 'GBP', 'MXN', 'SPX', 'NDX', 'UST2', 'UST5', 'UST10', 'USTB'],
                 groups={'currencies': ['CAD', 'EUR', 'JPY', 'CHF', 'GBP', 'MXN'], 'equity': ['SPX', 'NDX'], 'treasuries': ['UST2', 'UST5', 'UST10', 'USTB']}),
    'set2': dict(file='monthly_panel.csv', hs=(1, 3, 6), gap=6, markets=['AUD', 'CAD', 'DKK', 'JPY', 'NOK', 'SEK', 'CHF', 'GBP'], groups={}),
    'set2nine': dict(file='monthly_panel.csv', hs=(1, 3, 6), gap=6, markets=['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP'], groups={}),
}


def stats_for(hs):
    S = {f'P({h})': {h: 1.0} for h in hs}; S['A'] = {h: 1.0 / len(hs) for h in hs}
    if 26 in hs: S['C'] = {**{h: 1.0 / (len(hs) - 1) for h in hs if h <= 13}, 26: -1.0}
    else: S['H'] = dict(RL.HUMP)
    return S


def load(name):
    cfg = SETS[name]; W = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0)
    W = W[[c + m for m in cfg['markets'] for c in ('f_', 'r_')] + [c for c in W.columns if c.startswith('rw_') and c[3:] in cfg['markets']]].dropna()
    assert np.isfinite(W.values).all(), 'non-finite cell in ' + name; return W, cfg


def split_n(T, gap): n = int(np.floor(T / np.log(T))); return n, T - n - gap


def prepare(f, r, gap, direction, raw=False, innovate=True):
    """Flow innovation (residual of f on two own lags, coefficients from the training block), standardization on the training
    block, and the cause/outcome assignment. The first two observations are lost to the lags. The same-period return is NOT in
    the innovation regression (PROTOCOL.md, Section 9.3): a cause built from the outcome is not held fixed by a rotation of the
    outcome. The same-period return is conditioned on in both first stages, as Y_t in the engine's conditioning set."""
    T = len(f) - 2; n, tr = split_n(T, gap); y = f[2:]; rr = r[2:].astype(float).copy()
    Z = np.column_stack([np.ones(T), f[1:-1], f[:-2]]); beta = np.linalg.lstsq(Z[:tr], y[:tr], rcond=None)[0]; u = (y - Z @ beta) if innovate else y.astype(float).copy()
    if not raw: u = (u - u[:tr].mean()) / u[:tr].std(); rr = (rr - rr[:tr].mean()) / rr[:tr].std()
    # third value: the series whose same-period relation with the outcome the impact-preserving rotation keeps (PROTOCOL 10)
    if direction == 'FR': return u, rr, u
    if direction == 'RF': return rr, u, rr
    if direction == 'FR+': return np.maximum(u, 0.0), rr, u
    if direction == 'FR-': return np.minimum(u, 0.0), rr, u
    raise ValueError(direction)


def design(X, Y, h, p):
    t = np.arange(p, len(X) - h); C = np.column_stack([Y[t]] + [X[t - i] for i in range(1, p + 1)] + [Y[t - i] for i in range(1, p + 1)])
    return Y[t + h], X[t], np.column_stack([X[t], C]), np.column_stack([Y[t + h], C])


def stat(s1, X, Y, h, p):
    yf, xb, Zf, Zb = design(X, Y, h, p)
    ef = yf - s1.mf.predict(sieve(s1.scf.transform(Zf))); eb = xb - s1.mb.predict(sieve(s1.scb.transform(Zb))); return hsic2(eb, Zb) - hsic2(ef, Zf)


def make_units(panel, markets, hs, gap, direction, p=2, raw=False, dates=None, innovate=True):
    """{h: [core_lib unit, ...]} with the test-block arrays attached, one unit per market."""
    U = {h: [] for h in hs}
    for m in markets:
        X, Y, S = prepare(panel['f_' + m].values, panel['r_' + m].values, gap, direction, raw, innovate); idx = (dates if dates is not None else np.arange(len(panel)))[2:]
        d = pd.DataFrame({'x': X, 'y': Y}, index=idx); n, tr = split_n(len(d), gap)
        for h in hs:
            u = RL.unit(d, h, p, gap); u['Xte'] = X[tr + gap:]; u['Yte'] = Y[tr + gap:]; u['Ste'] = S[tr + gap:]; u['market'] = m; u['h'] = h; u['p'] = p; U[h].append(u)
    return U


def _split_same_period(u, basis='linear'):
    """Evaluation-block least-squares fit of the outcome on the same-period cause: (fit, residual), cached on the unit.
    basis='sieve' fits the same-period relation with the sieve of the residualizing regressions (used only in the positive
    control, where the same-period relation of volatility to the return is V-shaped)."""
    if ('ip', basis) not in u:
        S = u['Ste']; s = (S - S.mean()) / S.std(); Z = np.column_stack([np.ones(len(S)), s if basis == 'linear' else sieve(s.reshape(-1, 1))])
        fit = Z @ np.linalg.lstsq(Z, u['Yte'], rcond=None)[0]; u['ip', basis] = (fit, u['Yte'] - fit)
    return u['ip', basis]


def rotate(U, hs, p=2, step=1, ip=False):
    """Every rotation admissible at every horizon, and ONE integer calendar shift k applied to every market and horizon, so that
    each reference draw of a multi-horizon statistic is that statistic on one rotated data set (PROTOCOL.md, Sections 6 and 9.4).
    step > 1 keeps every step-th shift (used only in the Step 0 audits, to save time). The observed value goes through the same
    routine as the rotated ones. ip=True is the impact-preserving rotation of PROTOCOL 10: the same-period fit of the outcome on
    the cause stays in place and only its residual is rotated, Y(k) = fit + roll(w, k); Y(0) is the outcome itself."""
    n = len(U[hs[0]][0]['Xte']); lo = max(hs) + p + 2; ks = range(lo, n - lo, step)
    def turned(u, k):
        if not ip: return np.roll(u['Yte'], k)
        fit, w = _split_same_period(u, 'sieve' if ip == 'sieve' else 'linear'); return fit + np.roll(w, k)
    obs = {h: np.array([stat(u['s1'], u['Xte'], u['Yte'], h, p) for u in U[h]]) for h in hs}
    M = {h: np.array([[stat(u['s1'], u['Xte'], turned(u, k), h, p) for u in U[h]] for k in ks]) for h in hs}
    assert all(np.isfinite(obs[h]).all() and np.isfinite(M[h]).all() for h in hs), 'non-finite index'
    return obs, M, len(ks)


def _arrays(u):
    if 'arr' not in u:
        yf, xb, Zf, Zb = design(u['Xte'], u['Yte'], u['h'], u['p']); s1 = u['s1']
        u['arr'] = (yf - s1.mf.predict(sieve(s1.scf.transform(Zf))), xb - s1.mb.predict(sieve(s1.scb.transform(Zb))), Zf, Zb)
    return u['arr']


def paired_draws(U, hs, B, seed):
    """The paired block bootstrap of core_lib.pooled_contrast: one vector of relative block starts per draw, shared by all
    markets and horizons, first stages fixed. With a fixed first stage the residuals of resampled rows are the resampled
    residuals, so nothing is refitted or rebuilt. Returns {h: (B, markets)} of resampled indices."""
    units = [u for h in hs for u in U[h]]; nbar = int(np.mean([u['n'] for u in units])); ell = max(2, int(round(nbar ** (1 / 3))))
    rng = np.random.RandomState(seed + 1); nmax = max(u['n'] for u in units); D = {h: np.empty((B, len(U[h]))) for h in hs}
    for b in range(B):
        uu = rng.rand(int(np.ceil(nmax / ell)) + 1)
        for h in hs:
            for i, u in enumerate(U[h]):
                n = u['n']; nb = int(np.ceil(n / ell)); starts = (uu[:nb] * (n - ell + 1)).astype(int); ix = np.concatenate([np.arange(s, s + ell) for s in starts])[:n]
                ef, eb, Zf, Zb = _arrays(u); D[h][b, i] = hsic2(eb[ix], Zb[ix]) - hsic2(ef[ix], Zf[ix])
    return D


def paired_p(U, D, w):
    P = sum(w[h] * np.mean([u['obs'] for u in U[h]]) for h in w); bp = sum(w[h] * D[h].mean(1) for h in w); return (np.sum((bp - P) >= P) + 1) / (len(bp) + 1)


def rot_p(obs, M, K, w, cols=None, two_sided=False):
    c = slice(None) if cols is None else cols; o = sum(w[h] * obs[h][c].mean() for h in w); b = sum(w[h] * M[h][:, c].mean(1) for h in w)
    if two_sided: return o, b.mean(), (1 + np.sum(np.abs(b - b.mean()) >= abs(o - b.mean()))) / (K + 1)
    return o, b.mean(), (1 + np.sum(b >= o)) / (K + 1)


def holm(pv):
    """Holm-adjusted p-values for a dict of p-values."""
    k = sorted(pv, key=pv.get); m = len(k); out = {}; run = 0.0
    for i, key in enumerate(k): run = max(run, min(1.0, (m - i) * pv[key])); out[key] = run
    return out


def sim_outcomes(T, N, rng, rho=0.3, ar=0.05, a=0.10, b=0.85, df=6):
    """N outcome series independent of everything else: AR(1)-GARCH(1,1), Student-t innovations, one common factor."""
    def garch():
        z = rng.standard_t(df, T + 200) / np.sqrt(df / (df - 2)); e = np.empty(T + 200); s2 = 1.0; x = 0.0; out = np.empty(T + 200)
        for t in range(T + 200):
            e[t] = np.sqrt(s2) * z[t]; x = ar * x + e[t]; out[t] = x; s2 = (1 - a - b) + a * e[t] ** 2 + b * s2
        return out[200:]
    g = garch(); return np.column_stack([np.sqrt(rho) * g + np.sqrt(1 - rho) * garch() for _ in range(N)])
