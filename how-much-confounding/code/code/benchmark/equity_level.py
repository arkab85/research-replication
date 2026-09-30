"""Equity robustness: VAR(4) in returns and log-VIX changes augmented with the lagged log-VIX level.

The overidentifying restrictions of the regime-conditional covariances (Section 5.1) are rejected
with three and five regimes in the baseline VAR in changes, and the baseline innovations have
within-regime means that differ from zero (the VIX-change innovation is predictable from the VIX
level, i.e. mean reversion that a VAR in changes omits).  This script adds v_{t-1} = log VIX_{t-1}
as a regressor (log-VIX changes are in percent, so the level enters in log units), which is the error-correction form, and recomputes: within-regime means; the
recursive-ordering operator tests (joint .95, blocks of ten, 399 refitted draws with the main
seeds, regenerating the level recursively); the one-degree certified set at zero exposure; the
heteroskedasticity rotation with k = 2, 3, 5; and the robust (AR) rotation sets.
Writes results/equity_level.json.  Run: OPENBLAS_NUM_THREADS=1 python benchmark/equity_level.py
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from bcore import rot, whiten, boot_index, het_angle, CURV
from acore import op_inner_cond, quantile_bins, dev, qtile
from equity_benchmark import DRAW_SEEDS, second_angle
from equity_grid1 import ar_moments
from scipy.stats import chi2

OUT = os.path.join(HERE, 'results'); CODE = os.path.dirname(HERE)
P = 4; GRID = np.deg2rad(np.arange(0, 91, 1.0)); KS = (2, 3, 5)


def load():
    vix = pd.read_csv(os.path.join(CODE, 'vix-daily.csv'), parse_dates=['DATE']).set_index('DATE')['CLOSE']
    w = pd.read_csv(os.path.join(CODE, 'weekly.csv'), index_col=0, parse_dates=True)
    y = w.values
    dates = w.index
    lv0 = np.log(vix.resample('W-FRI').last().reindex(dates).values)[0]
    lv = lv0 + np.concatenate([[0.], np.cumsum(y[1:, 1]) / 100.])              # log VIX level consistent with the changes in weekly.csv
    W = vix.resample('W-FRI').last().reindex(dates[P:] - pd.Timedelta(days=7)).values
    assert not np.isnan(lv).any() and not np.isnan(W).any()
    return y, lv, W


def design(y, lv):
    T = len(y)
    X = np.column_stack([np.ones(T - P)] + [y[P - j:T - j] for j in range(1, P + 1)] + [lv[P - 1:T - 1]])
    return X, y[P:]


def fit(y, lv):
    X, Y = design(y, lv); Bc = np.linalg.lstsq(X, Y, rcond=None)[0]
    return Bc, Y - X @ Bc


def regenerate(Bc, y0, lv0, U):
    """y0: first P rows of y; lv0: log-VIX level at time P-1; U: innovations."""
    n = len(U); k = y0.shape[1]; y = np.zeros((P + n, k)); y[:P] = y0
    A = Bc[1:1 + P * k].reshape(P, k, k); g = Bc[1 + P * k]; lv = lv0
    for t in range(P, P + n):
        acc = Bc[0].copy()
        for j in range(1, P + 1):
            acc += y[t - j] @ A[j - 1]
        acc += lv * g
        y[t] = acc + U[t - P]
        lv = lv + y[t, 1] / 100.                # log-VIX level accumulates the change (dvix is in percent)
    return y


def main():
    y, lv, W = load(); Bc, U = fit(y, lv); u = U - U.mean(0); n = len(u)
    z, L = whiten(u); ang = np.array([0., second_angle(L)])
    res = dict(n=n, level_coef=Bc[-1].tolist(), angles_deg=np.rad2deg(ang).tolist())
    res['within_regime_means'] = {f'k{k}': [z[quantile_bins(W, k) == j].mean(0).tolist() for j in range(k)] for k in KS}
    one = np.zeros(n, int)
    E0 = [z @ rot(t) for t in ang]; v0 = [op_inner_cond(E, one, E, one, 1) for E in E0]
    EG = [z @ rot(t) for t in GRID]; vg = np.array([op_inner_cond(E, one, E, one, 1) for E in EG])
    ar0 = {k: ar_moments(z, W, k, GRID) for k in KS}
    het = {k: het_angle(u, W, k)[0] for k in KS}
    B = len(DRAW_SEEDS); D = np.empty((B, 2)); DG = np.empty((B, len(GRID))); AR = {k: np.empty((B, len(GRID), k - 1)) for k in KS}; HB = {k: np.empty(B) for k in KS}
    t0 = time.time()
    for i, s in enumerate(DRAW_SEEDS):
        idx = boot_index(n, 10, np.random.default_rng(s))
        ys = regenerate(Bc, y[:P], lv[P - 1], u[idx])
        lvs = np.concatenate([lv[:P], lv[P - 1] + np.cumsum(ys[P:, 1]) / 100.])
        _, ub = fit(ys, lvs); ub = ub - ub.mean(0); zb, Lb = whiten(ub); Wb = W[idx]
        angb = np.array([0., second_angle(Lb)])
        for j in range(2):
            D[i, j] = dev(zb @ rot(angb[j]), one, E0[j], one, 1, v0[j])
        for j, t in enumerate(GRID):
            DG[i, j] = dev(zb @ rot(t), one, EG[j], one, 1, vg[j])
        for k in KS:
            AR[k][i] = ar_moments(zb, Wb, k, GRID); HB[k][i] = het_angle(ub, Wb, k)[0]
        if i % 50 == 0:
            print(i, round(time.time() - t0), flush=True)
    q = qtile(D.max(1), .95); nr = np.sqrt(v0)
    res['recursive'] = dict(norms=nr.tolist(), q95=q, breakdown=np.sqrt(np.maximum(nr - q, 0)).tolist())
    qg = qtile(DG.max(1), .975); low = []
    for j in range(len(GRID) - 1):
        a, bv = vg[j], vg[j + 1]; c = op_inner_cond(EG[j], one, EG[j + 1], one, 1)
        vv = max(a + bv - 2 * c, 0); t = np.clip((a - c) / vv, 0, 1) if vv > 1e-16 else 0
        m = np.sqrt(max(a + 2 * t * (c - a) + t * t * vv, 0)); low.append(max(0., m - qg - CURV * (GRID[1] - GRID[0]) ** 2 / 8))
    low = np.array(low); res['certified'] = dict(q975=qg, set0=np.arange(90)[low == 0].tolist(), argmin_deg=float(np.rad2deg(GRID[int(np.argmin(vg))])))
    res['het'] = {}; res['ar'] = {}
    for k in KS:
        hb = HB[k]; hb = (hb - het[k] + 45) % 90 - 45 + het[k]
        res['het'][f'k{k}'] = dict(angle=het[k], ci95=np.percentile(hb, [2.5, 97.5]).tolist())
        Wst = np.array([float(ar0[k][j] @ np.linalg.solve(np.cov(AR[k][:, j, :].T).reshape(k - 1, k - 1), ar0[k][j])) for j in range(len(GRID))])
        crit = chi2.ppf(.95, k - 1)
        res['ar'][f'k{k}'] = dict(set=np.rad2deg(GRID)[Wst <= crit].tolist(), min_wald=float(Wst.min()), argmin_deg=float(np.rad2deg(GRID[int(np.argmin(Wst))])),
                                  min_p=float(chi2.sf(Wst.min(), k - 1)), wald=Wst.tolist())
    json.dump(res, open(os.path.join(OUT, 'equity_level.json'), 'w'), indent=1)
    print('level coef', np.round(Bc[-1], 4), 'means', {k: np.round(v, 3).tolist() for k, v in res['within_regime_means'].items()})
    print('recursive', res['recursive'], 'set0', res['certified']['set0'][:1], res['certified']['set0'][-1:])
    print('het', res['het']); print('ar', {k: (v['set'][:1], v['set'][-1:], round(v['min_p'], 3)) for k, v in res['ar'].items()})


if __name__ == '__main__':
    main()
