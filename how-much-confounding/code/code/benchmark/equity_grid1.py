"""Equity application: certified rotation sets on one-degree cells (raw and conditioned on three
lagged-VIX regimes), matching the resolution used for the monetary-policy surprises, and an
Anderson--Rubin-type confidence set for the heteroskedasticity-identified rotation that does not
presume strong identification.

Certified sets: 399 refitted draws with the seeds of the main equity run, .975 grid radius over the
91 one-degree grid points, curvature allowance M/8 per cell.

AR set: for each angle theta0 on a one-degree grid and each k in {2, 3, 5}, the moment vector is the
within-regime means of e_1(theta0) e_2(theta0) for regimes 1..k-1 (the last is implied by whitening);
its covariance is estimated from the refitted draws (regimes recomputed in every draw), and theta0 is
retained when the Wald statistic is below the chi-square(k-1) .95 quantile.  Under the latent-state
model E[e_1 e_2 | W] = 0 at the true rotation, so the set has asymptotic coverage .95 whatever the
strength of identification (Lewis, 2022, Section 4, for the same construction with regime dummies).

Run: OPENBLAS_NUM_THREADS=1 python benchmark/equity_grid1.py BLOCK
Writes results/equity_grid1_b{BLOCK}.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index, CURV
from acore import op_inner_cond, quantile_bins, dev, qtile
from equity_benchmark import load, DRAW_SEEDS
from svarcore import var_fit, var_regenerate
from scipy.stats import chi2

OUT = os.path.join(HERE, 'results')
GRID = np.deg2rad(np.arange(0, 91, 1.0))
K_GRID = 3
KS = (2, 3, 5)


def ar_moments(z, W, k, grid):
    b = quantile_bins(W, k)
    out = np.empty((len(grid), k - 1))
    for j, t in enumerate(grid):
        e = z @ rot(t); prod = e[:, 0] * e[:, 1]
        out[j] = [prod[b == r].mean() for r in range(k - 1)]
    return out


def cells(v, EG, bins, k, q):
    low = []
    for j in range(len(GRID) - 1):
        a, bv = v[j], v[j + 1]
        c = op_inner_cond(EG[j], bins, EG[j + 1], bins, k)
        vv = max(a + bv - 2 * c, 0)
        t = np.clip((a - c) / vv, 0, 1) if vv > 1e-16 else 0
        m = np.sqrt(max(a + 2 * t * (c - a) + t * t * vv, 0))
        low.append(max(0., m - q - CURV * (GRID[j + 1] - GRID[j]) ** 2 / 8))
    return np.array(low)


def main(block):
    y, p, Bc, U, W = load(); u = U - U.mean(0); n = len(u)
    z, L = whiten(u)
    one = np.zeros(n, int); b3 = quantile_bins(W, K_GRID)
    EG = [z @ rot(t) for t in GRID]
    v_raw = np.array([op_inner_cond(E, one, E, one, 1) for E in EG])
    v_cnd = np.array([op_inner_cond(E, b3, E, b3, K_GRID) for E in EG])
    ar0 = {k: ar_moments(z, W, k, GRID) for k in KS}
    B = len(DRAW_SEEDS)
    Dr = np.empty((B, len(GRID))); Dc = np.empty((B, len(GRID)))
    AR = {k: np.empty((B, len(GRID), k - 1)) for k in KS}
    t0 = time.time()
    for i, s in enumerate(DRAW_SEEDS):
        idx = boot_index(n, block, np.random.default_rng(s))
        ys = var_regenerate(Bc, y[:p], u[idx]); _, ub = var_fit(ys, p); ub = ub - ub.mean(0)
        zb, _ = whiten(ub); Wb = W[idx]; bb = quantile_bins(Wb, K_GRID)
        for j, t in enumerate(GRID):
            Eb = zb @ rot(t)
            Dr[i, j] = dev(Eb, one, EG[j], one, 1, v_raw[j])
            Dc[i, j] = dev(Eb, bb, EG[j], b3, K_GRID, v_cnd[j])
        for k in KS:
            AR[k][i] = ar_moments(zb, Wb, k, GRID)
        if i % 50 == 0:
            print(block, i, round(time.time() - t0), flush=True)
    qr, qc = qtile(Dr.max(1), .975), qtile(Dc.max(1), .975)
    low_r, low_c = cells(v_raw, EG, one, 1, qr), cells(v_cnd, EG, b3, K_GRID, qc)
    deg = np.arange(0, 90)
    res = dict(block=block, B=B, q_raw=qr, q_cond=qc,
               raw=dict(cell_lower_sqrt=np.sqrt(low_r).tolist(), set0=deg[low_r == 0].tolist()),
               cond=dict(cell_lower_sqrt=np.sqrt(low_c).tolist(), set0=deg[low_c == 0].tolist()),
               argmin_deg=float(np.rad2deg(GRID[int(np.argmin(v_raw))])), ar={})
    for k in KS:
        keep = []
        W_stat = np.empty(len(GRID))
        for j in range(len(GRID)):
            V = np.cov(AR[k][:, j, :].T).reshape(k - 1, k - 1)
            m = ar0[k][j]
            W_stat[j] = float(m @ np.linalg.solve(V, m))
        crit = chi2.ppf(.95, k - 1)
        res['ar'][f'k{k}'] = dict(wald=W_stat.tolist(), crit=crit, set=np.rad2deg(GRID)[W_stat <= crit].tolist(),
                                  argmin_deg=float(np.rad2deg(GRID[int(np.argmin(W_stat))])))
    json.dump(res, open(os.path.join(OUT, f'equity_grid1_b{block}.json'), 'w'), indent=1)
    print(block, 'raw set0', res['raw']['set0'][:1], res['raw']['set0'][-1:], len(res['raw']['set0']),
          'cond set0', res['cond']['set0'][:1], res['cond']['set0'][-1:], len(res['cond']['set0']), flush=True)
    for k in KS:
        s = res['ar'][f'k{k}']['set']; print(block, 'AR', k, (min(s), max(s), len(s)) if s else None, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
