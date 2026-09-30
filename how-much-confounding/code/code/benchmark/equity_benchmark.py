"""Equity application: VIX exposure floor, isotropy test, heteroskedasticity-identified
rotation, and the recursive-ordering tests after rescaling by lagged VIX.

The proxy W_t is the VIX close on the Friday ending week t-1 (predetermined).
Rescaling divides the VAR innovations of week t by W_t.  The rescaled bootstrap
holds the proxy path fixed: resampled rescaled innovations are multiplied back by
the original calendar's W_t, the VAR is regenerated and refitted, and the refitted
innovations are divided by W_t again.

Run: OPENBLAS_NUM_THREADS=1 python benchmark/equity_benchmark.py
Writes benchmark/results/equity_benchmark.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import (np, pd, rot, whiten, boot_index, rho_proxy_boot, isotropy_test, het_angle, op_inner_fast,
                   sq_norms, cell_lower, CURV)
from svarcore import var_fit, var_regenerate, coskew
from scipy.stats import chi2

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
CODE = os.path.dirname(HERE)
B_REC, B_BENCH, B_HET = 399, 999, 499
SEED = 2026092702
GRID = np.deg2rad(np.arange(0, 91, 3.0))
# The recursive tests and certified cells use the 399 per-draw seeds of the main equity run
# (certified_revision.py), so the raw analysis reproduces its published numbers exactly and the
# rescaled analysis uses the same draws.
DRAW_SEEDS = np.load(os.path.join(CODE, 'revision_results', 'equity.npz'))['seeds']


def load():
    vix = pd.read_csv(os.path.join(CODE, 'vix-daily.csv'), parse_dates=['DATE']).set_index('DATE')['CLOSE']
    w = pd.read_csv(os.path.join(CODE, 'weekly.csv'), index_col=0, parse_dates=True)
    y = w.values; p = 4
    Bc, U = var_fit(y, p)
    dates = w.index[p:]
    W = vix.resample('W-FRI').last().reindex(dates - pd.Timedelta(days=7)).values
    assert not np.isnan(W).any()
    return y, p, Bc, U, W


def second_angle(L):
    return float(np.arctan2(-L[1, 0], L[1, 1]) % (np.pi / 2))


def recursive_and_grid(y, p, Bc, U, scale, block, seeds):
    """Recursive-ordering operator tests, co-skewness Wald statistics and the certified rotation
    cells, fully refitted; one rng per draw seeded from `seeds` as in certified_revision.draw.
    scale: per-date positive scalars (ones for the raw analysis)."""
    n = len(U); B = len(seeds)
    ut = (U - U.mean(0)) / scale[:, None]
    ut = ut - ut.mean(0)
    z, L = whiten(ut)
    ang = np.array([0., second_angle(L)])
    orig_r = np.array([op_inner_fast(z @ rot(t), z @ rot(t)) for t in ang])
    orig_g = sq_norms(z, GRID)
    cs0 = coskew(z, ang)
    D = np.empty((B, 2)); DG = np.empty((B, len(GRID))); CS = np.empty((B, 2, 2))
    for b in range(B):
        idx = boot_index(n, block, np.random.default_rng(seeds[b]))
        innov = ut[idx] * scale[:, None]                       # calendar-fixed rescaling
        ys = var_regenerate(Bc, y[:p], innov)
        _, ub = var_fit(ys, p)
        ub = (ub - ub.mean(0)) / scale[:, None]; ub = ub - ub.mean(0)
        zb, Lb = whiten(ub)
        angb = np.array([0., second_angle(Lb)])
        CS[b] = coskew(zb, angb)
        for j in range(2):
            Eb = zb @ rot(angb[j]); E = z @ rot(ang[j])
            D[b, j] = np.sqrt(max(0., op_inner_fast(Eb, Eb) + orig_r[j] - 2 * op_inner_fast(Eb, E)))
        for j, (t, v) in enumerate(zip(GRID, orig_g)):
            Eb = zb @ rot(t)
            DG[b, j] = np.sqrt(max(0., op_inner_fast(Eb, Eb) + v - 2 * op_inner_fast(Eb, z @ rot(t))))
    qr = float(np.sort(D.max(1))[min(int(np.ceil(.95 * (B + 1))), B) - 1])
    qg = float(np.sort(DG.max(1))[min(int(np.ceil(.975 * (B + 1))), B) - 1])
    low, _ = cell_lower(z, GRID, orig_g, qg)
    norms = np.sqrt(orig_r)
    W = [float(cs0[j] @ np.linalg.solve(np.cov(CS[:, j, :].T), cs0[j])) for j in range(2)]
    return dict(angles_deg=np.rad2deg(ang).tolist(), norms=norms.tolist(), q95=qr,
                lower_budget=np.sqrt(np.maximum(norms - qr, 0)).tolist(), q_grid_975=qg,
                coskew_wald=W, coskew_p_asymptotic=chi2.sf(W, 2).tolist(),
                grid_norms=np.sqrt(orig_g).tolist(), cell_lower_sqrt=np.sqrt(low).tolist(),
                argmin_deg=float(np.rad2deg(GRID[np.argmin(orig_g)])), L=L.tolist())


def het_boot(y, p, Bc, U, W, scale, k, block, B, rng):
    n = len(U)
    ut = (U - U.mean(0)) / scale[:, None]; ut = ut - ut.mean(0)
    a, L, Dv = het_angle(ut, W, k)
    ab = np.empty(B)
    for b in range(B):
        idx = boot_index(n, block, rng)
        ys = var_regenerate(Bc, y[:p], ut[idx] * scale[:, None])
        _, ub = var_fit(ys, p)
        ub = (ub - ub.mean(0)) / scale[:, None]; ub = ub - ub.mean(0)
        ab[b] = het_angle(ub, W[idx], k)[0]
    ab = (ab - a + 45) % 90 - 45 + a            # unwrap around the estimate
    A = L @ rot(np.deg2rad(a))
    return dict(k=k, angle=a, ci95=np.percentile(ab, [2.5, 97.5]).tolist(), ci90=np.percentile(ab, [5, 95]).tolist(),
                bin_variances=Dv.tolist(), impact=A.tolist())


def main():
    y, p, Bc, U, W = load()
    n = len(U); nblk = int(np.ceil(np.sqrt(n)))
    rng = np.random.default_rng(SEED)
    res = dict(n=n, seed=SEED, vix_prev=dict(mean=float(W.mean()), sd=float(W.std())))
    for lab, scale in (('raw', np.ones(n)), ('rescaled', W)):
        t0 = time.time()
        r = {}
        for block in (10, nblk):
            r[f'block{block}'] = recursive_and_grid(y, p, Bc, U, scale, block, DRAW_SEEDS)
            print(lab, block, 'recursive lower', np.round(r[f'block{block}']['lower_budget'], 3),
                  'wald', np.round(r[f'block{block}']['coskew_wald'], 2), flush=True)
        ut = (U - U.mean(0)) / scale[:, None]; ut = ut - ut.mean(0)
        r['benchmark'] = {f'k{k}_block{blk}': rho_proxy_boot(ut, W, k, B_BENCH, blk, rng) for k in (2, 3, 5) for blk in (10, nblk)}
        r['isotropy'] = {f'k{k}_block{blk}': isotropy_test(ut, W, k, B_BENCH, blk, rng) for k in (2, 3, 5) for blk in (10, nblk)}
        r['het'] = {f'k{k}': het_boot(y, p, Bc, U, W, scale, k, 10, B_HET, rng) for k in (2, 3, 5)}
        r['kurtosis_whitened'] = (whiten(ut)[0] ** 4).mean(0).tolist()
        r['seconds'] = round(time.time() - t0, 1)
        print(lab, 'floor', {k: round(v['rho'], 3) for k, v in r['benchmark'].items()},
              'lower', {k: round(v['lower'], 3) for k, v in r['benchmark'].items()}, flush=True)
        print(lab, 'isotropy p', {k: round(v['p'], 3) for k, v in r['isotropy'].items()}, flush=True)
        print(lab, 'het', {k: (round(v['angle'], 1), np.round(v['ci95'], 1).tolist()) for k, v in r['het'].items()}, flush=True)
        res[lab] = r
    with open(os.path.join(OUT, 'equity_benchmark.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
