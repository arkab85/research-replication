"""FOMC application: independence-based decomposition of high-frequency monetary-policy
surprises, its certificate, and the pre-announcement-VIX exposure benchmark.

Data: Jarocinski and Karadi (2020) surprises, updated by the authors
(github.com/marekjarocinski/jkshocks_update_fed, file shocks_fed_jk_t.csv):
pc1 = first principal component of 30-minute interest-rate-futures surprises,
SP500 = 30-minute S&P 500 surprise (percent).  Announcements with a missing S&P 500
surprise are dropped (5 of 330).  The proxy W is the VIX close on the last trading day
strictly before the announcement date (code/vix-daily.csv).

Run:  OPENBLAS_NUM_THREADS=1 python benchmark/fomc_application.py
Writes benchmark/results/fomc.json and fomc_*.npz.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import (np, pd, rot, whiten, operator_radius, cell_lower, rho_proxy_boot, isotropy_test,
                   het_angle, boot_index, sq_norms, op_inner_fast, CURV)
from whiten_region import studentized_region, entry_bounds

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
GRID = np.deg2rad(np.arange(0, 91, 1.0))          # one-degree cells, endpoints included
B_OP, B_WH, B_BENCH = 999, 1999, 999
SEED = 2026092701


def load():
    d = pd.read_csv(os.path.join(HERE, 'data', 'shocks_fed_jk_t.csv'), parse_dates=['start'])
    d = d.dropna(subset=['pc1', 'SP500']).reset_index(drop=True)
    vix = pd.read_csv(os.path.join(os.path.dirname(HERE), 'vix-daily.csv'), parse_dates=['DATE']).set_index('DATE')['CLOSE']
    pos = vix.index.searchsorted(d.start.dt.normalize(), side='left') - 1
    d['vix_prev'] = vix.values[pos]
    d['vix_date'] = vix.index[pos]
    assert (d.vix_date < d.start.dt.normalize()).all()
    return d


def stock_first_angle(L):
    """Angle at which the first candidate shock has no stock response (A_21 = 0)."""
    return float(np.arctan2(-L[1, 0], L[1, 1]) % (np.pi / 2))


def cell_conclusions(reg, grid):
    """Per cell: can some (L, theta) give a positively co-moving column (information-type
    shock)?  Largest information share attainable, and can both columns be non-positive?"""
    rows = []
    for a, b in zip(grid[:-1], grid[1:]):
        lo, hi = entry_bounds(reg, a, b)
        smax, pos_any, nonpos_all = 0., False, True
        for k in range(2):
            pos = (hi[0, k] > 0 and hi[1, k] > 0) or (lo[0, k] < 0 and lo[1, k] < 0)
            nonpos = (lo[0, k] <= 0 and hi[1, k] >= 0) or (hi[0, k] >= 0 and lo[1, k] <= 0)
            share = np.cos(a) ** 2 if k == 0 else np.sin(b) ** 2      # A_1k^2 / Sigma_11 over the cell
            if pos:
                pos_any = True; smax = max(smax, share)
            nonpos_all = nonpos_all and nonpos
        rows.append((smax, pos_any, nonpos_all))
    return np.array(rows, dtype=float)


def recursive_test(u, B, block, rng):
    """Operator tests at the two recursive orderings (policy first: theta = 0; stock first:
    theta_c(L)), angles recomputed in every draw, joint .95 max-norm critical value."""
    z, L = whiten(u)
    ang = np.array([0., stock_first_angle(L)])
    orig = np.array([op_inner_fast(z @ rot(t), z @ rot(t)) for t in ang])
    n = len(u); D = np.empty((B, 2))
    for b in range(B):
        ub = u[boot_index(n, block, rng)]; zb, Lb = whiten(ub - ub.mean(0))
        angb = np.array([0., stock_first_angle(Lb)])
        for j in range(2):
            Eb = zb @ rot(angb[j]); E = z @ rot(ang[j])
            D[b, j] = np.sqrt(max(0., op_inner_fast(Eb, Eb) + orig[j] - 2 * op_inner_fast(Eb, E)))
    mx = np.sort(D.max(1)); q = float(mx[min(int(np.ceil(.95 * (B + 1))), B) - 1])
    norms = np.sqrt(orig)
    return dict(angles_deg=np.rad2deg(ang).tolist(), norms=norms.tolist(), q95=q,
                lower_budget=np.sqrt(np.maximum(norms - q, 0)).tolist())


def certificate(u_raw, label, block, rng):
    u = u_raw - u_raw.mean(0)
    z, L = whiten(u)
    t0 = time.time()
    q, orig, _ = operator_radius(u, GRID, B_OP, block, rng)
    low, seg = cell_lower(z, GRID, orig, q)
    reg = studentized_region(u_raw, 0, [0, 1], block, B_WH, rng)
    cc = cell_conclusions(reg, GRID)
    smax, pos_any, nonpos_all = cc[:, 0], cc[:, 1] > 0, cc[:, 2] > 0
    out = dict(label=label, n=len(u), block=block, B_operator=B_OP, B_whitening=B_WH,
               grid_degrees=1.0, curvature_allowance=float(CURV * (GRID[1] - GRID[0]) ** 2 / 8),
               q_operator_975=q, whitening_critical_value=reg['c'],
               L=L.tolist(), stock_first_angle_deg=np.rad2deg(stock_first_angle(L)),
               min_norm=float(np.sqrt(orig.min())), argmin_deg=float(np.rad2deg(GRID[np.argmin(orig)])),
               cell_lower_sqrt=np.sqrt(low).tolist())
    sets = {}
    for rho in (0, .025, .05, .075, .1, .15, .2, .25, .3):
        keep = low <= rho * rho
        sets[str(rho)] = dict(cell_share=float(keep.mean()),
                              info_share_max=float(smax[keep].max()) if keep.any() else None,
                              info_type_shock_possible=bool(pos_any[keep].any()),
                              no_info_type_shock_possible=bool(nonpos_all[keep].any()))
    out['sets'] = sets

    def breakdown(bad):
        return float(np.sqrt(low[bad].min())) if bad.any() else None
    out['breakdown'] = {
        'info_share_le_0.5': breakdown(smax > .5),
        'info_share_le_0.25': breakdown(smax > .25),
        'info_type_shock_exists': breakdown(nonpos_all),
        'no_info_type_shock': breakdown(pos_any)}
    out['seconds'] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(OUT, f'fomc_{label}_b{block}.npz'), grid=GRID, norms=np.sqrt(orig),
                        cell_lower=low, segment_min=seg, cells=cc, whitening_T=reg['T'], l=reg['l'], V=reg['V'])
    return out


def main():
    d = load()
    u_raw = d[['pc1', 'SP500']].values
    W = d.vix_prev.values
    res = dict(n=len(d), first=str(d.start.min()), last=str(d.start.max()),
               vix_prev=dict(mean=float(W.mean()), sd=float(W.std()), min=float(W.min()), max=float(W.max())),
               corr=float(np.corrcoef(u_raw.T)[0, 1]),
               jk_info_share=float(d.CBI_median.var() / d.pc1.var()),
               jk_info_share_pm=float(d.CBI_pm.var() / d.pc1.var()))
    z, _ = whiten(u_raw - u_raw.mean(0))
    res['kurtosis_whitened'] = (z ** 4).mean(0).tolist()
    nblk = int(np.ceil(np.sqrt(len(d))))
    rng = np.random.default_rng(SEED)
    res['raw'] = {}
    res['rescaled'] = {}
    for block in (1, nblk):
        res['raw'][f'block{block}'] = certificate(u_raw, 'raw', block, rng)
        print('raw', block, json.dumps(res['raw'][f'block{block}']['breakdown']), flush=True)
        res['raw'][f'block{block}']['recursive'] = recursive_test(u_raw - u_raw.mean(0), B_OP, block, rng)
    ut = u_raw / W[:, None]                      # rescaling by a predetermined scalar leaves A0 unchanged
    for block in (1, nblk):
        res['rescaled'][f'block{block}'] = certificate(ut, 'rescaled', block, rng)
        print('rescaled', block, json.dumps(res['rescaled'][f'block{block}']['breakdown']), flush=True)
        res['rescaled'][f'block{block}']['recursive'] = recursive_test(ut - ut.mean(0), B_OP, block, rng)
    # exposure benchmark and isotropy, raw and rescaled
    for lab, uu in (('raw', u_raw), ('rescaled', ut)):
        uu = uu - uu.mean(0)
        res[lab]['benchmark'] = {f'k{k}_block{blk}': rho_proxy_boot(uu, W, k, B_BENCH, blk, rng)
                                 for k in (2, 3, 5) for blk in (1, nblk)}
        res[lab]['isotropy'] = {f'k{k}_block{blk}': isotropy_test(uu, W, k, B_BENCH, blk, rng)
                                for k in (2, 3, 5) for blk in (1, nblk)}
        a, L, D = het_angle(uu, W, 3)
        res[lab]['het_angle_k3'] = dict(angle=a, bin_variances=D.tolist())
        print(lab, 'benchmark', {k: round(v['rho'], 3) for k, v in res[lab]['benchmark'].items()},
              'lower', {k: round(v['lower'], 3) for k, v in res[lab]['benchmark'].items()}, flush=True)
        print(lab, 'isotropy', {k: round(v['p'], 3) for k, v in res[lab]['isotropy'].items()}, flush=True)
    res['seed'] = SEED
    with open(os.path.join(OUT, 'fomc.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
