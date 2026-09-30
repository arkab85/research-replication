"""FOMC application: proxy-adjusted certificate and bandwidth-set certificate.

The proxy-adjusted operator conditions on k quantile bins of pre-announcement VIX (see acore.py).
The bootstrap resamples (surprise, proxy) pairs, iid or in moving blocks, and recomputes the
whitening and the proxy bins in every draw.  Economic conclusions reuse the per-cell projections
of the studentized whitening region stored by fomc_application.py (fomc_raw_b*.npz).

Run: OPENBLAS_NUM_THREADS=1 python benchmark/fomc_adjusted.py
Writes benchmark/results/fomc_adjusted.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index
from acore import op_inner_cond, quantile_bins, dev, qtile
from fomc_application import load, stock_first_angle

OUT = os.path.join(HERE, 'results')
KS = (2, 3, 5)
BWS = (0.5, 1.0, 2.0)
GRID = np.deg2rad(np.arange(0, 91, 1.0))
CURV = 8 + 4 * np.sqrt(3)
K_GRID = 3
B = 999
SEED = 2026092703


def run(u, W, block, rng):
    n = len(u)
    z, L = whiten(u)
    ang = np.array([0., stock_first_angle(L)])
    bins = {k: quantile_bins(W, k) for k in KS}
    one = np.zeros(n, int)
    E0 = [z @ rot(t) for t in ang]
    v_adj = {k: [op_inner_cond(E, bins[k], E, bins[k], k) for E in E0] for k in KS}
    v_bw = {s: [op_inner_cond(E, one, E, one, 1, s) for E in E0] for s in BWS}
    v_bw_adj = {s: [op_inner_cond(E, bins[K_GRID], E, bins[K_GRID], K_GRID, s) for E in E0] for s in BWS}
    EG = [z @ rot(t) for t in GRID]
    v_grid = {k: [op_inner_cond(E, bins[k], E, bins[k], k) for E in EG] for k in (K_GRID,)}
    D_adj = {k: np.empty((B, 2)) for k in KS}
    D_bw = {s: np.empty((B, 2)) for s in BWS}
    D_bw_adj = {s: np.empty((B, 2)) for s in BWS}
    D_grid = {k: np.empty((B, len(GRID))) for k in (K_GRID,)}
    for b in range(B):
        idx = boot_index(n, block, rng)
        ub = u[idx] - u[idx].mean(0)
        zb, Lb = whiten(ub)
        Wb = W[idx]
        angb = np.array([0., stock_first_angle(Lb)])
        Eb = [zb @ rot(t) for t in angb]
        bb = {k: quantile_bins(Wb, k) for k in KS}
        onb = np.zeros(n, int)
        for k in KS:
            D_adj[k][b] = [dev(Eb[j], bb[k], E0[j], bins[k], k, v_adj[k][j]) for j in range(2)]
        for s in BWS:
            D_bw[s][b] = [dev(Eb[j], onb, E0[j], one, 1, v_bw[s][j], s) for j in range(2)]
            D_bw_adj[s][b] = [dev(Eb[j], bb[K_GRID], E0[j], bins[K_GRID], K_GRID, v_bw_adj[s][j], s) for j in range(2)]
        EGb = [zb @ rot(t) for t in GRID]
        for k in (K_GRID,):
            for j in range(len(GRID)):
                D_grid[k][b, j] = dev(EGb[j], bb[k], EG[j], bins[k], k, v_grid[k][j])
    res = dict(block=block, B=B, n=n, angles_deg=np.rad2deg(ang).tolist())
    res['adjusted'] = {}
    for k in KS:
        q = qtile(D_adj[k].max(1), .95)
        nr = np.sqrt(v_adj[k])
        res['adjusted'][f'k{k}'] = dict(norms=nr.tolist(), q95=q, breakdown=np.sqrt(np.maximum(nr - q, 0)).tolist())
    for lab, V, D in (('bandwidth_raw', v_bw, D_bw), ('bandwidth_adjusted', v_bw_adj, D_bw_adj)):
        stat = np.max(np.stack([s * s * D[s] for s in BWS], 0), axis=(0, 2))
        q = qtile(stat, .95)
        per = {str(s): dict(norm=np.sqrt(V[s]).tolist(),
                            scaled_lower=[float(max(0., s * s * np.sqrt(V[s][j]) - q)) for j in range(2)]) for s in BWS}
        brk = [float(np.sqrt(max(per[str(s)]['scaled_lower'][j] for s in BWS))) for j in range(2)]
        single = {str(s): np.sqrt(np.maximum(np.sqrt(V[s]) - qtile(D[s].max(1), .95), 0)).tolist() for s in BWS}
        res[lab] = dict(q95_joint_scaled=q, per_bandwidth=per, breakdown_joint=brk, breakdown_single=single)
    # adjusted certified cells and economic conclusions
    cc = np.load(os.path.join(OUT, f'fomc_raw_b{block}.npz'))['cells']
    smax, pos_any, nonpos_all = cc[:, 0], cc[:, 1] > 0, cc[:, 2] > 0
    res['adjusted_grid'] = {}
    for k in (K_GRID,):
        qg = qtile(D_grid[k].max(1), .975)
        low = []
        for j in range(len(GRID) - 1):
            a, bv = v_grid[k][j], v_grid[k][j + 1]
            c = op_inner_cond(EG[j], bins[k], EG[j + 1], bins[k], k)
            vv = max(a + bv - 2 * c, 0)
            t = np.clip((a - c) / vv, 0, 1) if vv > 1e-16 else 0
            m = np.sqrt(max(a + 2 * t * (c - a) + t * t * vv, 0))
            low.append(max(0., m - qg - CURV * (GRID[j + 1] - GRID[j]) ** 2 / 8))
        low = np.array(low)
        sets = {}
        for rho in (0, .05, .1, .15, .2):
            keep = low <= rho * rho
            deg = np.rad2deg(GRID[:-1])[keep]
            sets[str(rho)] = dict(cell_share=float(keep.mean()),
                                  info_share_max=float(smax[keep].max()) if keep.any() else None,
                                  info_type_shock_possible=bool(pos_any[keep].any()),
                                  no_info_type_shock_possible=bool(nonpos_all[keep].any()),
                                  cells_deg=deg.tolist())

        def brk(bad):
            return float(np.sqrt(low[bad].min())) if bad.any() else None
        res['adjusted_grid'][f'k{k}'] = dict(q975=qg, grid_norms=np.sqrt(v_grid[k]).tolist(), cell_lower_sqrt=np.sqrt(low).tolist(),
                                             argmin_deg=float(np.rad2deg(GRID[int(np.argmin(v_grid[k]))])), sets=sets,
                                             breakdown={'info_share_le_0.5': brk(smax > .5), 'info_share_le_0.25': brk(smax > .25),
                                                        'info_type_shock_exists': brk(nonpos_all), 'no_info_type_shock': brk(pos_any)})
    return res


def main():
    d = load()
    u = d[['pc1', 'SP500']].values
    u = u - u.mean(0)
    W = d.vix_prev.values
    rng = np.random.default_rng(SEED)
    out = dict(seed=SEED, n=len(u))
    nblk = int(np.ceil(np.sqrt(len(u))))
    for block in (1, nblk):
        t0 = time.time()
        out[f'block{block}'] = r = run(u, W, block, rng)
        r['seconds'] = round(time.time() - t0, 1)
        print(block, 'adjusted', {k: np.round(v['breakdown'], 3).tolist() for k, v in r['adjusted'].items()},
              'bw raw', np.round(r['bandwidth_raw']['breakdown_joint'], 3), 'bw adj', np.round(r['bandwidth_adjusted']['breakdown_joint'], 3), flush=True)
        for k, g in r['adjusted_grid'].items():
            print(block, k, 'argmin', g['argmin_deg'], 'set0', g['sets']['0']['cells_deg'][:1], g['sets']['0']['cells_deg'][-1:],
                  'share', g['sets']['0']['cell_share'], 'brk', g['breakdown'], flush=True)
    with open(os.path.join(OUT, 'fomc_adjusted.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
