"""Equity application: proxy-adjusted certificate, bandwidth-set certificate, and co-skewness at
the heteroskedasticity-identified rotation.

Proxy-adjusted operator: bin-weighted within-bin cross-covariance operator, bins are k quantile
bins of lagged VIX (W_t).  The bootstrap resamples (innovation, proxy) pairs in moving blocks,
regenerates and refits the VAR as in equity_benchmark.py (same 399 per-draw seeds), and
recomputes the proxy bins in every draw.

Bandwidth set: operators with Gaussian bandwidths s in {0.5, 1, 2}; since ||C^s|| <= rho_i rho_j / s^2,
the joint .95 critical value of max_{s, ordering} s^2 ||C^{s*} - C^s|| gives the breakdown
max_s sqrt([s^2 ||C^s|| - q]_+).

Run: OPENBLAS_NUM_THREADS=1 python benchmark/equity_adjusted.py [block]
Writes benchmark/results/equity_adjusted_b{block}.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index, het_angle
from acore import op_inner_cond, quantile_bins, dev, qtile
from equity_benchmark import load, second_angle, DRAW_SEEDS
from svarcore import var_fit, var_regenerate, coskew

OUT = os.path.join(HERE, 'results')
KS = (2, 3, 5)
BWS = (0.5, 1.0, 2.0)
GRID = np.deg2rad(np.arange(0, 91, 3.0))
CURV = 8 + 4 * np.sqrt(3)
K_GRID = 3
CS_GRID = np.deg2rad(np.arange(0, 90, 0.5))


def main(block):
    y, p, Bc, U, W = load()
    n = len(U)
    u = U - U.mean(0)
    z, L = whiten(u)
    ang = np.array([0., second_angle(L)])
    bins = {k: quantile_bins(W, k) for k in KS}
    one = np.zeros(n, int)
    E0 = [z @ rot(t) for t in ang]
    # originals
    v_adj = {k: [op_inner_cond(E, bins[k], E, bins[k], k) for E in E0] for k in KS}
    v_bw = {s: [op_inner_cond(E, one, E, one, 1, s) for E in E0] for s in BWS}
    v_bw_adj = {s: [op_inner_cond(E, bins[K_GRID], E, bins[K_GRID], K_GRID, s) for E in E0] for s in BWS}
    EG = [z @ rot(t) for t in GRID]
    v_grid = [op_inner_cond(E, bins[K_GRID], E, bins[K_GRID], K_GRID) for E in EG]
    het = het_angle(u, W, 2)[0]
    cs0_het = coskew(z, [np.deg2rad(het)])[0]
    cs0_grid = coskew(z, CS_GRID)
    B = int(os.environ.get("BLIMIT", len(DRAW_SEEDS)))
    D_adj = {k: np.empty((B, 2)) for k in KS}
    D_bw = {s: np.empty((B, 2)) for s in BWS}
    D_bw_adj = {s: np.empty((B, 2)) for s in BWS}
    D_grid = np.empty((B, len(GRID)))
    CS_het = np.empty((B, 2)); CS_grid = np.empty((B, len(CS_GRID), 2))
    t0 = time.time()
    for b in range(B):
        idx = boot_index(n, block, np.random.default_rng(DRAW_SEEDS[b]))
        ys = var_regenerate(Bc, y[:p], u[idx])
        _, ub = var_fit(ys, p)
        ub = ub - ub.mean(0)
        zb, Lb = whiten(ub)
        Wb = W[idx]
        angb = np.array([0., second_angle(Lb)])
        Eb = [zb @ rot(t) for t in angb]
        for k in KS:
            bb = quantile_bins(Wb, k)
            D_adj[k][b] = [dev(Eb[j], bb, E0[j], bins[k], k, v_adj[k][j]) for j in range(2)]
        onb = np.zeros(n, int)
        bb3 = quantile_bins(Wb, K_GRID)
        for s in BWS:
            D_bw[s][b] = [dev(Eb[j], onb, E0[j], one, 1, v_bw[s][j], s) for j in range(2)]
            D_bw_adj[s][b] = [dev(Eb[j], bb3, E0[j], bins[K_GRID], K_GRID, v_bw_adj[s][j], s) for j in range(2)]
        for j, t in enumerate(GRID):
            D_grid[b, j] = dev(zb @ rot(t), bb3, EG[j], bins[K_GRID], K_GRID, v_grid[j])
        hb = het_angle(ub, Wb, 2)[0]
        CS_het[b] = coskew(zb, [np.deg2rad(hb)])[0]
        CS_grid[b] = coskew(zb, CS_GRID)
        if b % 50 == 0:
            print(block, b, round(time.time() - t0), flush=True)
    res = dict(block=block, B=B, n=n, angles_deg=np.rad2deg(ang).tolist())
    # adjusted recursive tests: joint .95 over the two orderings for each k
    res['adjusted'] = {}
    for k in KS:
        q = qtile(D_adj[k].max(1), .95)
        nr = np.sqrt(v_adj[k])
        res['adjusted'][f'k{k}'] = dict(norms=nr.tolist(), q95=q,
                                        breakdown=np.sqrt(np.maximum(nr - q, 0)).tolist())
    # bandwidth set, raw and adjusted (k = K_GRID)
    for lab, V, D in (('bandwidth_raw', v_bw, D_bw), ('bandwidth_adjusted', v_bw_adj, D_bw_adj)):
        stat = np.max(np.stack([s * s * D[s] for s in BWS], 0), axis=(0, 2))
        q = qtile(stat, .95)
        per = {str(s): dict(norm=np.sqrt(V[s]).tolist(),
                            scaled_lower=[float(max(0., s * s * np.sqrt(V[s][j]) - q)) for j in range(2)])
               for s in BWS}
        brk = [float(np.sqrt(max(per[str(s)]['scaled_lower'][j] for s in BWS))) for j in range(2)]
        # single-bandwidth versions with their own critical values for comparison
        single = {str(s): np.sqrt(np.maximum(np.sqrt(V[s]) - qtile(D[s].max(1), .95), 0)).tolist() for s in BWS}
        res[lab] = dict(q95_joint_scaled=q, per_bandwidth=per, breakdown_joint=brk, breakdown_single=single)
    # adjusted certified set (k = K_GRID), .975 grid radius
    qg = qtile(D_grid.max(1), .975)
    low = []
    for j in range(len(GRID) - 1):
        a, bv = v_grid[j], v_grid[j + 1]
        c = op_inner_cond(EG[j], bins[K_GRID], EG[j + 1], bins[K_GRID], K_GRID)
        vv = max(a + bv - 2 * c, 0)
        t = np.clip((a - c) / vv, 0, 1) if vv > 1e-16 else 0
        m = np.sqrt(max(a + 2 * t * (c - a) + t * t * vv, 0))
        low.append(max(0., m - qg - CURV * (GRID[j + 1] - GRID[j]) ** 2 / 8))
    low = np.array(low)
    res['adjusted_grid'] = dict(k=K_GRID, q975=qg, grid_norms=np.sqrt(v_grid).tolist(), cell_lower_sqrt=np.sqrt(low).tolist(),
                                argmin_deg=float(np.rad2deg(GRID[int(np.argmin(v_grid))])))
    # co-skewness at the heteroskedasticity rotation (re-estimated in every draw) and over a grid
    Vh = np.cov(CS_het.T)
    res['coskew_het'] = dict(angle=het, moments=cs0_het.tolist(), wald=float(cs0_het @ np.linalg.solve(Vh, cs0_het)))
    wg = np.array([cs0_grid[j] @ np.linalg.solve(np.cov(CS_grid[:, j, :].T), cs0_grid[j]) for j in range(len(CS_GRID))])
    res['coskew_grid'] = dict(deg=np.rad2deg(CS_GRID).tolist(), wald=wg.tolist(),
                              min_wald=float(wg.min()), argmin_deg=float(np.rad2deg(CS_GRID[wg.argmin()])))
    res['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f'equity_adjusted_b{block}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('adjusted_grid', 'coskew_grid')}, indent=0)[:3000])
    print('grid argmin', res['adjusted_grid']['argmin_deg'], 'coskew min', res['coskew_grid']['min_wald'], res['coskew_grid']['argmin_deg'])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
