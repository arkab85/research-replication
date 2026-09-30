"""Embedding-unit exposure floors and the floor-restricted certificate.

For a rotation theta and shock i, the embedding floor is
    v_i^W(theta) = E|| E[Phi(e_i(theta)) | W_k] - E Phi(e_i(theta)) ||^2
                 = sum_b p_b ||mu_b||^2 - ||mu||^2,
estimated without diagonal terms (U-statistics within bins and overall).  Under the latent
common-state model nu_i^2 >= v_i^W(Q_0), and sqrt(v_i^W(theta)) is 1-Lipschitz in theta for
whitened z and unit bandwidth, so a cell [a, b] of width D has per-shock floor
    [(nu_lo(a) + nu_lo(b) - D) / 2]_+.
The pair floor is f = (nu_1 nu_2)^{1/2}.

Bands: one-sided basic bootstrap, v >= v_hat - q with q the .975 quantile of
max_{theta, i} (v* - v_hat), jointly over the grid and both shocks, separately for each k.  At the
recursive orderings the angle is recomputed in every draw and a separate joint .975 value is used;
the recursive operator test is recomputed at level .975 in the same draws, so that operator and
floor statements hold jointly with probability at least .95.

Run: OPENBLAS_NUM_THREADS=1 python benchmark/nu_floors.py fomc|equity
Writes benchmark/results/nu_floors_{app}.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index
from acore import gram, quantile_bins, qtile
from fastop import op_inner_fast

OUT = os.path.join(HERE, 'results')
KS = (2, 3, 5)


def offmean(K):
    n = K.shape[0]
    return (K.sum() - np.trace(K)) / (n * (n - 1))


def vW(e, b, k):
    K = gram(e, e)
    n = len(e); p = np.bincount(b, minlength=k) / n
    return sum(p[j] * offmean(K[np.ix_(b == j, b == j)]) for j in range(k)) - offmean(K)


def vw_all(z, W, angles, k):
    b = quantile_bins(W, k)
    out = np.empty((len(angles), 2))
    for j, t in enumerate(angles):
        E = z @ rot(t)
        out[j] = [vW(E[:, 0], b, k), vW(E[:, 1], b, k)]
    return out


def cell_pair_floor(vlo, grid):
    nu = np.sqrt(np.maximum(vlo, 0))                     # len(grid) x 2
    D = np.diff(grid)
    c = np.maximum((nu[:-1] + nu[1:] - D[:, None]) / 2, 0)
    return np.sqrt(c[:, 0] * c[:, 1])


def summarize(grid, vhat, D, orders_hat, Dord, op_rec, cell_lower_sq, conclusions=None):
    """vhat: grid x 2; D: B x grid x 2 (v* - vhat); orders: 2 x 2 at ordering angles."""
    q = qtile(D.max(axis=(1, 2)), .975)
    qo = qtile(Dord.max(axis=(1, 2)), .975)
    vlo = vhat - q
    f_cell = cell_pair_floor(vlo, grid)
    f_point = np.sqrt(np.sqrt(np.maximum(vhat[:, 0], 0) * np.maximum(vhat[:, 1], 0)))
    ord_lo = np.sqrt(np.sqrt(np.maximum(orders_hat - qo, 0).prod(1)))
    ord_pt = np.sqrt(np.sqrt(np.maximum(orders_hat, 0).prod(1)))
    g = np.maximum(cell_lower_sq, f_cell ** 2)             # floor-restricted cell bound (squared units)
    res = dict(q_grid=q, q_orders=qo,
               pair_floor_point=f_point.tolist(), pair_floor_cell_lower=f_cell.tolist(),
               orders_pair_floor_point=ord_pt.tolist(), orders_pair_floor_lower=ord_lo.tolist(),
               nu_min_lower=float(np.sqrt(g.min())), nu_min_cell_deg=float(np.rad2deg(grid[int(np.argmin(g))])),
               floor_min_over_cells=float(f_cell.min()), floor_max_over_cells=float(f_cell.max()))
    # recursive orderings: certificate breakdown and floor-restricted breakdown at joint .95
    res['recursive'] = dict(norms=op_rec['norms'], q975=op_rec['q975'],
                            breakdown=op_rec['breakdown'],
                            breakdown_floor_restricted=[float(max(op_rec['breakdown'][j], ord_lo[j])) for j in range(2)])
    res['certified_set_restricted'] = {}
    for nu in (0.05, 0.075, 0.1, 0.125, 0.15, 0.2):
        keep = g <= nu * nu
        res['certified_set_restricted'][str(nu)] = np.rad2deg(grid[:-1][keep]).tolist()
    if conclusions is not None:
        res['conclusions'] = {}
        for name, bad in conclusions.items():
            res['conclusions'][name] = dict(
                breakdown=float(np.sqrt(cell_lower_sq[bad].min())) if bad.any() else None,
                breakdown_floor_restricted=float(np.sqrt(g[bad].min())) if bad.any() else None)
    return res


def recursive_devs(E0, Eb, orig):
    return [np.sqrt(max(0., op_inner_fast(Eb[j], Eb[j]) + orig[j] - 2 * op_inner_fast(Eb[j], E0[j]))) for j in range(2)]


def fomc(B=999, seed=2026092704):
    from fomc_application import load, stock_first_angle
    d = load(); u = d[['pc1', 'SP500']].values; u = u - u.mean(0); W = d.vix_prev.values
    n = len(u); grid = np.deg2rad(np.arange(0, 91, 1.0))
    z, L = whiten(u)
    ang = np.array([0., stock_first_angle(L)])
    E0 = [z @ rot(t) for t in ang]
    orig = [op_inner_fast(E, E) for E in E0]
    out = dict(n=n, B=B, seed=seed, angles_deg=np.rad2deg(ang).tolist())
    rng = np.random.default_rng(seed)
    for block in (1, int(np.ceil(np.sqrt(n)))):
        vhat = {k: vw_all(z, W, grid, k) for k in KS}
        ohat = {k: vw_all(z, W, ang, k) for k in KS}
        D = {k: np.empty((B, len(grid), 2)) for k in KS}
        Do = {k: np.empty((B, 2, 2)) for k in KS}
        R = np.empty((B, 2))
        t0 = time.time()
        for b in range(B):
            idx = boot_index(n, block, rng)
            ub = u[idx] - u[idx].mean(0); zb, Lb = whiten(ub); Wb = W[idx]
            angb = np.array([0., stock_first_angle(Lb)])
            R[b] = recursive_devs(E0, [zb @ rot(t) for t in angb], orig)
            for k in KS:
                D[k][b] = vw_all(zb, Wb, grid, k) - vhat[k]
                Do[k][b] = vw_all(zb, Wb, angb, k) - ohat[k]
        q = qtile(R.max(1), .975); nr = np.sqrt(orig)
        op_rec = dict(norms=nr.tolist(), q975=q, breakdown=np.sqrt(np.maximum(nr - q, 0)).tolist())
        npz = np.load(os.path.join(OUT, f'fomc_raw_b{block}.npz'))
        cl = npz['cell_lower']; cc = npz['cells']
        smax, pos_any, nonpos_all = cc[:, 0], cc[:, 1] > 0, cc[:, 2] > 0
        concl = {'info_share_le_0.5': smax > .5, 'info_share_le_0.25': smax > .25,
                 'info_type_shock_exists': nonpos_all, 'no_info_type_shock': pos_any}
        out[f'block{block}'] = {f'k{k}': summarize(grid, vhat[k], D[k], ohat[k], Do[k], op_rec, cl, concl) for k in KS}
        out[f'block{block}']['seconds'] = round(time.time() - t0, 1)
        print('fomc', block, {k: (round(v['nu_min_lower'], 3), np.round(v['recursive']['breakdown_floor_restricted'], 3).tolist(),
                                  np.round(v['orders_pair_floor_lower'], 3).tolist())
                              for k, v in out[f'block{block}'].items() if k != 'seconds'}, flush=True)
    with open(os.path.join(OUT, 'nu_floors_fomc.json'), 'w') as f:
        json.dump(out, f, indent=1)


def equity():
    from equity_benchmark import load, second_angle, DRAW_SEEDS
    from svarcore import var_fit, var_regenerate
    y, p, Bc, U, W = load()
    u = U - U.mean(0); n = len(u)
    grid = np.deg2rad(np.arange(0, 91, 3.0))
    z, L = whiten(u)
    ang = np.array([0., second_angle(L)])
    E0 = [z @ rot(t) for t in ang]
    orig = [op_inner_fast(E, E) for E in E0]
    eq = json.load(open(os.path.join(OUT, 'equity_benchmark.json')))
    out = dict(n=n, B=len(DRAW_SEEDS), angles_deg=np.rad2deg(ang).tolist())
    vhat = {k: vw_all(z, W, grid, k) for k in KS}
    ohat = {k: vw_all(z, W, ang, k) for k in KS}
    for block in (10, int(np.ceil(np.sqrt(n)))):
        B = len(DRAW_SEEDS)
        D = {k: np.empty((B, len(grid), 2)) for k in KS}
        Do = {k: np.empty((B, 2, 2)) for k in KS}
        R = np.empty((B, 2))
        t0 = time.time()
        for b in range(B):
            idx = boot_index(n, block, np.random.default_rng(DRAW_SEEDS[b]))
            ys = var_regenerate(Bc, y[:p], u[idx]); _, ub = var_fit(ys, p); ub = ub - ub.mean(0)
            zb, Lb = whiten(ub); Wb = W[idx]
            angb = np.array([0., second_angle(Lb)])
            R[b] = recursive_devs(E0, [zb @ rot(t) for t in angb], orig)
            for k in KS:
                D[k][b] = vw_all(zb, Wb, grid, k) - vhat[k]
                Do[k][b] = vw_all(zb, Wb, angb, k) - ohat[k]
        q = qtile(R.max(1), .975); nr = np.sqrt(orig)
        op_rec = dict(norms=nr.tolist(), q975=q, breakdown=np.sqrt(np.maximum(nr - q, 0)).tolist())
        cl = np.array(eq['raw'][f'block{block}']['cell_lower_sqrt']) ** 2
        out[f'block{block}'] = {f'k{k}': summarize(grid, vhat[k], D[k], ohat[k], Do[k], op_rec, cl) for k in KS}
        out[f'block{block}']['seconds'] = round(time.time() - t0, 1)
        print('equity', block, {k: (round(v['nu_min_lower'], 3), np.round(v['recursive']['breakdown_floor_restricted'], 3).tolist(),
                                    np.round(v['orders_pair_floor_lower'], 3).tolist(), np.round(v['orders_pair_floor_point'], 3).tolist())
                                for k, v in out[f'block{block}'].items() if k != 'seconds'}, flush=True)
    with open(os.path.join(OUT, 'nu_floors_equity.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    {'fomc': fomc, 'equity': equity}[sys.argv[1]]()
