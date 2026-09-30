"""Confidence band for the embedding floors through the Hilbert-valued vector
    a_i(theta) = (sqrt(p_b) (mu_b - mu))_b,   ||a_i(theta)||^2 = v_i^W(theta),
so that on the event max ||a* - a|| <= q the reverse triangle inequality gives
    v_i^W(theta)^{1/2} >= [||a_hat|| - q]_+
without the degeneracy of a quadratic functional at zero.  q is the .975 bootstrap quantile of the
maximum over the rotation grid and both shocks of ||a*_i(theta) - a_hat_i(theta)|| (plug-in feature
means), separately for each k; a pointwise version at a given angle is also reported.
Writes results/nu_band_{app}.json.  Run: OPENBLAS_NUM_THREADS=1 python benchmark/nu_band.py fomc|equity
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index
from acore import gram, quantile_bins, qtile

OUT = os.path.join(HERE, 'results'); KS = (2, 3, 5)


def weights(b, k):
    n = len(b); p = np.bincount(b, minlength=k) / n
    W = np.zeros((n, k)); W[np.arange(n), b] = 1.0 / np.bincount(b, minlength=k)[b]
    return (W - 1.0 / n) * np.sqrt(p)[None, :]


def run(z, W, grid, draws, extra=None):
    """draws: iterable of (zb, Wb).  Returns point norms and per-draw max deviations."""
    n = len(z); ang = list(grid) + ([extra] if extra is not None else [])
    E = [z @ rot(t) for t in ang]
    Wo = {k: weights(quantile_bins(W, k), k) for k in KS}
    base = {k: np.array([[np.einsum('ik,ij,jk->', Wo[k], gram(e[:, i], e[:, i]), Wo[k]) for i in range(2)] for e in E]) for k in KS}
    norm = {k: np.sqrt(np.maximum(base[k], 0)) for k in KS}
    dev = {k: [] for k in KS}
    for zb, Wb in draws:
        Wbk = {k: weights(quantile_bins(Wb, k), k) for k in KS}
        d = {k: np.empty((len(ang), 2)) for k in KS}
        for j, t in enumerate(ang):
            eb = zb @ rot(t)
            for i in range(2):
                Kbb = gram(eb[:, i], eb[:, i]); Kbo = gram(eb[:, i], E[j][:, i])
                for k in KS:
                    v = np.einsum('ik,ij,jk->', Wbk[k], Kbb, Wbk[k]) - 2 * np.einsum('ik,ij,jk->', Wbk[k], Kbo, Wo[k]) + base[k][j, i]
                    d[k][j, i] = np.sqrt(max(v, 0))
        for k in KS:
            dev[k].append(d[k])
    out = {}
    for k in KS:
        D = np.array(dev[k]); G = len(grid)
        q = qtile(D[:, :G].max(axis=(1, 2)), .975)
        lo = np.maximum(norm[k][:G] - q, 0)
        res = dict(q_joint=q, nu_plugin=norm[k][:G].tolist(), nu_lower=lo.tolist(),
                   pair_lower_max=float(np.sqrt(lo[:, 0] * lo[:, 1]).max()))
        if extra is not None:
            qp = qtile(D[:, G].max(1), .975)
            lo_e = np.maximum(norm[k][G] - qp, 0)
            res['extra'] = dict(angle_deg=float(np.rad2deg(extra)), nu_plugin=norm[k][G].tolist(), q_pointwise=qp,
                                nu_lower_pointwise=lo_e.tolist(), pair_lower_pointwise=float(np.sqrt(lo_e.prod())))
        out[f'k{k}'] = res
    return out


def fomc():
    from fomc_application import load
    d = load(); u = d[['pc1', 'SP500']].values; u = u - u.mean(0); W = d.vix_prev.values; n = len(u)
    z, _ = whiten(u); grid = np.deg2rad(np.arange(0, 91, 1.0))
    rng = np.random.default_rng(2026092708); out = {}
    for block in (1, int(np.ceil(np.sqrt(n)))):
        def draws():
            for _ in range(999):
                idx = boot_index(n, block, rng); ub = u[idx] - u[idx].mean(0)
                yield whiten(ub)[0], W[idx]
        t0 = time.time(); out[f'block{block}'] = run(z, W, grid, draws())
        print('fomc', block, {k: round(v['pair_lower_max'], 3) for k, v in out[f'block{block}'].items()}, round(time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(OUT, 'nu_band_fomc.json'), 'w'), indent=1)


def equity():
    from equity_benchmark import load, DRAW_SEEDS
    from svarcore import var_fit, var_regenerate
    y, p, Bc, U, W = load(); u = U - U.mean(0); n = len(u)
    z, _ = whiten(u); grid = np.deg2rad(np.arange(0, 91, 3.0))
    het = json.load(open(os.path.join(OUT, 'equity_benchmark.json')))['raw']['het']['k2']['angle']
    out = {}
    for block in (10, int(np.ceil(np.sqrt(n)))):
        def draws():
            for s in DRAW_SEEDS:
                idx = boot_index(n, block, np.random.default_rng(s))
                ys = var_regenerate(Bc, y[:p], u[idx]); _, ub = var_fit(ys, p); ub = ub - ub.mean(0)
                yield whiten(ub)[0], W[idx]
        t0 = time.time(); out[f'block{block}'] = run(z, W, grid, draws(), extra=np.deg2rad(het))
        print('equity', block, {k: (round(v['pair_lower_max'], 3), round(v['extra']['pair_lower_pointwise'], 3)) for k, v in out[f'block{block}'].items()}, round(time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(OUT, 'nu_band_equity.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'fomc': fomc, 'equity': equity}[sys.argv[1]]()
