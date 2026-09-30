"""Bandwidth-set version of the recursive-ordering tests in the four-variable monetary application:
bandwidths s in {0.5, 1, 2}, joint .95 critical value of max over bandwidths, orderings and pairs of
s^2 ||C^{s*} - C^s||, breakdown max_s sqrt([s^2 max_pairs ||C^s|| - q]_+), raw and conditioned on
three VIX regimes.  Writes results/fomc4_bw.json."""
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from dcore import whiten, pair_sq_norms, pair_devs, ordering_rotations
from acore import quantile_bins, qtile
from bcore import boot_index
from fomc4 import load, K

BWS = (0.5, 1.0, 2.0); B = int(os.environ.get('BLIMIT', 499)); SEED = 2026092711


def run(u, W, block, rng):
    n, d = u.shape; z, L = whiten(u); orders = ordering_rotations(L, d); keys = list(orders)
    bins = quantile_bins(W, K); one = np.zeros(n, int)
    E0 = {pi: z @ Q for pi, Q in orders.items()}
    v0 = {(c, s, pi): pair_sq_norms(E0[pi], None if c == 'raw' else bins, K, s) for c in ('raw', 'cond') for s in BWS for pi in keys}
    D = {c: np.empty((B, len(BWS), len(keys), 6)) for c in ('raw', 'cond')}
    for b in range(B):
        idx = boot_index(n, block, rng); ub = u[idx] - u[idx].mean(0); zb, Lb = whiten(ub)
        bb = quantile_bins(W[idx], K); ob = ordering_rotations(Lb, d)
        for m, pi in enumerate(keys):
            Eb = zb @ ob[pi]
            for si, s in enumerate(BWS):
                D['raw'][b, si, m] = s * s * pair_devs(Eb, np.zeros(n, int), E0[pi], one, 1, v0[('raw', s, pi)], s)
                D['cond'][b, si, m] = s * s * pair_devs(Eb, bb, E0[pi], bins, K, v0[('cond', s, pi)], s)
    res = {}
    for c in ('raw', 'cond'):
        q = qtile(D[c].max(axis=(1, 2, 3)), .95)
        brk = {''.join(map(str, pi)): float(np.sqrt(max(0., max(s * s * np.sqrt(v0[(c, s, pi)]).max() - q for s in BWS)))) for pi in keys}
        res[c] = dict(q=q, breakdown=brk, rejected=int(sum(v > 0 for v in brk.values())))
    return res


if __name__ == '__main__':
    d, u, W = load(); u = u - u.mean(0); n = len(u)
    rng = np.random.default_rng(SEED); out = dict(n=n, B=B)
    for block in (1, int(np.ceil(np.sqrt(n)))):
        out[f'block{block}'] = r = run(u, W, block, rng)
        print(block, {c: (v['rejected'], round(max(v['breakdown'].values()), 3)) for c, v in r.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'fomc4_bw.json'), 'w'), indent=1)
