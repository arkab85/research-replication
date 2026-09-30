"""Recompute the exposure floors with the boundary pretest, and add per-shock floors at the
heteroskedasticity-identified rotation, for all three applications.  Results are merged into
the existing JSON files under the key 'floors'.

Run: OPENBLAS_NUM_THREADS=1 python benchmark/floors_update.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, pd, rho_proxy_boot, shock_floors, whiten
from svarcore import var_fit
import fomc_application as fa
import equity_benchmark as eb
import oil_benchmark as ob

OUT = os.path.join(HERE, 'results')
B = 999; SEED = 2026092705


def update(path, blocks, groups):
    res = json.load(open(path))
    rng = np.random.default_rng(SEED)
    for lab, u, W, hetkey in groups:
        u = u - u.mean(0)
        fl = {}
        for k in (2, 3, 5):
            for blk in blocks:
                fl[f'k{k}_block{blk}'] = rho_proxy_boot(u, W, k, B, blk, rng)
            if hetkey is not None:
                ang = res[lab]['het'][hetkey(k)]['angle']
                fl[f'k{k}_shock_floors'] = shock_floors(u, W, k, ang)
        res[lab]['floors'] = fl
        print(lab, {kk: (round(v['rho'], 3), round(v['lower'], 3), round(v['equal_means_p'], 4)) for kk, v in fl.items() if 'shock' not in kk})
        print(lab, {kk: v for kk, v in fl.items() if 'shock' in kk}, flush=True)
    json.dump(res, open(path, 'w'), indent=1)


def main():
    # FOMC
    d = fa.load(); u = d[['pc1', 'SP500']].values; W = d.vix_prev.values; nb = int(np.ceil(np.sqrt(len(u))))
    update(os.path.join(OUT, 'fomc.json'), (1, nb),
           [('raw', u, W, lambda k: f'k{k}_block1'), ('rescaled', u / W[:, None], W, lambda k: f'k{k}_block1')])
    # equity
    y, p, Bc, U, W = eb.load(); n = len(U); nb = int(np.ceil(np.sqrt(n)))
    update(os.path.join(OUT, 'equity_benchmark.json'), (10, nb),
           [('raw', U, W, lambda k: f'k{k}'), ('rescaled', U / W[:, None], W, lambda k: f'k{k}')])
    # oil
    CODE = os.path.dirname(HERE)
    y = np.loadtxt(os.path.join(CODE, 'oil_extended.txt')); Bc, U = var_fit(y, 24); u = (U - U.mean(0))[:, [0, 2]]
    W = ob.ewma(u)
    res = json.load(open(os.path.join(OUT, 'oil_benchmark.json')))
    rng = np.random.default_rng(SEED)
    res['floors'] = {f'k{k}': rho_proxy_boot(u, W, k, B, 12, rng) for k in (2, 3, 5)}
    print('oil', {kk: (round(v['rho'], 3), round(v['lower'], 3), round(v['equal_means_p'], 4)) for kk, v in res['floors'].items()})
    json.dump(res, open(os.path.join(OUT, 'oil_benchmark.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
