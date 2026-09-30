"""DEVELOPMENT check of the candidate frozen whitening procedure (whiten_region.py).
Seeds SeedSequence(7003xx).  Not a confirmatory result."""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from designs import simulate
from whiten_region import studentized_region, contains
from dev_whitening import DEV

def one(task):
    d, n, ell, seed = task
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(d, n, rng)
    reg = studentized_region(y, 1, [0, 1], ell, 1999, rng)
    return dict(cover=contains(reg, [L0[0, 0], L0[1, 0], L0[1, 1]]), c=reg['c'])

if __name__ == '__main__':
    reps = int(sys.argv[1]); out = {}; t = time.time()
    for di, (name, d) in enumerate(DEV.items()):
        for n in (300, 600):
            ell = 1 if d['vol'][0] == 'iid2' else 10
            seeds = np.random.SeedSequence(700300 + 10 * di + n // 300).generate_state(reps)
            with ProcessPoolExecutor(2) as ex:
                rows = list(ex.map(one, [(d, n, ell, int(s)) for s in seeds], chunksize=8))
            out[f'{name}_n{n}'] = dict(cover=float(np.mean([r['cover'] for r in rows])), c_median=float(np.median([r['c'] for r in rows])))
            print(name, n, out[f'{name}_n{n}'], round(time.time() - t), flush=True)
    json.dump(out, open('dev_frozen_check.json', 'w'), indent=2)
