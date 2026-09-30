"""EXPLORATORY, post hoc (not part of the pre-specified protocol): block-length
sensitivity of the studentized whitening region in the two designs where the
pre-specified blocks of ten undercovered (V6 log-normal SV, V7 two-state stay .98).
Seeds: SeedSequence([2026092700, block, design]) -- disjoint from all earlier families."""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from confirm_study import W_DESIGNS, l_of, RES
from designs import simulate
from whiten_region import studentized_region, contains

def one(task):
    name, ell, seed = task
    d, n, _ = W_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(d, n, rng)
    reg = studentized_region(y, 1, [0, 1], ell, 1999, rng)
    return dict(seed=seed, cover=contains(reg, l_of(L0)), c=reg['c'])

if __name__ == '__main__':
    out = {}; reps = 1000
    for di, name in enumerate(('V6_logsv_n600', 'V7_mk98_mix_n600')):
        for ell in (20, 30):
            seeds = [int(s) for s in np.random.SeedSequence([2026092700, ell, di]).generate_state(reps)]
            t = time.time()
            with ProcessPoolExecutor(2) as ex:
                rows = list(ex.map(one, [(name, ell, s) for s in seeds], chunksize=4))
            p = float(np.mean([r['cover'] for r in rows]))
            out[f'{name}_block{ell}'] = dict(reps=reps, coverage=100 * p, mc_se=100 * np.sqrt(p * (1 - p) / reps),
                                             median_critical=float(np.median([r['c'] for r in rows])))
            print(name, ell, out[f'{name}_block{ell}'], round(time.time() - t), flush=True)
            json.dump(out, open(os.path.join(RES, 'exploratory_blocks.json'), 'w'), indent=2)
