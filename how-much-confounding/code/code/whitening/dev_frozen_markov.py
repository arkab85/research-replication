import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from dev_frozen_check import one
from dev_whitening import DEV
if __name__ == '__main__':
    reps = 1000; out = {}; t = time.time(); d = DEV['markov_d4_exp']; di = 2
    for n in (300, 600):
        seeds = np.random.SeedSequence(700300 + 10 * di + n // 300).generate_state(reps)
        with ProcessPoolExecutor(2) as ex:
            rows = list(ex.map(one, [(d, n, 10, int(s)) for s in seeds], chunksize=8))
        out[f'markov_d4_exp_n{n}'] = dict(cover=float(np.mean([r['cover'] for r in rows])), c_median=float(np.median([r['c'] for r in rows])))
        print(out, round(time.time() - t), flush=True)
    json.dump(out, open('dev_frozen_markov.json', 'w'), indent=2)
