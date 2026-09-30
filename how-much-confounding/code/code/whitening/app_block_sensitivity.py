"""Pre-specified block-length sensitivity for the applications (PRESPECIFICATION_ADDENDUM.md).

Equity: recursive-ordering operator tests and co-skewness Wald statistics recomputed with the
original 399 seeds and moving blocks of 10 (reproduction check), 20 and ceil(sqrt(n)) = 33.
Only the recursive functional is computed; the draw sequence is identical to
certified_revision.draw, so block 10 reproduces the published numbers exactly.
Oil: studentized whitening region with ceil(sqrt(n)) = 25-month blocks and the implied projection.
"""
import os, sys, json
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import chi2
from certified_revision import spec, fit, rec, OUT, TH, ROOT
from svarcore import var_regenerate, mbb_indices, whiten, coskew
from confirm_study import distances_fast
from whiten_region import studentized_region, elasticity_cells

S = None


def init(s):
    global S; S = s


def draw(seed):
    s = S; rng = np.random.default_rng(seed)
    idx = mbb_indices(len(s['U']), s['block'], rng)
    pool = s['U'][idx]
    innovations = np.zeros((len(s['keep']), s['y'].shape[1])); innovations[s['keep']] = pool
    ys = var_regenerate(s['B'], s['y'][:s['p']], innovations + s['F'])
    _, _, u, _ = fit(ys, s['p'], s['drop']); z, l = whiten(u[:, s['cols']]); a = rec(l)
    return distances_fast(z, s['Z'], a, s['aa'], s['origr']), coskew(z, a)


def equity(block, workers=2):
    s = spec('equity'); s['block'] = block
    seeds = np.load(OUT / 'equity.npz')['seeds']
    with ProcessPoolExecutor(workers, initializer=init, initargs=(s,)) as ex:
        res = list(ex.map(draw, seeds, chunksize=8))
    DR = np.array([r[0] for r in res]); CS = np.array([r[1] for r in res])
    qr = float(np.quantile(DR.max(1), .95, method='higher'))
    norms = np.sqrt(s['origr'])
    cs = coskew(s['Z'], s['aa'])
    W = [float(cs[j] @ np.linalg.solve(np.cov(CS[:, j, :].T), cs[j])) for j in range(2)]
    return dict(block=block, B=len(seeds), recursive_q95=qr, recursive_norms=norms.tolist(),
                recursive_budget_lower=np.sqrt(np.maximum(norms - qr, 0)).tolist(), recursive_wald=W,
                recursive_p_asymptotic=chi2.sf(W, 2).tolist())


def oil(block=25):
    y = np.loadtxt(ROOT / 'oil_extended.txt')
    reg = studentized_region(y, 24, [0, 2], block, 1999, np.random.default_rng(2026092814))
    lower = np.load(OUT / 'oil.npz')['cell_lower']; b = elasticity_cells(reg, TH)
    sets = {}
    for rho in (0, .025, .05, .1, .15):
        keep = lower <= rho * rho
        hi = np.nanmax(b[keep, 1])
        sets[str(rho)] = [float(np.nanmin(b[keep, 0])), 'unbounded' if np.isinf(hi) else float(hi)]
    return dict(block=block, critical_value=reg['c'], sets=sets)


if __name__ == '__main__':
    out = {'equity': {}, 'oil': oil(25)}
    print('oil', out['oil'], flush=True)
    for block in (10, 20, 33):
        out['equity'][str(block)] = r = equity(block)
        print('equity', block, r['recursive_budget_lower'], r['recursive_wald'], flush=True)
    pub = json.loads((OUT / 'equity.json').read_text())
    assert np.allclose(out['equity']['10']['recursive_budget_lower'], pub['recursive_budget_lower'], atol=1e-12)
    assert np.allclose(out['equity']['10']['recursive_wald'], pub['recursive_wald'], atol=1e-9)
    out['equity_block10_reproduces_published'] = True
    (OUT / 'block_sensitivity.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
