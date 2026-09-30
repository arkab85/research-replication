"""Applications: studentised whitening region and the revised oil economic projection.

The operator part (bootstrap .975 max-norm radius, segment minima, curvature
allowance, retained cells) is taken unchanged from revision_results/<name>.npz.
Only the whitening region, and therefore the elasticity projection, is new.
"""
import os, sys, json
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import pandas as pd
from whiten_region import studentized_region, elasticity_cells, linear_range
from certified_revision import TH, OUT, ROOT
from wcore import var_fit

B = 1999
SEEDS = {'oil': 2026092611, 'equity': 2026092612}


def data(name):
    if name == 'oil':
        return np.loadtxt(ROOT / 'oil_extended.txt'), 24, [0, 2], 12
    d = pd.read_csv(ROOT / 'weekly.csv', index_col=0, parse_dates=True)
    return d.values, 4, [0, 1], 10


def main():
    summary = {}
    for name in ('oil', 'equity'):
        y, p, cols, ell = data(name)
        rng = np.random.default_rng(SEEDS[name])
        reg = studentized_region(y, p, cols, ell, B, rng)
        Bh, U = var_fit(y, p)
        Lnp = np.linalg.cholesky(np.cov(U[:, cols].T))
        prev = json.loads((OUT / (name + '.json')).read_text())
        old = np.load(OUT / (name + '.npz'))
        lower = old['cell_lower']
        l = reg['l']; sd = np.sqrt(np.diag(reg['V']) / reg['n'])
        ranges = {k: [float(x) for x in linear_range(reg, a)] for k, a in
                  (('L11', [1, 0, 0]), ('L21', [0, 1, 0]), ('L22', [0, 0, 1]))}
        # radius of the earlier Frobenius ball implied by the new ellipsoid, for comparison only
        res = dict(name=name, n=reg['n'], p=p, K=reg['K'], block=ell, B=B, seed=SEEDS[name], level=reg['level'],
                   l_hat=l.tolist(), l_np_cov=[Lnp[0, 0], Lnp[1, 0], Lnp[1, 1]], se=sd.tolist(),
                   critical_value=reg['c'], chi2_3_975=9.348404, coordinate_ranges=ranges,
                   previous_percentile_radius=prev['r_whitening_975'], operator_radius=prev['q_operator_975'])
        np.savez_compressed(OUT / (name + '_whitening.npz'), T=reg['T'], l=l, V=reg['V'], c=reg['c'], seed=SEEDS[name])
        if name == 'oil':
            bounds = elasticity_cells(reg, TH)
            sets = {}
            for rho in (0, .025, .05, .1, .15, .2):
                keep = lower <= rho * rho
                lo = float(np.nanmin(bounds[keep, 0])) if keep.any() else None
                hi = float(np.nanmax(bounds[keep, 1])) if keep.any() else None
                sets[str(rho)] = dict(cell_share=float(keep.mean()), elasticity_min=lo,
                                      elasticity_max='unbounded' if hi is not None and np.isinf(hi) else hi)
            breakdown = {}
            for c in (.0258, .05, .1, .2):
                bad = bounds[:, 1] >= c       # cells whose outer range reaches an elasticity of at least c
                breakdown[str(c)] = float(np.sqrt(np.min(lower[bad]))) if bad.any() else None
            # cells that are finite / informative
            finite = np.isfinite(bounds[:, 1])
            res.update(sets=sets, breakdown_lower=breakdown,
                       cells_with_finite_upper=int(finite.sum()),
                       cell_bounds=[[float(a), (float(b) if np.isfinite(b) else 'unbounded')] for a, b in bounds])
            np.save(OUT / 'oil_elasticity_bounds_studentized.npy', bounds)
        (OUT / (name + '_whitening.json')).write_text(json.dumps(res, indent=2))
        summary[name] = {k: res[k] for k in ('critical_value', 'l_hat', 'se', 'coordinate_ranges', 'previous_percentile_radius')}
        if name == 'oil':
            summary[name]['sets'] = res['sets']; summary[name]['breakdown'] = res['breakdown_lower']
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
