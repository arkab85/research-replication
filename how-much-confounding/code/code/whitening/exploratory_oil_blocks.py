"""EXPLORATORY, post hoc: oil whitening region with 24-month blocks (operator part unchanged)."""
import os, sys, json
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from whiten_region import studentized_region, elasticity_cells
from certified_revision import TH, OUT, ROOT
y = np.loadtxt(ROOT / 'oil_extended.txt')
reg = studentized_region(y, 24, [0, 2], 24, 1999, np.random.default_rng(2026092613))
lower = np.load(OUT / 'oil.npz')['cell_lower']; b = elasticity_cells(reg, TH)
res = {'block': 24, 'critical_value': reg['c'], 'sets': {}}
for rho in (0, .025, .05, .1, .15):
    keep = lower <= rho * rho
    res['sets'][str(rho)] = [float(np.nanmin(b[keep, 0])), 'unbounded' if np.isinf(np.nanmax(b[keep, 1])) else float(np.nanmax(b[keep, 1]))]
(OUT / 'oil_whitening_block24_exploratory.json').write_text(json.dumps(res, indent=2)); print(res)
