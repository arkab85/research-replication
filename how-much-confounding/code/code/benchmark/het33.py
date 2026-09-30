"""Heteroskedasticity-identified rotation for the equity application with blocks of 33 weeks
(499 refitted draws), complementing the blocks-of-ten intervals of equity_benchmark.py.
Writes results/equity_het33.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np
from equity_benchmark import load, het_boot
y, p, Bc, U, W = load(); n = len(U)
rng = np.random.default_rng(2026092709)
out = {f'k{k}': het_boot(y, p, Bc, U, W, np.ones(n), k, int(np.ceil(np.sqrt(n))), 499, rng) for k in (2, 3, 5)}
json.dump(out, open(os.path.join(HERE, 'results', 'equity_het33.json'), 'w'), indent=1)
print({k: (round(v['angle'], 1), np.round(v['ci95'], 1).tolist()) for k, v in out.items()})
