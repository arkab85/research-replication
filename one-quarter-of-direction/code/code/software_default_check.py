"""An off-the-shelf implementation at its defaults. hyppo (Panda et al.), independence.Hsic(): Gaussian kernel, bandwidth by
the median heuristic, p-value by permutation (auto=False, reps=199) -- nothing is set by hand except the number of
permutations. Design of Table 1: X = B*V independent of Z, residual e = X - 0.01*tanh(Z), n = 200.
    python code/software_default_check.py [R]"""
import sys, os, inspect, numpy as np, pandas as pd, hyppo
from hyppo.independence import Hsic
import hyppo.tools.common as C
R = int(sys.argv[1]) if len(sys.argv) > 1 else 100; g = np.random.RandomState(0); rows = []
src = inspect.getsource(C.compute_kern); print('hyppo', hyppo.__version__, '| median heuristic in compute_kern:', 'median' in src)
for pi0 in (0.0, 0.3, 0.6, 0.75, 0.85, 0.97):
    rej = []
    for r in range(R):
        Z = g.normal(size=200); X = g.normal(size=200) * (g.rand(200) >= pi0); e = X - 0.01 * np.tanh(Z); _, p = Hsic().test(e.reshape(-1, 1), Z.reshape(-1, 1), reps=199, auto=False, random_state=r); rej.append(p <= .05)
    rows.append(dict(atom_mass=pi0, hyppo_default_rejects_at_5pct=float(np.mean(rej)), reps=R)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'software_default_check.csv'), index=False)
