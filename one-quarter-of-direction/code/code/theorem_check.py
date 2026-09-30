"""Numerical check of the non-vanishing result. X is independent of Z. The residual is e = X - delta * h(Z): delta is the
size of the first-stage estimation error, which goes to zero as the training sample grows. With a continuous X the
dependence between e and Z is of order delta^2 and n * HSIC returns to its null level. With an atom of mass pi0 at zero
and a median-heuristic bandwidth, the bandwidth is proportional to delta, the kernel on the zero dates is scale-free, and
n * HSIC does NOT fall as delta -> 0.  n = 400, 300 replications.      python code/theorem_check.py"""
import numpy as np, pandas as pd, os
def gram(v, bw2=None):
    v = np.asarray(v, float).reshape(len(v), -1); sq = (v ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * v @ v.T, 0)
    med = np.median(d2[d2 > 0]) if bw2 is None else bw2; K = np.exp(-d2 / (med + 1e-300)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
def nhsic(e, z, fixed=False): n = len(e); return n * float(np.sum(gram(e, 2 * np.var(e) if fixed else None) * gram(z))) / (n - 1) ** 2
g = np.random.RandomState(0); n, R = 400, 300; rows = []
for pi0 in (0.0, 0.3, 0.5, 0.75, 0.85, 0.97):
    for delta in (1.0, 0.3, 0.1, 0.03, 0.01, 0.0):
        a, b = [], []
        for r in range(R):
            Z = g.normal(size=n); X = g.normal(size=n) * (g.rand(n) >= pi0); e = X - delta * np.tanh(Z)
            if delta == 0 and pi0 > 0: e = e + 1e-12 * g.normal(size=n)          # break exact ties so the median is defined the same way
            a.append(nhsic(e, Z)); b.append(nhsic(e, Z, fixed=True))
        rows.append(dict(atom_mass=pi0, first_stage_error=delta, n_HSIC_median_bandwidth=np.mean(a), n_HSIC_fixed_bandwidth=np.mean(b)))
    print(pd.DataFrame(rows[-6:]).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'theorem_check.csv'), index=False)
