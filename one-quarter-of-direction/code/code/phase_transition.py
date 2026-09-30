"""The phase transition of Theorem 1, on a fine grid. X = B * V with P(B = 0) = pi0, V standard normal, independent of
Z ~ N(0,1). Residual e = X - delta * tanh(Z), delta = size of the first-stage error. n * HSIC(e, Z) with Gaussian kernels,
median-heuristic bandwidth on e (and, for comparison, a bandwidth fixed at the s.d. of e). Under independence and
delta = 0 the statistic is O(1). The theorem says: for pi0 < 1/sqrt(2) it returns to that level as delta -> 0; for
pi0 > 1/sqrt(2) it converges to n times a positive constant for every delta > 0, so it DIVERGES with n.
    python code/phase_transition.py [R]"""
import sys, os, numpy as np, pandas as pd
R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
def gram(v, bw2=None):
    v = np.asarray(v, float).reshape(len(v), -1); sq = (v ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * v @ v.T, 0)
    med = np.median(d2[d2 > 0]) if bw2 is None else bw2; K = np.exp(-d2 / (med + 1e-300)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
def nhsic(e, z, fixed=False): n = len(e); return n * float(np.sum(gram(e, 2 * np.var(e) if fixed else None) * gram(z))) / (n - 1) ** 2
g = np.random.RandomState(0); rows = []
for n in (200, 800):
    for pi0 in (0.0, 0.3, 0.5, 0.6, 0.65, 0.69, 0.72, 0.75, 0.8, 0.9, 0.97):
        for delta in (0.1, 0.01):
            a, b = [], []
            for r in range(R):
                Z = g.normal(size=n); X = g.normal(size=n) * (g.rand(n) >= pi0); e = X - delta * np.tanh(Z); a.append(nhsic(e, Z)); b.append(nhsic(e, Z, True))
            rows.append(dict(n=n, atom_mass=pi0, first_stage_error=delta, n_HSIC_median=np.mean(a), n_HSIC_fixed=np.mean(b), share_zero_zero_pairs=pi0 ** 2))
        print(rows[-2], flush=True); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'phase_transition.csv'), index=False)
