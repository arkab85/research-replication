"""Remark (iv) of the paper: a permutation calibration (residual permuted against the regressors) does not repair the
failure. Setting of theorem_check.py, n = 200, first-stage error 0.01, 199 permutations, 200 replications."""
import numpy as np, pandas as pd, os
def gram(v):
    v = np.asarray(v, float).reshape(len(v), -1); sq = (v ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * v @ v.T, 0); K = np.exp(-d2 / np.median(d2[d2 > 0])); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
g = np.random.RandomState(0); n, R, B = 200, 200, 199; rows = []
for pi0 in (0.0, 0.3, 0.6, 0.75, 0.85, 0.97):
    rej = []
    for r in range(R):
        Z = g.normal(size=n); X = g.normal(size=n) * (g.rand(n) >= pi0); e = X - 0.01 * np.tanh(Z); K, L = gram(e), gram(Z); obs = np.sum(K * L)
        b = np.array([np.sum(K[np.ix_(q, q)] * L) for q in (g.permutation(n) for _ in range(B))]); rej.append((1 + np.sum(b >= obs)) / (B + 1) <= .05)
    rows.append(dict(atom_mass=pi0, permutation_test_rejects_at_5pct=np.mean(rej))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'permutation_check.csv'), index=False)
