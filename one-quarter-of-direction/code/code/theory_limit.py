"""The quantitative prediction of Theorem 1(a) for the design of Table 1, and the finite-sample smoothing of the threshold.
(i) s* = F00^{-1}(1/(2 pi0^2)) with F00 the cdf of (tanh Z - tanh Z')^2; eta = HSIC(tanh Z, Z; kappa_{s*}, l), l Gaussian
with the median-heuristic bandwidth; predicted n*HSIC_n = n * pi0^2 * eta. (ii) n*HSIC_n at first-stage error 0.001 and
0.01, n = 200 and 800. (iii) the probability that the REALISED share of zero-zero pairs exceeds one half.
    python code/theory_limit.py [R]"""
import sys, os, numpy as np, pandas as pd
R = int(sys.argv[1]) if len(sys.argv) > 1 else 150; g = np.random.RandomState(1)
def gram(v, bw2=None):
    v = np.asarray(v, float).reshape(len(v), -1); sq = (v ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * v @ v.T, 0)
    med = np.median(d2[d2 > 0]) if bw2 is None else bw2; K = np.exp(-d2 / (med + 1e-300)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
z1, z2 = g.normal(size=4_000_000), g.normal(size=4_000_000); d00 = np.sort((np.tanh(z1) - np.tanh(z2)) ** 2); rows = []
for pi0 in (0.0, 0.3, 0.5, 0.6, 0.65, 0.69, 0.72, 0.75, 0.8, 0.85, 0.9, 0.97):
    rec = dict(atom_mass=pi0, pair_share=pi0 ** 2)
    if pi0 ** 2 > 0.5:
        s = d00[int(len(d00) / (2 * pi0 ** 2))]; eta = np.mean([(lambda Z: float(np.sum(gram(np.tanh(Z), s) * gram(Z))) / len(Z) ** 2)(g.normal(size=2500)) for _ in range(12)]); rec.update(s_star=s, eta=eta, predicted_HSIC=pi0 ** 2 * eta)
    for n in (200, 800):
        for delta in (0.01, 0.001):
            a, cross = [], []
            for r in range(R):
                Z = g.normal(size=n); B0 = g.rand(n) < pi0; X = g.normal(size=n) * (~B0); e = X - delta * np.tanh(Z); a.append(n * float(np.sum(gram(e) * gram(Z))) / (n - 1) ** 2); m = B0.sum(); cross.append(m * (m - 1) > 0.5 * n * (n - 1))
            rec[f'nHSIC_n{n}_d{delta}'] = np.mean(a); rec[f'P_realised_share_gt_half_n{n}'] = np.mean(cross)
    rows.append(rec); print({k: (round(float(v), 4)) for k, v in rec.items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'theory_limit.csv'), index=False)
