"""Co-skewness Wald statistics over rotations (FOMC, iid bootstrap covariance, 999 draws).
Under the latent common-state model the co-skewness moments vanish at the true rotation
(Remark on restrictions that survive latent confounding).  Writes results/coskew_fomc.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, whiten, boot_index
from svarcore import coskew
from fomc_application import load
d = load(); u = d[['pc1', 'SP500']].values; u = u - u.mean(0); n = len(u)
z, L = whiten(u)
G = np.deg2rad(np.arange(-45, 45, 1.0))
c0 = coskew(z, G)
rng = np.random.default_rng(2026092705)
CB = np.empty((999, len(G), 2))
for b in range(999):
    ub = u[boot_index(n, 1, rng)]; zb, _ = whiten(ub - ub.mean(0)); CB[b] = coskew(zb, G)
w = np.array([c0[j] @ np.linalg.solve(np.cov(CB[:, j].T), c0[j]) for j in range(len(G))])
out = dict(deg=np.rad2deg(G).tolist(), wald=w.tolist(), min=float(w.min()), argmin=float(np.rad2deg(G[w.argmin()])))
json.dump(out, open(os.path.join(HERE, 'results', 'coskew_fomc.json'), 'w'), indent=1)
for a in (-16, -10, 0, 10, 20, 31, 34):
    print(a, round(w[list(np.rad2deg(G).round()).index(a)], 2))
print('min', out['min'], 'at', out['argmin'])
