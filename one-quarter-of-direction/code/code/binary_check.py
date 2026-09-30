r"""Executes code/BINARY_CHECK.md, anchored 2026-09-24T17:50:34Z before this was run.

The binary-indicator case of Corollary 1, simulated. The paper asserts it and never exhibits it:
Table 2 is the atom-plus-continuous design X = B*V with p2 = pi0^2, a different formula.

    python binary_check.py [reps]
"""
import sys, os, numpy as np, pandas as pd

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PI = (0.40, 0.50, 0.55, 0.70, 0.85)
NS = (200, 800)
DELTA = 0.01
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def hsic(e, Z, sig2):
    """n * HSIC with a Gaussian kernel of squared bandwidth sig2 on e and median heuristic on Z."""
    n = len(e)
    De = (e[:, None] - e[None, :]) ** 2
    Dz = (Z[:, None] - Z[None, :]) ** 2
    sz = np.median(Dz[np.triu_indices(n, 1)])
    K = np.exp(-De / sig2)
    L = np.exp(-Dz / (sz if sz > 0 else 1.0))
    H = np.eye(n) - 1.0 / n
    return n * np.trace(K @ H @ L @ H) / n ** 2


rows = []
for pi0 in PI:
    p2 = pi0 ** 2 + (1 - pi0) ** 2
    for n in NS:
        stats, rej, above, zero_med = [], 0, 0, 0
        for r in range(REPS):
            rng = np.random.RandomState(10_000 * int(pi0 * 100) + 17 * n + r)
            X = (rng.rand(n) < pi0).astype(float)
            Z = rng.randn(n)
            e = X - DELTA * np.tanh(Z)
            d = (e[:, None] - e[None, :]) ** 2
            iu = np.triu_indices(n, 1)
            med = np.median(d[iu])
            zero_med += (med <= 0)
            above += (np.mean(np.isclose(X[iu[0]], X[iu[1]])) > 0.5)
            sig2 = med if med > 0 else 1.0
            s = hsic(e, Z, sig2)
            stats.append(s)
            # standard permutation test, 199 draws
            null = np.empty(199)
            for b in range(199):
                null[b] = hsic(e, Z[rng.permutation(n)], sig2)
            rej += ((1 + (null >= s).sum()) / 200.0 <= 0.05)
        rows.append(dict(pi0=pi0, p2=round(p2, 4), n=n, nhsic=np.mean(stats),
                         reject_pct=100.0 * rej / REPS,
                         tied_share_above_half_pct=100.0 * above / REPS,
                         zero_median_pct=100.0 * zero_med / REPS, reps=REPS))
        print('pi0 %.2f  p2 %.3f  n %3d | nHSIC %7.3f | reject %5.1f%% | tie>1/2 %5.1f%% | zero med %4.1f%%'
              % (pi0, p2, n, rows[-1]['nhsic'], rows[-1]['reject_pct'],
                 rows[-1]['tied_share_above_half_pct'], rows[-1]['zero_median_pct']), flush=True)

D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, 'binary_check.csv'), index=False, float_format='%.5g')
print('\nwrote results/binary_check.csv')
