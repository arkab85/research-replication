"""Exact rotation tests, shared by the hold-out, cross-fit and repaired-statistic scripts (same logic as paper_tables.py).
A 'unit' here is a triple (first stage, X_test, Y_test); units may come from any split, so the same functions serve the
honest end-of-sample split and the cross-fitted blocks."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC


def rotate_units(S, hs, p=P_LAGS, stat=None):
    """S[h] = list of (s1, X, Y). Returns obs[h] (units,), M[h] (K, units), K, all units and horizons rotated together."""
    stat = stat or SC.stat; span = {h: [len(X) - 2 * (h + p + 2) for _, X, _ in S[h]] for h in hs}; K = max(max(v) for v in span.values())
    obs = {h: np.array([stat(s1, X, Y, h, p) for s1, X, Y in S[h]]) for h in hs}
    M = {h: np.array([[stat(s1, X, np.roll(Y, h + p + 2 + int(round(j / (K - 1) * (sp - 1)))), h, p) for (s1, X, Y), sp in zip(S[h], span[h])] for j in range(K)]) for h in hs}
    return obs, M, K
def pvalue(obs, M, K, cols, w):
    o = sum(w[h] * obs[h][cols].mean() for h in w); b = sum(w[h] * M[h][:, cols].mean(1) for h in w)
    return dict(value=o, mean_under_rotation=b.mean(), p_rotation=(1 + np.sum(b >= o)) / (K + 1), min_p=1 / (K + 1), rotations=K)
STATS = {'P(1)': {1: 1.}, 'P(3)': {3: 1.}, 'P(6)': {6: 1.}, 'hump': HUMP, 'P(3)-P(1)': D31, 'P(3)-P(6)': D36}
def report(rows, block, keys, S, sets, stats=STATS, p=P_LAGS, stat=None):
    obs, M, K = rotate_units(S, HS, p, stat)
    for name, ks in sets.items():
        c = [i for i, k in enumerate(keys) if k in ks]
        for st, w in stats.items():
            pos = int(sum((obs[h][c] > 0).sum() for h in w)) if len(w) == 1 else np.nan
            rows.append(dict(block=block, set=name, pairs=len(set(ks)), units=len(c), statistic=st, positive=pos, **pvalue(obs, M, K, c, w)))
    print(block, 'done; K =', K, flush=True); return obs, M, K
def end_split(ds, hs=HS, p=P_LAGS): return {h: [SC.split(d, h, p) for d in ds] for h in hs}
