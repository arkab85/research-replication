"""What drives the no-channel over-rejection at MONTHLY frequency, where the atom (about 0.3) is below the 1/sqrt(2)
threshold of Theorem 1? For each actual shock series as the cause of an independent AR(1) outcome (design of
audit_actual_series.py) this reports the wild / paired / intersection rejection rates at 5 percent under
  (a) the default median-heuristic bandwidth on the residual,
  (b) the residual bandwidth FIXED at the residual's standard deviation,
  (c) the default bandwidth with the series' values randomly re-ordered in time (same marginal, no time structure),
together with two diagnostics of the evaluation block: tau, the share of pairs with |X_i - X_j| <= |g_i - g_j| (g = fitted
backward first stage), and the block's share of the series' sum of squares.
    python code/monthly_mechanism.py <part 0|1> [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, engine
_default = engine.cgram
def _cgram_fixed_resid(Z):
    Z = np.asarray(Z, float)
    if Z.ndim > 1 and Z.shape[1] > 1: return _default(Z)
    v = Z.reshape(-1, 1); n = len(v); d2 = (v - v.T) ** 2; K = np.exp(-d2 / (2 * np.var(v) + 1e-300)); H = np.eye(n) - np.ones((n, n)) / n; return H @ K @ H
from core_lib import *
part = int(sys.argv[1]) if len(sys.argv) > 1 else 0; R = int(sys.argv[2]) if len(sys.argv) > 2 else 150
D = load_panel(); Hh = pd.read_csv(os.path.join(ROOT, 'holdout', 'holdout_panel.csv'), index_col=0); Hh.index = Hh.index.astype(str)
S = [('Narrative (Romer-Romer, Acosta)', D['rr'].loc['1974-02':'2019-12']), ('Jarocinski-Karadi', D['jk'].loc['1990-01':'2019-12']), ('Bauer-Swanson', D['bs'].loc['1988-02':'2019-12']), ('Kanzig oil supply news', D['oil'].loc['1975-01':'2019-12']),
     ('Swanson federal funds rate factor', Hh['sw_ffr'].loc['1991-07':'2019-06']), ('Swanson forward guidance factor', Hh['sw_fg'].loc['1991-07':'2019-06']), ('Swanson LSAP factor', Hh['sw_lsap'].loc['1991-07':'2019-06']), ('Bu-Rogers-Wu', Hh['brw'].loc['1994-01':'2019-12'])]
S = S[part::2]
def ar1(T, g):
    e = g.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
    return Y
rows = []
for name, s in S:
    x0 = s.dropna().values; T = len(x0); n = int(np.floor(T / np.log(T))); blk = x0[-n:]; rec = {'series': name, 'T': T, 'zero_share': float(np.mean(x0 == 0)), 'block_share_of_sum_of_squares': float(np.sum(blk ** 2) / np.sum(x0 ** 2)), 'reps': R}
    for tag, fixed, shuffle in (('median', False, False), ('fixed', True, False), ('shuffled', False, True)):
        engine.cgram = _cgram_fixed_resid if fixed else _default; g = np.random.RandomState(0); out = []; taus = []
        for r in range(R):
            x = g.permutation(x0) if shuffle else x0; d = pd.DataFrame({'x': x, 'y': ar1(T, g)}, index=np.arange(T).astype(str)); u = unit(d, 3); pw, pp = cell_tests(u, 199, 99, seed=r); out.append((u['obs'], pw, pp))
            if tag == 'median':
                ef, eb = u['s1'].resid(u['dte']); X = u['dte']['X_t'].values; gh = X - eb; iu = np.triu_indices(len(X), 1); taus.append(np.mean(np.abs(X[:, None] - X[None, :])[iu] <= np.abs(gh[:, None] - gh[None, :])[iu]))
        o = np.array(out); rec.update({f'{tag}_mean_index': o[:, 0].mean(), f'{tag}_wild_05': (o[:, 1] <= .05).mean(), f'{tag}_paired_05': (o[:, 2] <= .05).mean(), f'{tag}_both_05': (np.maximum(o[:, 1], o[:, 2]) <= .05).mean()})
        if taus: rec['tau_near_tie_pair_share'] = float(np.mean(taus))
    engine.cgram = _default; rows.append(rec); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, f'monthly_mechanism_part{part}.csv'), index=False)
