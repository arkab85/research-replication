"""Why the index is positive under sparse shocks with no channel. Sparse design (X zero w.p. 0.85, Y an independent AR(1),
T = 551, h = 3, honest split, 200 replications); the index is recomputed with three changes, one at a time:
  (a) baseline: median-heuristic bandwidth on the residual;
  (b) residual bandwidth fixed at the residual's standard deviation (not set by the bulk of near-zero residual pairs);
  (c) backward residual replaced by X_t - mean(X) (no first stage at all: removes the fitted function of the regressors);
  (d) index computed on the nonzero-shock dates only.
    python code/mechanism_check.py [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
def gram(Z, bw2=None):
    Z = np.asarray(Z, float); Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z; sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(d2[d2 > 0]) if bw2 is None else bw2; K = np.exp(-d2 / (2 * (med / 2.0) + 1e-12)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
def hs(e, Z, fixed): n = len(e); return float(np.sum(gram(e, 2 * np.var(e) if fixed else None) * gram(Z))) / (n - 1) ** 2
out = {k: [] for k in 'abcd'}; bwr = []; g = np.random.RandomState(0)
for r in range(R):
    T = 551; X = g.normal(size=T) * (g.rand(T) < .15); e = g.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
    u = unit(pd.DataFrame({'x': X, 'y': Y}, index=np.arange(T).astype(str)), 3); d = u['dte']; ef, eb = u['s1'].resid(d); Zf, Zb = d[d.attrs['f']].values, d[d.attrs['b']].values
    out['a'].append(hs(eb, Zb, False) - hs(ef, Zf, False)); out['b'].append(hs(eb, Zb, True) - hs(ef, Zf, True))
    out['c'].append(hs(d['X_t'].values - d['X_t'].values.mean(), Zb, False) - hs(ef, Zf, False))
    nz = d['X_t'].values != 0; out['d'].append(hs(eb[nz], Zb[nz], False) - hs(ef[nz], Zf[nz], False) if nz.sum() > 8 else np.nan)
    sq = (eb[:, None] - eb[None, :]) ** 2; bwr.append(np.sqrt(np.median(sq[sq > 0]) / 2) / np.std(eb))
lab = {'a': 'baseline (median-heuristic bandwidth)', 'b': 'residual bandwidth fixed at its standard deviation', 'c': 'backward residual without a first stage', 'd': 'nonzero-shock dates only'}
save(pd.DataFrame([dict(variant=lab[k], mean_index=np.nanmean(v), share_positive=np.nanmean(np.array(v) > 0)) for k, v in out.items()] + [dict(variant='median-heuristic bandwidth / residual s.d. (mean)', mean_index=np.mean(bwr), share_positive=np.nan)]), 'mechanism_check.csv')
