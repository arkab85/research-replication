"""No-channel audit with every ACTUAL identified-shock series in the role of the cause. The outcome is an AR(1) that is
independent of the shock, so every null is true. For each series: share of exact zeros, mean index, share positive, and
rejection rates at 5 percent of the wild bootstrap, the paired block bootstrap, their intersection, and the rotation test
(exact enumeration of the separated rotations). h = 3, two lags, honest split, exactly as in the application.
Monthly series: narrative, Jarocinski-Karadi, Bauer-Swanson, Kanzig oil, Swanson FFR / forward guidance / LSAP factors,
Bu-Rogers-Wu. Daily series: Jarocinski-Karadi on announcement days (h = 21, fewer replications, O(n^2) Gram routine).
    python code/audit_actual_series.py [R_monthly] [R_daily]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, numpy as np
def _cgram_fast(Z):
    Z = np.asarray(Z, float); Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z; sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0; K = np.exp(-d2 / (2 * (med / 2.0) + 1e-12)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
engine.cgram = _cgram_fast
from core_lib import *
import sparsity_check as SC
RM = int(sys.argv[1]) if len(sys.argv) > 1 else 200; RD = int(sys.argv[2]) if len(sys.argv) > 2 else 40
D = load_panel(); H = pd.read_csv(os.path.join(ROOT, 'holdout', 'holdout_panel.csv'), index_col=0); H.index = H.index.astype(str)
series = {'Narrative (Romer-Romer, Acosta)': D['rr'].loc['1974-02':'2019-12'], 'Jarocinski-Karadi': D['jk'].loc['1990-01':'2019-12'], 'Bauer-Swanson': D['bs'].loc['1988-02':'2019-12'], 'Kanzig oil supply news': D['oil'].loc['1975-01':'2019-12'],
          'Swanson federal funds rate factor': H['sw_ffr'].loc['1991-07':'2019-06'], 'Swanson forward guidance factor': H['sw_fg'].loc['1991-07':'2019-06'], 'Swanson LSAP factor': H['sw_lsap'].loc['1991-07':'2019-06'], 'Bu-Rogers-Wu': H['brw'].loc['1994-01':'2019-12']}
DL = pd.read_csv(os.path.join(ROOT, 'extension', 'daily_panel.csv'), index_col=0); daily = {'Jarocinski-Karadi, daily (announcement days)': DL['jk'].dropna()}
def ar1(T, g):
    e = g.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
    return Y
def rot_p(d, h, p, gap):
    s1, X, Y = SC.split(d, h, p, gap); lo = h + p + 2; obs = SC.stat(s1, X, Y, h, p); ks = range(lo, len(X) - lo) if len(X) < 200 else np.random.RandomState(7).randint(lo, len(X) - lo, size=99)
    b = np.array([SC.stat(s1, X, np.roll(Y, int(k)), h, p) for k in ks]); return (1 + np.sum(b >= obs)) / (len(b) + 1)
rows = []
for freq, S, R, h, gap, Bw, Bp in (('monthly', series, RM, 3, 6, 199, 99), ('daily', daily, RD, 21, 126, 199, 49)):
    for name, s in S.items():
        x = s.dropna().values; T = len(x); g = np.random.RandomState(0); out = []
        for r in range(R):
            d = pd.DataFrame({'x': x, 'y': ar1(T, g)}, index=np.arange(T).astype(str)); u = unit(d, h, p=2, gap=gap); pw, pp = cell_tests(u, Bw, Bp, seed=r); out.append((u['obs'], pw, pp, rot_p(d, h, 2, gap)))
        o = np.array(out); sd = np.std(x); rows.append(dict(frequency=freq, series=name, T=T, n_test=u['n'], zero_share=float(np.mean(x == 0)), near_zero_share=float(np.mean(np.abs(x) < 0.05 * sd)), kurtosis=float(pd.Series(x).kurt()), reps=R,
                                                         mean_index=o[:, 0].mean(), share_positive=(o[:, 0] > 0).mean(), wild_05=(o[:, 1] <= .05).mean(), paired_05=(o[:, 2] <= .05).mean(), both_05=(np.maximum(o[:, 1], o[:, 2]) <= .05).mean(), rotation_05=(o[:, 3] <= .05).mean()))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
save(pd.DataFrame(rows), 'audit_actual_series.csv')
