"""Size of the ENUMERATED rotation test (the version used in the tables) with many replications, and its power.
  size   : each actual monthly shock series as the cause of an independent AR(1); separated rotations and the full
           group; attainable level reported.                      python code/rotation_size_large.py size <part 0|1> [R]
  daily  : the daily Jarocinski-Karadi series, 99 random rotations. python code/rotation_size_large.py daily [R]
  power  : the designs of power_shift_test.csv, enumerated rotations, plus an impact effect with a delayed channel.
                                                                   python code/rotation_size_large.py power [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, engine
def _fast(Z):
    Z = np.asarray(Z, float); Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z; sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0; K = np.exp(-d2 / (2 * (med / 2.0) + 1e-12)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
engine.cgram = _fast
from core_lib import *
import sparsity_check as SC
mode = sys.argv[1]
def ar1(T, g, x=None, beta=0.0, s=0.0):
    e = g.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t] + (beta * x[t] if x is not None else 0.0) + (s * (x[t - 3] ** 2 - np.mean(x ** 2)) if (x is not None and s and t >= 3) else 0.0)
    return Y
def rot_both(d, h=3, p=2, gap=6):
    s1, X, Y = SC.split(d, h, p, gap); n = len(X); lo = h + p + 2; obs = SC.stat(s1, X, Y, h, p); b = np.array([SC.stat(s1, X, np.roll(Y, k), h, p) for k in range(1, n)])
    sep = b[lo - 1: n - lo - 1]; return (1 + np.sum(sep >= obs)) / (len(sep) + 1), (1 + np.sum(b >= obs)) / (len(b) + 1), len(sep), len(b)
D = load_panel()
if mode == 'size':
    part = int(sys.argv[2]); R = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; H = pd.read_csv(os.path.join(ROOT, 'holdout', 'holdout_panel.csv'), index_col=0); H.index = H.index.astype(str)
    S = [('Narrative (Romer-Romer, Acosta)', D['rr'].loc['1974-02':'2019-12']), ('Jarocinski-Karadi', D['jk'].loc['1990-01':'2019-12']), ('Bauer-Swanson', D['bs'].loc['1988-02':'2019-12']), ('Kanzig oil supply news', D['oil'].loc['1975-01':'2019-12']),
         ('Swanson federal funds rate factor', H['sw_ffr'].loc['1991-07':'2019-06']), ('Swanson forward guidance factor', H['sw_fg'].loc['1991-07':'2019-06']), ('Swanson LSAP factor', H['sw_lsap'].loc['1991-07':'2019-06']), ('Bu-Rogers-Wu', H['brw'].loc['1994-01':'2019-12'])][part::2]; rows = []
    for name, s in S:
        x = s.dropna().values; T = len(x); g = np.random.RandomState(11); o = np.array([rot_both(pd.DataFrame({'x': x, 'y': ar1(T, g)})) for r in range(R)]); Ks, Kf = int(o[0, 2]), int(o[0, 3])
        rows.append(dict(series=name, reps=R, separated_05=(o[:, 0] <= .05).mean(), full_group_05=(o[:, 1] <= .05).mean(), separated_10=(o[:, 0] <= .10).mean(), full_group_10=(o[:, 1] <= .10).mean(), K_separated=Ks, K_full=Kf,
                         attainable_05_separated=np.floor(.05 * (Ks + 1)) / (Ks + 1), attainable_05_full=np.floor(.05 * (Kf + 1)) / (Kf + 1))); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'rotation_size_large_part{part}.csv'), index=False)
elif mode == 'daily':
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 200; DL = pd.read_csv(os.path.join(ROOT, 'extension', 'daily_panel.csv'), index_col=0); x = DL['jk'].dropna().values; T = len(x); g = np.random.RandomState(11); ps = []
    for r in range(R):
        d = pd.DataFrame({'x': x, 'y': ar1(T, g)}); s1, X, Y = SC.split(d, 21, 2, 126); lo = 25; obs = SC.stat(s1, X, Y, 21, 2); b = np.array([SC.stat(s1, X, np.roll(Y, int(k)), 21, 2) for k in g.randint(lo, len(X) - lo, size=99)]); ps.append((1 + np.sum(b >= obs)) / 100)
        if (r + 1) % 20 == 0: print(r + 1, 'size at 5%:', np.mean(np.array(ps) <= .05), flush=True)
    pd.DataFrame([dict(series='Jarocinski-Karadi, daily', reps=R, rotation_05=np.mean(np.array(ps) <= .05), rotation_10=np.mean(np.array(ps) <= .10))]).to_csv(os.path.join(RES, 'rotation_size_daily.csv'), index=False)
else:
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 300; rr = sample(D, 'rr', 'sp500')['x'].values; jk = sample(D, 'jk', 'sp500')['x'].values; rows = []
    designs = [('dense', lambda g: g.normal(size=551)), ('sparse p=0.15', lambda g: g.normal(size=551) * (g.rand(551) < .15)), ('actual narrative shock', lambda g: rr), ('actual Jarocinski-Karadi shock', lambda g: jk / jk.std())]
    for name, draw in designs:
        for beta, s in ((0.0, 0.5), (0.0, 1.0), (0.0, 1.5), (0.5, 0.0), (0.5, 1.0)):
            g = np.random.RandomState(3); ps, sh = [], []
            for r in range(R):
                X = draw(g); Y = ar1(len(X), g, X, beta, s); ps.append(rot_both(pd.DataFrame({'x': X, 'y': Y}))[0]); sh.append(np.var(s * (X ** 2 - np.mean(X ** 2))) / np.var(Y))
            rows.append(dict(design=name, impact_beta=beta, channel_s=s, channel_variance_share=np.mean(sh), reps=R, reject_05=np.mean(np.array(ps) <= .05), reject_10=np.mean(np.array(ps) <= .10))); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'rotation_power_enumerated.csv'), index=False)
