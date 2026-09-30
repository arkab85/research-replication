"""What time-averaging of the outcome does when the truth is a PURE IMPACT EFFECT.
Daily log price: random walk, 21 trading days per month, T = 551 months. In a month with an announcement (70 percent of
months) a shock X_m ~ N(0,1) arrives on a uniformly drawn day and moves the price permanently by beta * X_m on that day.
Nothing else: no delayed response, no volatility response. Two monthly outcome series are built from the same daily path:
  month-end : p(last day of m) - p(last day of m-1)          (the finance convention)
  averaged  : mean_d p(m,d) - mean_d p(m-1,d)                (Shiller's S&P 500 and yields; FRED/H.10 monthly rates)
For each: the local projection of Y_{m+h} on X_m (h = 0,1,3,6), and the index at h = 1,3,6 with the exact rotation test.
    python code/averaging_sim.py [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200; T, ND, beta = 551, 21, 0.5; g = np.random.RandomState(0)
def lp(x, y, h, L=2):
    n = len(y); t = np.arange(L, n - h); Z = np.column_stack([np.ones(len(t)), x[t]] + [x[t - l] for l in range(1, L + 1)] + [y[t - l] for l in range(1, L + 1)]); return np.linalg.lstsq(Z, y[t + h], rcond=None)[0][1]
def rot(d, hs=HS, p=P_LAGS):
    S = {h: SC.split(d, h) for h in hs}; span = {h: len(S[h][1]) - 2 * (h + p + 2) for h in hs}; K = max(span.values())
    obs = {h: SC.stat(*S[h], h) for h in hs}; M = {h: np.array([SC.stat(S[h][0], S[h][1], np.roll(S[h][2], h + p + 2 + int(round(j / (K - 1) * (span[h] - 1)))), h) for j in range(K)]) for h in hs}
    out = {f'p{h}': (1 + np.sum(M[h] >= obs[h])) / (K + 1) for h in hs}; out.update({f'dii{h}': obs[h] for h in hs})
    o = sum(HUMP[h] * obs[h] for h in hs); b = sum(HUMP[h] * M[h] for h in hs); out['p_hump'] = (1 + np.sum(b >= o)) / (K + 1); return out
rec = {'month-end': [], 'averaged': []}
for r in range(R):
    X = g.normal(size=T) * (g.rand(T) < 0.7); day = g.randint(0, ND, size=T); ret = g.normal(size=(T, ND)) / np.sqrt(ND); ret[np.arange(T), day] += beta * X
    p = np.cumsum(ret.ravel()).reshape(T, ND); Y = {'month-end': np.diff(p[:, -1], prepend=0.0), 'averaged': np.diff(p.mean(1), prepend=0.0)}
    for k, y in Y.items():
        d = pd.DataFrame({'x': X[1:], 'y': y[1:]}, index=np.arange(T - 1).astype(str)); o = rot(d); o.update({f'lp{h}': lp(d['x'].values, d['y'].values, h) for h in (0, 1, 3, 6)}); rec[k].append(o)
    if (r + 1) % 25 == 0: print(r + 1, {k: {q: round(float(np.mean([v[q] <= .05 for v in rec[k]])), 3) for q in ('p1', 'p3', 'p6', 'p_hump')} for k in rec}, flush=True)
rows = []
for k, v in rec.items():
    q = pd.DataFrame(v); rows.append(dict(outcome=k, reps=R, **{f'LP_h{h}': q[f'lp{h}'].mean() for h in (0, 1, 3, 6)}, **{f'index_h{h}': q[f'dii{h}'].mean() for h in HS}, **{f'reject05_h{h}': (q[f'p{h}'] <= .05).mean() for h in HS}, reject05_hump=(q['p_hump'] <= .05).mean()))
save(pd.DataFrame(rows), 'averaging_sim.csv')
