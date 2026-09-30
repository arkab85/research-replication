"""Size of the ORIGINAL pooled calibrations (dependent wild bootstrap, paired block bootstrap, intersection) with the
actual shock series and no channel: nine pairs from the narrative, Jarocinski-Karadi and oil series on their calendars
against three synthetic outcomes independent of every shock (AR(1), cross-correlation 0.5). Pooled index at h = 3 and the
hump, one calendar-indexed weight path and one set of block starts shared across pairs, as in the published tables.
    python code/pooled_bootstrap_size.py [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
R = int(sys.argv[1]) if len(sys.argv) > 1 else 100; D = load_panel(); cal = D.loc['1974-02':'2019-12'].index
shocks = {x: D[x].loc[START[x]:'2019-12'].dropna() for x in ('rr', 'jk', 'oil')}; out = []
for r in range(R):
    g = np.random.RandomState(5000 + r); f = g.normal(size=len(cal)); Ys = []
    for k in range(3):
        e = np.sqrt(.5) * f + np.sqrt(.5) * g.normal(size=len(cal)); y = np.zeros(len(cal))
        for t in range(1, len(cal)): y[t] = 0.3 * y[t - 1] + e[t]
        Ys.append(pd.Series(y, index=cal))
    ds = [pd.concat([s.rename('x'), y.rename('y')], axis=1).dropna() for s in shocks.values() for y in Ys]; U = {h: [unit(d, h) for d in ds] for h in HS}
    P3, pw3, pp3 = pooled(U[3], 199, 99, seed=r); H, pwH, ppH = pooled_contrast(U, HUMP, 199, 99, seed=r); out.append(dict(P3=P3, wild3=pw3, paired3=pp3, wildH=pwH, pairedH=ppH))
    if (r + 1) % 10 == 0: q = pd.DataFrame(out); print(r + 1, {k: round(float((q[k] <= .05).mean()), 3) for k in ('wild3', 'paired3', 'wildH', 'pairedH')}, 'both3', round(float(((q.wild3 <= .05) & (q.paired3 <= .05)).mean()), 3), 'mean P3', round(float(q.P3.mean()), 5), flush=True)
q = pd.DataFrame(out); rows = [dict(statistic='pooled index, h=3', wild_05=(q.wild3 <= .05).mean(), paired_05=(q.paired3 <= .05).mean(), both_05=((q.wild3 <= .05) & (q.paired3 <= .05)).mean(), wild_10=(q.wild3 <= .10).mean(), paired_10=(q.paired3 <= .10).mean(), both_10=((q.wild3 <= .10) & (q.paired3 <= .10)).mean()),
                           dict(statistic='hump', wild_05=(q.wildH <= .05).mean(), paired_05=(q.pairedH <= .05).mean(), both_05=((q.wildH <= .05) & (q.pairedH <= .05)).mean(), wild_10=(q.wildH <= .10).mean(), paired_10=(q.pairedH <= .10).mean(), both_10=((q.wildH <= .10) & (q.pairedH <= .10)).mean())]
save(pd.DataFrame(rows).assign(reps=R, mean_pooled_index=q.P3.mean()), 'pooled_bootstrap_size.csv')
