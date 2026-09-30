"""Two simulations for the circular-shift (rotation) test.

  power : rejection rate at 5 and 10 percent when a channel is present, Y_t = 0.3 Y_{t-1} + s (X_{t-3}^2 - E X^2) + e_t,
          h = 3, T = 551, honest split; X dense, sparse (zero w.p. 0.85), or the actual narrative shock series.
  pooled: size of the POOLED tests (P(3), hump, P(3)-P(6)) on nine pairs built from the three actual shock series
          (narrative, Jarocinski-Karadi, oil) on their actual calendars and three synthetic outcomes that are
          independent of every shock (AR(1), cross-correlated 0.5 through a common factor), one rotation shared across
          pairs and horizons exactly as in title_test.py.

    python code/shift_power_and_pooled_size.py power [R]      python code/shift_power_and_pooled_size.py pooled [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC
mode = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 150; D = load_panel()

if mode == 'power':
    rr = sample(D, 'rr', 'sp500')['x'].values; T = 551; rows = []
    for name, draw in (('dense', lambda g: g.normal(size=T)), ('sparse p=0.15', lambda g: g.normal(size=T) * (g.rand(T) < .15)), ('actual narrative shock', lambda g: rr)):
        for s in (0.5, 1.0, 1.5):
            g = np.random.RandomState(1); ps, sh = [], []
            for r in range(R):
                X = draw(g); x2 = X ** 2 - np.mean(X ** 2); e = g.normal(size=T); Y = np.zeros(T)
                for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + (s * x2[t - 3] if t >= 3 else 0.0) + e[t]
                ps.append(SC.shift_cell(pd.DataFrame({'x': X, 'y': Y}), 3, 199, seed=r)[1]); sh.append(np.var(s * x2) / np.var(Y))
            ps = np.array(ps); rows.append(dict(design=name, s=s, variance_share=np.mean(sh), reps=R, power_05=(ps <= .05).mean(), power_10=(ps <= .10).mean())); print(rows[-1], flush=True)
    save(pd.DataFrame(rows), 'power_shift_test.csv')
else:
    cal = D.loc['1974-02':'2019-12'].index; shocks = {x: D[x].loc[START[x]:'2019-12'].dropna() for x in ('rr', 'jk', 'oil')}; B = 99; out = []
    for r in range(R):
        g = np.random.RandomState(1000 + r); f = g.normal(size=len(cal)); Ys = {}
        for k in range(3):
            e = np.sqrt(.5) * f + np.sqrt(.5) * g.normal(size=len(cal)); y = np.zeros(len(cal))
            for t in range(1, len(cal)): y[t] = 0.3 * y[t - 1] + e[t]
            Ys[k] = pd.Series(y, index=cal)
        ds = [pd.concat([s.rename('x'), Ys[k].rename('y')], axis=1).dropna() for s in shocks.values() for k in range(3)]
        O, M = {}, {}
        for h in HS: O[h], M[h] = SC.shift_matrix(ds, h, B, r)
        rec = {}
        for nm, w in (('P3', {3: 1.}), ('hump', HUMP), ('P3-P6', D36)):
            o = sum(w[h] * O[h].mean() for h in w); b = sum(w[h] * M[h].mean(1) for h in w); rec[nm] = (np.sum(b >= o) + 1) / (B + 1)
        rec['P3_value'] = O[3].mean(); out.append(rec)
        if (r + 1) % 10 == 0: q = pd.DataFrame(out); print(r + 1, 'reps | size at 5%:', {k: round(float((q[k] <= .05).mean()), 3) for k in ('P3', 'hump', 'P3-P6')}, '| mean pooled index under no channel:', round(float(q['P3_value'].mean()), 5), flush=True)
    q = pd.DataFrame(out); save(pd.DataFrame([dict(statistic=k, reps=R, size_05=(q[k] <= .05).mean(), size_10=(q[k] <= .10).mean(), mean_pooled_index_no_channel=q['P3_value'].mean()) for k in ('P3', 'hump', 'P3-P6')]), 'size_pooled_shift_tests.csv')
