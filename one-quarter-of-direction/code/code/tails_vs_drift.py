"""Two sources of residualizing error relative to the evaluation-block scale, crossed: scale drift and heavy tails.

One pair, T = 540, h = 3, two lags, end-of-sample split. The cause is i.i.d.: Gaussian or Student-t(2.5), multiplied by a
scale path that is constant or falls linearly from 1 to 0.25 over the sample. The outcome is an AR(1) independent of it.
Reported: rejection rates at 5 percent of the dependent wild bootstrap and of the enumerated rotation test, and the mean
share of the cause's sum of squares that falls in the evaluation block (the block holds about 16 percent of the observations).
    python code/tails_vs_drift.py [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flows'))
import flows_lib as FL                                  # loads the O(n^2) centering of the Gram matrix (checked against the engine)
from core_lib import *
import sparsity_check as SC
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300; T = 540; h = 3; rows = []
for tails in ('Gaussian', 'Student-t(2.5)'):
    for drift in ('constant scale', 'scale falls to 0.25'):
        rej_w = rej_r = 0; ss = []
        for rep in range(R):
            rng = np.random.RandomState(100 * rep + 7); z = rng.normal(size=T) if tails == 'Gaussian' else rng.standard_t(2.5, T) / np.sqrt(5.0)
            x = z * (np.linspace(1.0, 0.25, T) if drift != 'constant scale' else 1.0); y = np.zeros(T); e = rng.normal(size=T)
            for t in range(1, T): y[t] = 0.3 * y[t - 1] + e[t]
            d = pd.DataFrame({'x': x, 'y': y}); u = unit(d, h); _, pw, _ = wild_p(u['Q'], u['n'], 199, np.random.RandomState(rep))
            s1, X, Y = SC.split(d, h); obs = SC.stat(s1, X, Y, h); lo = h + P_LAGS + 2; ref = np.array([SC.stat(s1, X, np.roll(Y, k), h) for k in range(lo, len(X) - lo)])
            rej_w += pw <= 0.05; rej_r += (1 + np.sum(ref >= obs)) / (len(ref) + 1) <= 0.05; ss.append((X ** 2).sum() / (x ** 2).sum())
        rows.append(dict(tails=tails, scale=drift, reps=R, eval_share_of_ss=np.mean(ss), wild=100 * rej_w / R, rotation=100 * rej_r / R)); print(rows[-1], flush=True)
save(pd.DataFrame(rows), 'tails_vs_drift.csv')
