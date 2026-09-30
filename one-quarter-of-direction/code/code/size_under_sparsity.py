"""Null behaviour of the index and of both tests when the shock is sparse and there is NO channel.

Y is an AR(1) independent of X. Designs: dense Gaussian X; X zero with probability 0.85; the actual narrative shock
series; and a daily-like design (X nonzero on 3 percent of dates, T = 3000). Reports the mean index, the share of
positive estimates, and rejection rates of the wild, paired and intersection tests at the 5 and 10 percent levels.

    python code/size_under_sparsity.py [R]      # default R = 200 replications (R/2 for the daily-like design)"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D = load_panel(); rr = sample(D, 'rr', 'sp500')['x'].values
def ar1y(T, rng):
    e = rng.normal(size=T); Y = np.zeros(T)
    for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + e[t]
    return Y
DESIGNS = {'dense, T=551': (551, lambda T, rng: rng.normal(size=T), R), 'sparse p=0.15, T=551': (551, lambda T, rng: rng.normal(size=T) * (rng.rand(T) < .15), R),
           'actual narrative shock, T=551': (551, lambda T, rng: rr, R), 'daily-like p=0.03, T=3000': (3000, lambda T, rng: rng.normal(size=T) * (rng.rand(T) < .03), max(R // 2, 20))}
rows = []
for name, (T, draw, reps) in DESIGNS.items():
    rng = np.random.RandomState(0); out = []
    for r in range(reps):
        d = pd.DataFrame({'x': draw(T, rng), 'y': ar1y(T, rng)}, index=np.arange(T).astype(str)); u = unit(d, 3); pw, pp = cell_tests(u, 199, 99, seed=r); out.append((u['obs'], pw, pp, u['diag']))
    o = np.array(out); both = np.maximum(o[:, 1], o[:, 2])
    rows.append(dict(design=name, reps=reps, n_test=u['n'], mean_DII=o[:, 0].mean(), share_positive=(o[:, 0] > 0).mean(), mean_n_rho_f=o[:, 3].mean(),
                     wild_05=(o[:, 1] <= .05).mean(), wild_10=(o[:, 1] <= .10).mean(), paired_05=(o[:, 2] <= .05).mean(), paired_10=(o[:, 2] <= .10).mean(), inter_05=(both <= .05).mean(), inter_10=(both <= .10).mean()))
    print(rows[-1], flush=True)
save(pd.DataFrame(rows), 'size_under_sparsity.csv')
