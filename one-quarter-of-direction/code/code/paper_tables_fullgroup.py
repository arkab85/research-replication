"""Headline rotation tests with the FULL cyclic group (every rotation k = 1, ..., n-1), for which the randomization
test is exact under rotation invariance of the outcome (Lehmann and Romano 2005, Theorem 15.2.1). The main tables drop
the rotations within h+p+2 rows of the identity, which contain the alternative at neighbouring horizons; this file shows
the conclusions do not depend on that choice.        python code/paper_tables_fullgroup.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC
D = load_panel(); rows = []
def full(block, keys, ds, sets, stats):
    S = {h: [SC.split(d, h) for d in ds] for h in HS}; K = max(len(X) for h in HS for _, X, _ in S[h]) - 1
    obs = {h: np.array([SC.stat(s1, X, Y, h) for s1, X, Y in S[h]]) for h in HS}
    M = {h: np.array([[SC.stat(s1, X, np.roll(Y, 1 + int(round(j / (K - 1) * (len(X) - 2)))), h) for s1, X, Y in S[h]] for j in range(K)]) for h in HS}
    for name, ks in sets.items():
        c = [keys.index(k) for k in ks]
        for st, w in stats.items():
            o = sum(w[h] * obs[h][c].mean() for h in w); b = sum(w[h] * M[h][:, c].mean(1) for h in w); rows.append(dict(block=block, set=name, statistic=st, value=o, p_full_group=(1 + np.sum(b >= o)) / (K + 1), rotations=K))
    print(block, 'done', flush=True)
ST = {'P(1)': {1: 1.}, 'P(3)': {3: 1.}, 'P(6)': {6: 1.}, 'hump': HUMP, 'P(3)-P(6)': D36}
full('domestic', PAIRS18, [sample(D, x, y) for x, y in PAIRS18], {'nine price and volatility pairs': [q for q in PV if q[0] != 'bs'], 'twelve price and volatility pairs': PV, 'eighteen rule-based pairs': PAIRS18}, ST)
k19 = [('rr', 'fx_' + c) for c in FLOAT9 + EURO10]; full('currencies 1974-2001', k19, [sample(D, x, y, end='2001-12') for x, y in k19], {'19 currencies': k19}, ST)
save(pd.DataFrame(rows), 'paper_rotation_tests_fullgroup.csv')
