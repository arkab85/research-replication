"""The title's claim under a test that is valid for sparse shocks: is the footprint at three months, and is it
concentrated there?  Hump H = P(3) - [P(1) + P(6)] / 2 and the two differences, with the circular-shift reference
distribution; one rotation fraction per draw is shared by all units AND all horizons (same seed), so the dependence
across horizons is preserved, as in Table 5.        python code/title_test.py [B]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC
B = int(sys.argv[1]) if len(sys.argv) > 1 else 999; D = load_panel(); keys = PAIRS18 + [q for q in PV if q not in PAIRS18]
ds = [sample(D, x, y) for x, y in keys]; O, M = {}, {}
for h in HS: O[h], M[h] = SC.shift_matrix(ds, h, B, 0); print(f'h={h} done', flush=True)
sets = {'nine price and volatility pairs': [q for q in PV if q[0] != 'bs'], 'twelve price and volatility pairs': PV, 'eighteen rule-based pairs': PAIRS18,
        'six pairs that are not price or volatility': [q for q in PAIRS18 if q not in PV], 'oil pairs (dense shock)': [q for q in PAIRS18 if q[0] == 'oil'],
        'Jarocinski-Karadi pairs': [q for q in PAIRS18 if q[0] == 'jk'], 'Bauer-Swanson pairs': [q for q in PAIRS18 if q[0] == 'bs'], 'narrative pairs': [q for q in PAIRS18 if q[0] == 'rr']}
stats = {'P(1)': {1: 1.}, 'P(3)': {3: 1.}, 'P(6)': {6: 1.}, 'hump H': HUMP, 'P(3)-P(1)': D31, 'P(3)-P(6)': D36}; rows = []
for name, pairs in sets.items():
    c = [keys.index(q) for q in pairs]
    for st, w in stats.items():
        o = sum(w[h] * O[h][c].mean() for h in w); b = sum(w[h] * M[h][:, c].mean(1) for h in w)
        rows.append(dict(set=name, pairs=len(c), statistic=st, value=o, mean_under_rotation=b.mean(), excess=o - b.mean(), p_shift=(np.sum(b >= o) + 1) / (B + 1)))
save(pd.DataFrame(rows), 'title_test_shift.csv')
