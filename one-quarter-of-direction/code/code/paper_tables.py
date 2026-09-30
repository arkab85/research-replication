"""Every rotation-test number used in the revised manuscript, by EXACT enumeration of the admissible rotations.

A test block of n rows admits the rotations k = lo, ..., n - lo - 1 with lo = h + p + 2 (the two series are kept at
least lo rows apart). With n between 50 and 85 there are only K = 40 to 70 of them, so a p-value from many random
draws would overstate its resolution. Here all K rotations are used once, p = (1 + #{T_k >= T_obs}) / (K + 1), and the
smallest attainable p-value, 1 / (K + 1), is reported next to every test. In pooled tests rotation j moves unit u by
lo_u + round(j / (K - 1) * (n_u - 2 lo_u - 1)) rows, K being the largest unit's count, so all units and all horizons
turn together and cross-unit and cross-horizon dependence is preserved.          python code/paper_tables.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *
import sparsity_check as SC
D = load_panel(); fx = lambda c: 'fx_' + c


def rot_matrix(ds, hs, p=P_LAGS, gap=GAP):
    """obs[h] (units,), M[h] (K, units): every unit at every horizon under the same K rotations."""
    S = {h: [SC.split(d, h, p, gap) for d in ds] for h in hs}; span = {h: [len(X) - 2 * (h + p + 2) for _, X, _ in S[h]] for h in hs}; K = max(max(v) for v in span.values())
    obs = {h: np.array([SC.stat(s1, X, Y, h, p) for s1, X, Y in S[h]]) for h in hs}; M = {}
    for h in hs:
        lo = h + p + 2; M[h] = np.array([[SC.stat(s1, X, np.roll(Y, lo + int(round(j / (K - 1) * (sp - 1)))), h, p) for (s1, X, Y), sp in zip(S[h], span[h])] for j in range(K)])
    return obs, M, K
def test(obs, M, K, cols, w):
    o = sum(w[h] * obs[h][cols].mean() for h in w); b = sum(w[h] * M[h][:, cols].mean(1) for h in w); return dict(value=o, mean_under_rotation=b.mean(), p_rotation=(1 + np.sum(b >= o)) / (K + 1), min_p=1 / (K + 1), rotations=K)
STATS = {'P(1)': {1: 1.}, 'P(3)': {3: 1.}, 'P(6)': {6: 1.}, 'hump': HUMP, 'P(3)-P(1)': D31, 'P(3)-P(6)': D36}
rows = []
def report(block, keys, ds, sets, stats=STATS, cells=False, **kw):
    obs, M, K = rot_matrix(ds, HS, **kw)
    if cells:
        for j, k in enumerate(keys):
            for h in HS: rows.append(dict(block=block, set=f'{k[0]} x {k[1]}', pairs=1, statistic=f'P({h})', positive=int(obs[h][j] > 0), **test(obs, M, K, [j], {h: 1.})))
    for name, ks in sets.items():
        c = [keys.index(k) for k in ks]
        for st, w in stats.items(): rows.append(dict(block=block, set=name, pairs=len(c), statistic=st, positive=int(sum(obs[h][c] > 0 for h in w if len(w) == 1).sum()) if len(w) == 1 else np.nan, **test(obs, M, K, c, w)))
    print(block, 'done; K =', K, flush=True)

# 1. the eighteen rule-based pairs
nine = [q for q in PV if q[0] != 'bs']; fam = lambda x: [q for q in PV if q[0] != x]
report('domestic', PAIRS18, [sample(D, x, y) for x, y in PAIRS18], {'eighteen rule-based pairs': PAIRS18, 'fifteen pairs excl. EBP': [q for q in PAIRS18 if q[1] != 'ebp'],
       'twelve pairs excl. Bauer-Swanson and EBP': [q for q in PAIRS18 if q[0] != 'bs' and q[1] != 'ebp'], 'nine price and volatility pairs': nine, 'twelve price and volatility pairs': PV,
       'six credit, EBP and dollar pairs': [q for q in PAIRS18 if q not in PV], 'PV, drop narrative': fam('rr'), 'PV, drop Jarocinski-Karadi': fam('jk'), 'PV, drop oil': fam('oil'),
       'PV, narrative only': [q for q in PV if q[0] == 'rr'], 'PV, Jarocinski-Karadi only': [q for q in PV if q[0] == 'jk'], 'PV, Bauer-Swanson only': [q for q in PV if q[0] == 'bs'], 'PV, oil only': [q for q in PV if q[0] == 'oil']}, cells=True)
# 2. currencies
k27 = [(x, fx(c)) for x in ('rr', 'jk', 'bs') for c in FLOAT9]
report('currencies, own samples', k27, [sample(D, x, y) for x, y in k27], {'narrative x 9 floaters': k27[:9], 'Jarocinski-Karadi x 9 floaters': k27[9:18], 'Bauer-Swanson x 9 floaters': k27[18:], 'all 27 currency pairs': k27}, cells=True)
k19 = [('rr', fx(c)) for c in FLOAT9 + EURO10]
report('currencies, common window 1974-2001', k19, [sample(D, x, y, end='2001-12') for x, y in k19], {'19 currencies': k19, '9 floaters': k19[:9], '10 pre-euro currencies': k19[9:]})
# 3. corrected baskets (all nineteen as foreign currency per dollar)
Bk = D.copy(); bsets = {'all 19 basket': FLOAT9 + EURO10, 'European basket': EURO10, 'non-European basket': ['AUD', 'CAD', 'NZD', 'JPY']}
for nm, cs in bsets.items(): Bk['bk_' + nm] = D[[fx(c) for c in cs]].mean(axis=1, skipna=True)
kb = [('rr', 'bk_' + nm) for nm in bsets]; report('baskets (corrected)', kb, [sample(Bk, x, y) for x, y in kb], {nm: [('rr', 'bk_' + nm)] for nm in bsets}, stats={k: STATS[k] for k in ('P(1)', 'P(3)', 'P(6)')})
# 4. reverse labelling: the asset price in the role of the cause, the shock in the role of the outcome
def rev(x, y, **kw): d = sample(D, x, y, **kw); return d.rename(columns={'x': 'y', 'y': 'x'})[['x', 'y']]
report('reverse labelling', PAIRS18, [rev(x, y) for x, y in PAIRS18], {'nine price and volatility pairs': nine, 'twelve price and volatility pairs': PV, 'eighteen rule-based pairs': PAIRS18})
report('reverse labelling, currencies 1974-2001', k19, [rev(x, y, end='2001-12') for x, y in k19], {'19 currencies': k19})
# 5. lag order
for p in (1, 3):
    report(f'lags={p}', PAIRS18, [sample(D, x, y) for x, y in PAIRS18], {'nine price and volatility pairs': nine, 'twelve price and volatility pairs': PV}, p=p)
    report(f'lags={p}, currencies 1974-2001', k19, [sample(D, x, y, end='2001-12') for x, y in k19], {'19 currencies': k19}, p=p)
# 6. extension samples (spread and dollar to 2019-08; floaters with shocks through 2024-01)
E = pd.read_csv(os.path.join(ROOT, 'extension', 'monthly_panel_ext.csv'), index_col=0); E.index = E.index.astype(str); E['ebp'] = D['ebp'].reindex(E.index); ymap = {'spread': 'spread_ext', 'dollar': 'dollar_ext'}
new = [('jk', 'spread'), ('jk', 'dollar'), ('bs', 'spread'), ('bs', 'dollar')]; p22 = PAIRS18 + new; smp = lambda P, x, y, a, b: pd.concat([P[x].rename('x'), P[y].rename('y')], axis=1).loc[a:b].dropna()
report('extension A: 22 pairs', p22, [smp(E, x, ymap.get(y, y), START[x], '2019-12') for x, y in p22], {'22 rule-based pairs': p22, '18 original pairs, longer spread and dollar samples': PAIRS18, '4 new pairs': new})
kB = [(x, fx(c)) for x in ('jk', 'bs') for c in FLOAT9]
report('extension B: floaters to 2024', kB, [smp(E, x + '_ext', y, {'jk': '1990-01', 'bs': '1988-02'}[x], '2024-01') for x, y in kB], {'Jarocinski-Karadi x 9 floaters': kB[:9], 'Bauer-Swanson x 9 floaters': kB[9:], 'both, 18 pairs': kB})
save(pd.DataFrame(rows), 'paper_rotation_tests.csv')
