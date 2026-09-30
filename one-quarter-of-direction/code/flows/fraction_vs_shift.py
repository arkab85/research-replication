"""Common FRACTION against common calendar SHIFT across horizons (PROTOCOL 9.4), on simulated outcomes only.
For each replication: actual flows, simulated independent outcomes; every admissible shift is computed once per horizon
and the reference draws of the average over horizons (A) and of the contrast (H or C) are formed both ways.
Reported: the standard deviation of the reference draws under each scheme, and the mean correlation across horizons.
    python flows/fraction_vs_shift.py [R]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from run_flows import with_outcomes
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10; rows = []
for name in ('set2',):
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); p = 2
    for rep in range(R):
        P = with_outcomes(W, mk, sim_outcomes(len(W), len(mk), np.random.RandomState(9000 + rep))); U = make_units(P, mk, hs, cfg['gap'], 'FR'); n = len(U[hs[0]][0]['Xte'])
        full = {h: {k: np.mean([stat(u['s1'], u['Xte'], np.roll(u['Yte'], k), h, p) for u in U[h]]) for k in range(h + p + 2, n - (h + p + 2))} for h in hs}
        lo = max(hs) + p + 2; shift = {h: np.array([full[h][k] for k in range(lo, n - lo)]) for h in hs}
        span = {h: n - 2 * (h + p + 2) for h in hs}; K = max(span.values()); frac = {h: np.array([full[h][h + p + 2 + int(round(j / (K - 1) * (span[h] - 1)))] for j in range(K)]) for h in hs}
        for scheme, M in (('common shift', shift), ('common fraction', frac)):
            cc = np.corrcoef(np.array([M[h] for h in hs])); r = dict(set=name, rep=rep, scheme=scheme, mean_corr_across_horizons=float(cc[np.triu_indices(len(hs), 1)].mean()))
            for st in ('A', 'H' if 'H' in S else 'C'): r['sd_' + st] = float(np.std(sum(S[st][h] * M[h] for h in S[st])))
            rows.append(r)
        print('rep', rep, flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'fraction_vs_shift.csv'), index=False, float_format='%.6g'); print(D.groupby(['set', 'scheme']).mean(numeric_only=True).drop(columns='rep'))
