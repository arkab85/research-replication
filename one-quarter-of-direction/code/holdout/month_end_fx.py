"""The nineteen-currency result with MONTH-END exchange rates. The main analysis uses monthly AVERAGES of the H.10 daily
noon rates (the G.5 / FRED convention). An average-to-average change in month t+1 mechanically contains part of a level
shift that happened inside month t, so with averaged rates a pure impact effect shows up 'one month after the shock'.
Here the same design is run on last-business-day rates. Also the local projections under both conventions.
    python holdout/month_end_fx.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
import build_data as bd
D = load_panel(); E = D.copy(); eur = bd.h10_daily('eu')
for cur, code in {**bd.FLOAT9, **bd.EURO10}.items():
    s = bd.h10_daily(code); s = 1.0 / s if cur in bd.USD_PER_FX else s
    if cur in bd.EURO_CONV: s = pd.concat([s[:'1998-12-31'], (bd.EURO_CONV[cur] / eur)['1999-01-01':'2001-12-31']])
    m = s.groupby(s.index.to_period('M')).last(); m.index = m.index.astype(str); E['fx_' + cur] = (100 * np.log(m).diff()).reindex(E.index)
k19 = [('rr', 'fx_' + c) for c in FLOAT9 + EURO10]; k27 = [(x, 'fx_' + c) for x in ('rr', 'jk', 'bs') for c in FLOAT9]; rows = []
def lp(x, y, h, L=2):
    n = len(y); t = np.arange(L, n - h); Z = np.column_stack([np.ones(len(t)), x[t]] + [x[t - l] for l in range(1, L + 1)] + [y[t - l] for l in range(1, L + 1)]); return np.linalg.lstsq(Z, y[t + h], rcond=None)[0][1]
for tag, P in (('monthly averages (main analysis)', D), ('month-end rates', E)):
    report(rows, tag + ' | 19 currencies, 1974-2001', k19, end_split([sample(P, x, y, end='2001-12') for x, y in k19]), {'19 currencies': k19, '9 floaters': k19[:9], '10 pre-euro': k19[9:]})
    report(rows, tag + ' | 27 pairs, own samples', k27, end_split([sample(P, x, y) for x, y in k27]), {'27 currency pairs': k27, 'narrative x 9': k27[:9], 'Jarocinski-Karadi x 9': k27[9:18], 'Bauer-Swanson x 9': k27[18:]})
    b = {h: np.array([lp(*(lambda d: (d['x'].values, d['y'].values))(sample(P, 'rr', y)), h) for _, y in k19]) for h in (0, 1, 3, 6)}
    print(tag, '| mean local projection h=0,1,3,6:', [round(float(v.mean()), 2) for v in b.values()], '| positive of 19:', [int((v > 0).sum()) for v in b.values()], flush=True)
    rows.append(dict(block=tag + ' | local projection', set='mean beta h=0/1/3/6', statistic='LP', value=np.nan, **{f'lp{h}': float(b[h].mean()) for h in b}, **{f'lp_pos{h}': int((b[h] > 0).sum()) for h in b}))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'month_end_fx.csv'), index=False, float_format='%.6g')
for (bl, s), gq in R[R.statistic != 'LP'].groupby(['block', 'set'], sort=False):
    f = lambda st: gq[gq.statistic == st].iloc[0]; print(f"{bl[:48]:48s} {s:22s} | P1,3,6 = {f('P(1)').value:+.4f} {f('P(3)').value:+.4f} {f('P(6)').value:+.4f} | p1={f('P(1)').p_rotation:.3f} p3={f('P(3)').p_rotation:.3f} p6={f('P(6)').p_rotation:.3f} (min {f('P(1)').min_p:.3f}) pos={int(f('P(1)').positive)}/{int(f('P(3)').positive)}/{int(f('P(6)').positive)}")
