"""Does the one-quarter hump depend on time-averaged outcomes? The main analysis takes the S&P 500 and the ten-year yield
from Shiller's file, which are MONTHLY AVERAGES of daily values, and averages the VIX within the month. Here the same
eighteen-pair design is run with MONTH-END values of the three price and volatility outcomes (Yahoo S&P 500 close, H.15
ten-year constant-maturity yield, CBOE VIX close), everything else unchanged. Not pre-specified: run after the hold-out
sets, which use month-end foreign indices, failed to show the hump.        python holdout/month_end_check.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
XR = os.path.join(ROOT, 'extension', 'data_raw'); D = load_panel(); E = D.copy()
j = json.load(open(os.path.join(XR, 'GSPC_yahoo.json')))['chart']['result'][0]; sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna()
h = pd.read_csv(os.path.join(XR, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h.columns if c.endswith('Y10_N.B')][0]; y10 = pd.to_numeric(h[c10], errors='coerce'); y10.index = pd.to_datetime(h.iloc[:, 0]); y10 = y10.dropna()
v = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'VIX_History.csv')); vix = pd.Series(v['CLOSE'].values, index=pd.to_datetime(v['DATE'], format='%m/%d/%Y'))
def me(s): m = s.groupby(s.index.to_period('M')).last(); m.index = m.index.astype(str); return m
E['sp500'] = (100 * np.log(me(sp)).diff()).reindex(E.index); E['dy10'] = me(y10).diff().reindex(E.index); E['vix'] = (100 * np.log(me(vix)).diff()).reindex(E.index)
print('correlation of month-end with monthly-average series:', {k: round(float(pd.concat([D[k], E[k]], axis=1).dropna().corr().iloc[0, 1]), 2) for k in ('sp500', 'dy10', 'vix')})
rows = []; nine = [q for q in PV if q[0] != 'bs']; sets = {'nine price and volatility pairs': nine, 'twelve price and volatility pairs': PV, 'eighteen rule-based pairs': PAIRS18,
        'Jarocinski-Karadi': [q for q in PV if q[0] == 'jk'], 'oil': [q for q in PV if q[0] == 'oil'], 'Bauer-Swanson': [q for q in PV if q[0] == 'bs'], 'narrative': [q for q in PV if q[0] == 'rr']}
for tag, P in (('monthly averages (main analysis)', D), ('month-end values', E)):
    report(rows, tag, PAIRS18, end_split([sample(P, x, y) for x, y in PAIRS18]), sets)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'month_end_check.csv'), index=False, float_format='%.6g')
for (bl, s), g in R.groupby(['block', 'set'], sort=False):
    f = lambda st: g[g.statistic == st].iloc[0]; print(f"{bl[:16]:16s} {s[:34]:34s} | P1,3,6 = {f('P(1)').value:+.4f} {f('P(3)').value:+.4f} {f('P(6)').value:+.4f} | p1={f('P(1)').p_rotation:.3f} p3={f('P(3)').p_rotation:.3f} p6={f('P(6)').p_rotation:.3f} hump={f('hump').p_rotation:.3f} 3-6={f('P(3)-P(6)').p_rotation:.3f}")
