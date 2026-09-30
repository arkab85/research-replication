"""Follow-up to month_end_check.py: (i) switch one outcome at a time to month-end values; (ii) lag orders 1 and 3 with all
three outcomes at month-end; (iii) the reverse: foreign indices cannot be averaged (monthly closes only), so instead the
S&P 500 alone, averaged vs month-end, against all four shocks.     python holdout/month_end_check2.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
XR = os.path.join(ROOT, 'extension', 'data_raw'); D = load_panel()
j = json.load(open(os.path.join(XR, 'GSPC_yahoo.json')))['chart']['result'][0]; sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna()
h = pd.read_csv(os.path.join(XR, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h.columns if c.endswith('Y10_N.B')][0]; y10 = pd.to_numeric(h[c10], errors='coerce'); y10.index = pd.to_datetime(h.iloc[:, 0]); y10 = y10.dropna()
v = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'VIX_History.csv')); vix = pd.Series(v['CLOSE'].values, index=pd.to_datetime(v['DATE'], format='%m/%d/%Y'))
def me(s): m = s.groupby(s.index.to_period('M')).last(); m.index = m.index.astype(str); return m
def av(s): m = s.groupby(s.index.to_period('M')).mean(); m.index = m.index.astype(str); return m
ME = {'sp500': 100 * np.log(me(sp)).diff(), 'dy10': me(y10).diff(), 'vix': 100 * np.log(me(vix)).diff()}
AV = {'sp500': 100 * np.log(av(sp)).diff(), 'dy10': av(y10).diff()}       # my own monthly averages of the same daily sources, to separate 'averaging' from 'Shiller vs Yahoo/H.15'
nine = [q for q in PV if q[0] != 'bs']; rows = []
def run(tag, repl, p=2):
    E = D.copy()
    for k, s in repl.items(): E[k] = s.reindex(E.index)
    report(rows, tag, nine, end_split([sample(E, x, y) for x, y in nine], p=p), {'nine price and volatility pairs': nine}, p=p)
run('baseline (Shiller averages, VIX average)', {})
run('own monthly averages of daily S&P and 10y', AV)
for k in ME: run(f'only {k} at month-end', {k: ME[k]})
run('all three at month-end', ME); run('all three at month-end, one lag', ME, p=1); run('all three at month-end, three lags', ME, p=3)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'month_end_check2.csv'), index=False, float_format='%.6g')
for bl, g in R.groupby('block', sort=False):
    f = lambda st: g[g.statistic == st].iloc[0]; print(f"{bl[:44]:44s} | P1,3,6 = {f('P(1)').value:+.4f} {f('P(3)').value:+.4f} {f('P(6)').value:+.4f} | p1={f('P(1)').p_rotation:.3f} p3={f('P(3)').p_rotation:.3f} p6={f('P(6)').p_rotation:.3f} hump={f('hump').p_rotation:.3f}")
