"""NOT pre-specified. Sets B and C of holdout/PROTOCOL.md were run on the time-averaged outcomes of the main panel
(Shiller S&P 500 and ten-year yield; monthly-average H.10 rates). This reruns them with month-end values, reruns Set A
without the narrative shock (whose evaluation block holds 2 percent of its variation), and writes the share of each
series' sum of squares that falls in its evaluation block.        python holdout/holdout_month_end.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
import build_data as bd
P = pd.read_csv(os.path.join(HERE, 'holdout_panel.csv'), index_col=0); P.index = P.index.astype(str); XR = os.path.join(ROOT, 'extension', 'data_raw')
j = json.load(open(os.path.join(XR, 'GSPC_yahoo.json')))['chart']['result'][0]; sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna()
h = pd.read_csv(os.path.join(XR, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h.columns if c.endswith('Y10_N.B')][0]; y10 = pd.to_numeric(h[c10], errors='coerce'); y10.index = pd.to_datetime(h.iloc[:, 0]); y10 = y10.dropna()
def me(s): m = s.groupby(s.index.to_period('M')).last(); m.index = m.index.astype(str); return m
E = P.copy(); E['sp500'] = (100 * np.log(me(sp)).diff()).reindex(E.index); E['dy10'] = me(y10).diff().reindex(E.index)
EM = {'MXN': 'mx', 'KRW': 'ko', 'ZAR': 'sf', 'INR': 'in', 'THB': 'th', 'SGD': 'si'}; old = bd.H10; bd.H10 = os.path.join(HERE, 'data_raw', 'h10')
for cur, code in EM.items(): E['fx_' + cur] = (100 * np.log(me(bd.h10_daily(code))).diff()).reindex(E.index)
bd.H10 = old; ST = dict(START, sw_ffr='1991-07', sw_fg='1991-07', sw_lsap='1991-07', brw='1994-01'); smp = lambda Q, x, y: pd.concat([Q[x].rename('x'), Q[y].rename('y')], axis=1).loc[ST[x]:'2019-12'].dropna(); rows = []
Bp = [(x, y) for x in ('sw_ffr', 'sw_fg', 'sw_lsap', 'brw') for y in ('sp500', 'dy10')]; C = [(x, 'fx_' + c) for x in ('jk', 'bs') for c in EM]
IDX = ['FTSE', 'DAX', 'CAC', 'NIKKEI', 'HSI', 'TSX', 'AORD', 'SMI']; A2 = [(x, 'eq_' + k) for x in ('jk', 'bs') for k in IDX]
report(rows, 'B at month-end (not pre-specified)', Bp, end_split([smp(E, x, y) for x, y in Bp]), {'Swanson factors and BRW x S&P 500, 10y yield': Bp})
report(rows, 'C at month-end (not pre-specified)', C, end_split([smp(E, x, y) for x, y in C]), {'two high-frequency shocks x six EM currencies': C})
report(rows, 'A without the narrative shock (not pre-specified)', A2, end_split([smp(P, x, y) for x, y in A2]), {'two high-frequency shocks x eight foreign indices': A2})
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'holdout_month_end.csv'), index=False, float_format='%.6g')
for (bl, s), gq in R.groupby(['block', 'set'], sort=False):
    f = lambda st: gq[gq.statistic == st].iloc[0]; print(f"{bl[:50]:50s} | P1,3,6 = {f('P(1)').value:+.4f} {f('P(3)').value:+.4f} {f('P(6)').value:+.4f} | p1={f('P(1)').p_rotation:.3f} p3={f('P(3)').p_rotation:.3f} p6={f('P(6)').p_rotation:.3f} hump={f('hump').p_rotation:.3f} (min {f('hump').min_p:.3f})")
D = load_panel(); ser = {'rr': D['rr'].loc['1974-02':'2019-12'], 'jk': D['jk'].loc['1990-01':'2019-12'], 'bs': D['bs'].loc['1988-02':'2019-12'], 'oil': D['oil'].loc['1975-01':'2019-12'], 'sw_ffr': P['sw_ffr'].loc['1991-07':'2019-06'], 'sw_fg': P['sw_fg'].loc['1991-07':'2019-06'], 'sw_lsap': P['sw_lsap'].loc['1991-07':'2019-06'], 'brw': P['brw'].loc['1994-01':'2019-12']}
out = []
for k, s in ser.items():
    s = s.dropna(); T = len(s); n = int(np.floor(T / np.log(T))); b = s.iloc[-n:]; out.append(dict(series=k, T=T, n_T=n, block_from=b.index[0], block_to=b.index[-1], share_of_months=n / T, share_of_sum_of_squares=float((b ** 2).sum() / (s ** 2).sum()), zeros_in_block=float((b == 0).mean())))
pd.DataFrame(out).to_csv(os.path.join(ROOT, 'results', 'evaluation_block_share.csv'), index=False); print(pd.DataFrame(out).to_string(index=False))
