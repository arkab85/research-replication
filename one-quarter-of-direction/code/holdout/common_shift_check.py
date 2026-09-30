"""The pooled rotation tests of the event-shock application with ONE calendar shift for every pair and horizon.

paper_tables.py and month_end_check.py turn each horizon by a common fraction of its admissible range, so a statistic
that combines horizons (the hump, P(3)-P(6)) is compared with draws in which the horizons sit at different shifts. That
leaves each single-horizon test unchanged and makes cross-horizon contrasts conservative. Here every pair and every
horizon is rotated by the same number of months k, max(h)+p+2 <= k < n_min-max(h)-p-2, n_min the shortest evaluation
block in the set (all blocks end in December 2019, so rows that do not wrap share the calendar shift).
Not pre-specified: added after a code review of the flows application found the issue.     python holdout/common_shift_check.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
XR = os.path.join(ROOT, 'extension', 'data_raw'); D = load_panel(); E = D.copy()
j = json.load(open(os.path.join(XR, 'GSPC_yahoo.json')))['chart']['result'][0]; sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna()
h15 = pd.read_csv(os.path.join(XR, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h15.columns if c.endswith('Y10_N.B')][0]; y10 = pd.to_numeric(h15[c10], errors='coerce'); y10.index = pd.to_datetime(h15.iloc[:, 0]); y10 = y10.dropna()
v = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'VIX_History.csv')); vix = pd.Series(v['CLOSE'].values, index=pd.to_datetime(v['DATE'], format='%m/%d/%Y'))
def me(s): m = s.groupby(s.index.to_period('M')).last(); m.index = m.index.astype(str); return m
E['sp500'] = (100 * np.log(me(sp)).diff()).reindex(E.index); E['dy10'] = me(y10).diff().reindex(E.index); E['vix'] = (100 * np.log(me(vix)).diff()).reindex(E.index)
nine = [q for q in PV if q[0] != 'bs']; SETS_ = {'nine price and volatility pairs': nine, 'twelve price and volatility pairs': PV, 'eighteen rule-based pairs': PAIRS18}; rows = []
IP = len(sys.argv) > 1 and sys.argv[1] == 'ip'      # also the impact-preserving rotation (flows/PROTOCOL.md, Section 10); run only after its Step 0c audit
def split_same_period(X, Y): Z = np.column_stack([np.ones(len(X)), X]); fit = Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]; return fit, Y - fit
for tag, P in (('monthly averages', D), ('month-end values', E)):
    for p in (2, 1, 3):
        S = end_split([sample(P, x, y) for x, y in PAIRS18], HS, p)
        for name, ks in SETS_.items():
            c = [PAIRS18.index(k) for k in ks]; nmin = min(len(S[1][i][1]) for i in c); lo = max(HS) + p + 2; shifts = range(lo, nmin - lo)
            obs = {h: np.array([SC.stat(*S[h][i], h, p) for i in c]) for h in HS}
            M = {h: np.array([[SC.stat(S[h][i][0], S[h][i][1], np.roll(S[h][i][2], k), h, p) for i in c] for k in shifts]) for h in HS}
            if IP:
                sp = {i: split_same_period(S[1][i][1], S[1][i][2]) for i in c}
                M2 = {h: np.array([[SC.stat(S[h][i][0], S[h][i][1], sp[i][0] + np.roll(sp[i][1], k), h, p) for i in c] for k in shifts]) for h in HS}
            for st, w in STATS.items():
                r = dict(outcomes=tag, lags=p, set=name, pairs=len(c), statistic=st, **pvalue(obs, M, len(shifts), slice(None), w))
                if IP: q = pvalue(obs, M2, len(shifts), slice(None), w); r['mean_under_rotation_ip'] = q['mean_under_rotation']; r['p_rotation_ip'] = q['p_rotation']
                rows.append(r)
        print(tag, 'lags', p, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'common_shift_check.csv'), index=False, float_format='%.6g')
pd.set_option('display.width', 220); print(R[R.statistic.isin(['P(3)', 'hump', 'P(3)-P(6)'])].to_string(index=False))
