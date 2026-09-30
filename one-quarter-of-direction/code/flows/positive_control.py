"""Positive control (designed after the primary analysis; PROTOCOL.md Section 11): can the calibrated procedure detect a
channel that is known to exist, on the same markets, calendar, split and tests?

Cause: the weekly return of each of the twelve markets. Outcome: the log of the market's realized variance in the week
(sum of squared daily changes between report dates). That large returns, and in equities negative returns, raise later
volatility is among the best-documented nonlinear lagged relations in finance. The reverse labelling, volatility to later
returns, is the negative control. Reported: plain rotation, impact-preserving rotation with a linear same-period fit, and
with the sieve same-period fit (the same-period relation of volatility to the return is V-shaped, not linear).
    python flows/positive_control.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
import build_flows_data as B
W, cfg = load('set1'); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); cal = pd.DatetimeIndex(pd.to_datetime(W.index)); P = B.cash(); V = pd.DataFrame(index=W.index)
for m in mk:
    s = P[m]; d = (s.diff() if m in B.UST else 100 * np.log(s).diff()).dropna(); wk = pd.Series(np.searchsorted(cal.values, d.index.values, side='left'), index=d.index)     # week = first report date on or after the day
    rv = (d ** 2).groupby(wk).sum(); rv = rv[(rv.index > 0) & (rv.index < len(cal))]      # week 0 has no earlier report date to start from
    V['f_' + m] = W['r_' + m].values; V['r_' + m] = np.log(rv.reindex(range(len(cal))).values + 1e-8)
V = V.replace([np.inf, -np.inf], np.nan).dropna(); print('panel', V.shape, V.index[0], V.index[-1]); dates = pd.to_datetime(V.index).values; rows = []
for d, lab in (('FR', 'returns to volatility (positive control)'), ('RF', 'volatility to returns (negative control)')):
    U = make_units(V, mk, hs, cfg['gap'], d, dates=dates); R_ = {sch: rotate(U, hs, 2, ip=ip) for sch, ip in (('plain', False), ('impact-preserving, linear', True), ('impact-preserving, sieve', 'sieve'))}
    for g, ms in [('all twelve', mk)] + list(cfg['groups'].items()):
        c = [mk.index(x) for x in ms]
        for st, w in S.items():
            r = dict(direction=lab, group=g, markets=len(c), statistic=st)
            for sch, (obs, M, K) in R_.items():
                o, mu, p = rot_p(obs, M, K, w, c); r['value'] = o; r['mean_rot_' + sch] = mu; r['p_' + sch] = p
            r['positive'] = int((R_['plain'][0][int(st[2:-1])][c] > 0).sum()) if st.startswith('P(') else np.nan; r['min_p'] = 1 / (K + 1); rows.append(r)
    print(lab, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'positive_control.csv'), index=False, float_format='%.6g'); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(D[D.group == 'all twelve'][['direction', 'statistic', 'value', 'positive', 'p_plain', 'p_impact-preserving, linear', 'p_impact-preserving, sieve', 'min_p']].to_string(index=False))
print(D[(D.group != 'all twelve') & D.statistic.isin(['P(1)', 'A'])][['direction', 'group', 'statistic', 'value', 'p_plain', 'p_impact-preserving, linear', 'p_impact-preserving, sieve']].to_string(index=False))
