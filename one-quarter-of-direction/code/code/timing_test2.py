r"""Executes code/TIMING_TEST_2.md, anchored 2026-09-24T03:23:48Z before this was run.

Same hypothesis as timing_test.py, on the design the manuscript already documents as having an
identified impact effect: the Romer-Romer narrative shock at its FOMC meeting dates against the
nineteen currencies of Table 12, 1974-2001.

    python timing_test2.py <path to One_Quarter_RFS_Replication>
"""
import sys, os, numpy as np, pandas as pd
import statsmodels.api as sm

PKG = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__.strip().splitlines()[4].strip())
sys.path.insert(0, os.path.join(PKG, 'code'))
import build_data as bd

# ---------- the shock, at its meeting dates ----------
A = pd.read_csv(os.path.join(PKG, 'data', 'raw', 'acosta_rrshocks.csv'), parse_dates=['fomc'])
A = A[A.rr_update.notna() & (A.rr_update != 0)].copy()
A['ym'] = A.fomc.dt.to_period('M')
cnt = A.groupby('ym').size()
adm = cnt[cnt == 1].index
A = A[A.ym.isin(adm)].set_index('ym')
print('narrative shock: %d meetings with a nonzero shock, %d months admitted (one meeting), %d excluded'
      % (len(cnt), len(adm), int((cnt >= 2).sum())))

# ---------- H.10 business-day calendar, and w ----------
cal = bd.h10_daily('al').index                       # the H.10 daily calendar
cal = pd.DatetimeIndex(sorted(set(cal)))
cdf = pd.DataFrame({'date': cal}); cdf['ym'] = cdf.date.dt.to_period('M')
w = {}
for ym, blk in cdf.groupby('ym'):
    if ym not in A.index:
        continue
    days = blk.date.reset_index(drop=True)
    md = A.loc[ym, 'fomc']
    nxt = days[days >= md]
    if len(nxt) == 0:
        continue
    d = int(nxt.index[0]) + 1                        # 1-indexed position among business days
    Dm = len(days)
    w[ym] = dict(s=float(A.loc[ym, 'rr_update']), D=Dm, d=d, w=(Dm - d + 1) / Dm)
W = pd.DataFrame(w).T
W.index = W.index.astype(str)
print('months with a usable within-month position: %d   mean w %.3f   range %.2f to %.2f'
      % (len(W), W.w.mean(), W.w.min(), W.w.max()))

# ---------- the nineteen currencies under both conventions ----------
eur = bd.h10_daily('eu')
AVG, ME = {}, {}
for cur, code in {**bd.FLOAT9, **bd.EURO10}.items():
    s = bd.h10_daily(code)
    s = 1.0 / s if cur in bd.USD_PER_FX else s
    if cur in bd.EURO_CONV:
        s = pd.concat([s[:'1998-12-31'], (bd.EURO_CONV[cur] / eur)['1999-01-01':'2001-12-31']])
    g = s.groupby(s.index.to_period('M'))
    a, m = g.mean(), g.last()
    a.index, m.index = a.index.astype(str), m.index.astype(str)
    AVG[cur] = 100 * np.log(a).diff()
    ME[cur] = 100 * np.log(m).diff()
CURS = list(AVG)
print('currencies: %d' % len(CURS))

# ---------- stack the panel ----------
LO, HI = '1974-01', '2001-12'
rows = []
for cur in CURS:
    a, m = AVG[cur], ME[cur]
    idx = [t for t in W.index if LO <= t <= HI and t in a.index]
    for t in idx:
        i = a.index.get_loc(t)
        if i < 2 or i + 1 >= len(a):
            continue
        prev = [W.loc[u, 's'] if u in W.index else np.nan for u in (a.index[i - 1], a.index[i - 2])]
        rows.append(dict(cur=cur, ym=t, s=W.loc[t, 's'], w=W.loc[t, 'w'],
                         d_me=m.iloc[i], d_avg=a.iloc[i], d_avg_next=a.iloc[i + 1],
                         dy1=a.iloc[i - 1], dy2=a.iloc[i - 2], ds1=prev[0], ds2=prev[1]))
P = pd.DataFrame(rows).dropna()
P['sw'] = P.s * P.w
P['s1w'] = P.s * (1 - P.w)
print('panel: %d currency-months, %d distinct months, %s to %s'
      % (len(P), P.ym.nunique(), P.ym.min(), P.ym.max()))


def panel_fit(y, key):
    X = pd.get_dummies(P.cur, prefix='c', drop_first=True).astype(float)
    X[key] = P[key].values
    for c in ('dy1', 'dy2', 'ds1', 'ds2'):
        X[c] = P[c].values
    X = sm.add_constant(X, has_constant='add')
    r = sm.OLS(P[y].values, X).fit(cov_type='cluster', cov_kwds={'groups': P.ym.values})
    return r.params[key], r.bse[key], r.params[key] / r.bse[key]


print('\n=== pooled panel, currency fixed effects, clustered on calendar month ===')
b_me, se_me, t_me = panel_fit('d_me', 's')
print('  PRECONDITION  B_me   = %+.3f (se %.3f, t %+.2f)   %s'
      % (b_me, se_me, t_me, 'identified' if abs(t_me) > 1.96 else 'NOT identified'))
b_now, se_now, t_now = panel_fit('d_avg', 'sw')
b_lat, se_lat, t_lat = panel_fit('d_avg_next', 's1w')
print('                B_now  = %+.3f (se %.3f, t %+.2f)' % (b_now, se_now, t_now))
print('                B_late = %+.3f (se %.3f, t %+.2f)' % (b_lat, se_lat, t_lat))
print('  ratios to B_me:  now %+.2f   late %+.2f' % (b_now / b_me, b_lat / b_me))
print('  ratio tests (|ratio-1| vs its se, delta method on the ratio):')
for nm, b_, se_ in (('now', b_now, se_now), ('late', b_lat, se_lat)):
    r_ = b_ / b_me
    se_r = abs(r_) * np.sqrt((se_ / b_) ** 2 + (se_me / b_me) ** 2)
    print('     %-4s ratio %+.2f  se %.2f  t vs 1 = %+.2f  %s'
          % (nm, r_, se_r, (r_ - 1) / se_r, 'cannot reject 1' if abs((r_ - 1) / se_r) < 1.96 else 'rejects 1'))

print('\n=== diagnostic: split at the median w (not a criterion) ===')
med = P.w.median()
for lab, sel in (('late-month meetings (w <= median)', P.w <= med), ('early-month meetings (w > median)', P.w > med)):
    Q = P[sel]
    X = pd.get_dummies(Q.cur, prefix='c', drop_first=True).astype(float)
    X['s1w'] = Q.s1w.values
    for c in ('dy1', 'dy2', 'ds1', 'ds2'):
        X[c] = Q[c].values
    X = sm.add_constant(X, has_constant='add')
    r = sm.OLS(Q.d_avg_next.values, X).fit(cov_type='cluster', cov_kwds={'groups': Q.ym.values})
    print('  %-36s B_late %+.3f (t %+.2f, %d obs)' % (lab, r.params['s1w'], r.params['s1w'] / r.bse['s1w'], len(Q)))

rec = [dict(stat='B_me', b=b_me, se=se_me, t=t_me),
       dict(stat='B_now', b=b_now, se=se_now, t=t_now),
       dict(stat='B_late', b=b_lat, se=se_lat, t=t_lat)]
for nm, b_, se_ in (('now', b_now, se_now), ('late', b_lat, se_lat)):
    r_ = b_ / b_me
    se_r = abs(r_) * np.sqrt((se_ / b_) ** 2 + (se_me / b_me) ** 2)
    rec.append(dict(stat='ratio_' + nm, b=r_, se=se_r, t=(r_ - 1) / se_r))
med = P.w.median()
for tag, sel in (('split_late_month', P.w <= med), ('split_early_month', P.w > med)):
    Q = P[sel]
    X = pd.get_dummies(Q.cur, prefix='c', drop_first=True).astype(float)
    X['s1w'] = Q.s1w.values
    for c in ('dy1', 'dy2', 'ds1', 'ds2'):
        X[c] = Q[c].values
    X = sm.add_constant(X, has_constant='add')
    r = sm.OLS(Q.d_avg_next.values, X).fit(cov_type='cluster', cov_kwds={'groups': Q.ym.values})
    rec.append(dict(stat=tag, b=r.params['s1w'], se=r.bse['s1w'],
                    t=r.params['s1w'] / r.bse['s1w'], n=len(Q)))
rec.append(dict(stat='panel', b=len(P), se=P.ym.nunique(), t=len(CURS), n=len(P)))
out = os.path.join(PKG, 'results', 'timing_test2.csv')
pd.DataFrame(rec).to_csv(out, index=False)
print('\nwrote', os.path.relpath(out, PKG))
