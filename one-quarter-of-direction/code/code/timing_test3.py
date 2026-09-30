r"""Executes code/TIMING_TEST_3_CORRECTION.md, anchored 2026-09-24T03:56:36Z before this was run.

Corrects C1 (lag construction), C2 (split significance -> joint interaction test),
C3 (ratio covariance -> stacked system, CI for the difference), C4 (information-arrival date and
regime split) and C6 (serial dependence) in the announcement-date test. Supersedes timing_test.py
and timing_test2.py.

    python timing_test3.py <path to One_Quarter_RFS_Replication>
"""
import sys, os, numpy as np, pandas as pd
import statsmodels.api as sm

if len(sys.argv) < 2:
    sys.exit(__doc__.strip().splitlines()[-1].strip())
PKG = sys.argv[1]
sys.path.insert(0, os.path.join(PKG, 'code'))
import build_data as bd

LO, HI = '1974-01', '2001-12'
ANN = pd.Period('1994-02', 'M')          # first announced FOMC decision

# ---------------- C1: complete monthly shock series, genuine zeros, calendar lags ----------------
A = pd.read_csv(os.path.join(PKG, 'data', 'raw', 'acosta_rrshocks.csv'), parse_dates=['fomc'])
A['ym'] = A.fomc.dt.to_period('M')
have = A[A.rr_update.notna()]
cover = pd.period_range(have.ym.min(), have.ym.max(), freq='M')     # where the series is defined
nz = have[have.rr_update != 0]
S = pd.Series(0.0, index=cover)                                     # no meeting  ==>  zero shock
for ym, g in have.groupby('ym'):
    S.loc[ym] = float(g.rr_update.sum())
nmeet = have.groupby('ym').size().reindex(cover).fillna(0)
adm = set(nmeet[nmeet == 1].index) & set(nz.ym)                     # exactly one, and nonzero

# ---------------- C4: w from the first business day AFTER the meeting ----------------
cal = pd.DatetimeIndex(sorted(set(bd.h10_daily('al').index)))
cdf = pd.DataFrame({'date': cal}); cdf['ym'] = cdf.date.dt.to_period('M')
W = {}
for ym, blk in cdf.groupby('ym'):
    if ym not in adm:
        continue
    days = blk.date.reset_index(drop=True)
    md = nz[nz.ym == ym].fomc.iloc[0]
    after = days[days > md]                                         # STRICTLY after: noon rates
    if len(after) == 0:
        continue                                                    # meeting on/after last bus. day
    d = int(after.index[0]) + 1
    Dm = len(days)
    W[ym] = dict(s=float(S.loc[ym]), D=Dm, d=d, w=(Dm - d + 1) / Dm,
                 post=int(ym >= ANN))
W = pd.DataFrame(W).T
print('shock series defined %s to %s' % (cover.min(), cover.max()))
print('months admitted (one meeting, nonzero): %d   with a usable post-meeting day: %d'
      % (len(adm), len(W)))

# ---------------- outcomes, both conventions ----------------
eur = bd.h10_daily('eu')
AVG, ME = {}, {}
for cur, code in {**bd.FLOAT9, **bd.EURO10}.items():
    s = bd.h10_daily(code)
    s = 1.0 / s if cur in bd.USD_PER_FX else s
    if cur in bd.EURO_CONV:
        s = pd.concat([s[:'1998-12-31'], (bd.EURO_CONV[cur] / eur)['1999-01-01':'2001-12-31']])
    g = s.groupby(s.index.to_period('M'))
    AVG[cur] = 100 * np.log(g.mean()).diff()
    ME[cur] = 100 * np.log(g.last()).diff()
CURS = list(AVG)

# ---------------- stack the panel: calendar lags from the COMPLETE series ----------------
rows = []
for cur in CURS:
    a, m = AVG[cur], ME[cur]
    for ym in W.index:
        if not (pd.Period(LO) <= ym <= pd.Period(HI)):
            continue
        if ym not in a.index or (ym + 1) not in a.index:
            continue
        l1, l2 = ym - 1, ym - 2
        if l1 not in a.index or l2 not in a.index or l1 not in S.index or l2 not in S.index:
            continue
        rows.append(dict(cur=cur, ym=ym, s=W.loc[ym, 's'], w=W.loc[ym, 'w'], post=W.loc[ym, 'post'],
                         d_me=m.loc[ym], d_avg=a.loc[ym], d_avg_next=a.loc[ym + 1],
                         dy1=a.loc[l1], dy2=a.loc[l2], ds1=S.loc[l1], ds2=S.loc[l2]))
P = pd.DataFrame(rows).dropna().reset_index(drop=True)
P['sw'] = P.s * P.w
P['s1w'] = P.s * (1 - P.w)
P['t'] = (P.ym - P.ym.min()).apply(lambda x: x.n)
print('panel: %d currency-months, %d months, %s to %s   (runs 1-2 had 93 months)'
      % (len(P), P.ym.nunique(), P.ym.min(), P.ym.max()))
print('  pre-1994 months %d   1994+ months %d'
      % (P[P.post == 0].ym.nunique(), P[P.post == 1].ym.nunique()))

FE = lambda Q: pd.get_dummies(Q.cur, prefix='c', drop_first=True).astype(float)
CTL = ['dy1', 'dy2', 'ds1', 'ds2']


def fit(Q, y, regs, dk=False):
    X = FE(Q)
    for r in regs:
        X[r] = Q[r].values
    for c in CTL:
        X[c] = Q[c].values
    X = sm.add_constant(X, has_constant='add')
    mod = sm.OLS(Q[y].values, X)
    if dk:
        # Driscoll-Kraay is statsmodels' 'hac-groupsum', indexed by TIME, not 'hac-panel',
        # which is a within-unit panel HAC and does not protect against cross-sectional
        # dependence. An earlier version of this script used 'hac-panel' and described the
        # result as Driscoll-Kraay; it was not. `time` must be a contiguous integer index in
        # calendar order, so that a lag of one is one month and omitted months are not
        # silently treated as adjacent.
        tix = (Q.ym - Q.ym.min()).apply(lambda x: x.n).values
        return mod.fit(cov_type='hac-groupsum', cov_kwds={'time': tix, 'maxlags': 6})
    return mod.fit(cov_type='cluster', cov_kwds={'groups': Q.ym.astype(str).values})


out = {}
print('\n=== corrected estimates (month-clustered | Driscoll-Kraay, 6 lags) ===')
for lab, y, r in (('B_me  ', 'd_me', 's'), ('B_now ', 'd_avg', 'sw'), ('B_late', 'd_avg_next', 's1w')):
    c, k = fit(P, y, [r]), fit(P, y, [r], dk=True)
    out[lab.strip()] = (c.params[r], c.bse[r], k.bse[r])
    print('  %s = %+.3f   se %.3f (t %+.2f)  |  DK se %.3f (t %+.2f)'
          % (lab, c.params[r], c.bse[r], c.params[r] / c.bse[r], k.bse[r], c.params[r] / k.bse[r]))

# ---------------- C3: stacked system for B_late - B_me, with the covariance ----------------
St = pd.concat([P.assign(eqn=0, y=P.d_me, reg=P.s),
                P.assign(eqn=1, y=P.d_avg_next, reg=P.s1w)], ignore_index=True)
X = pd.get_dummies(St.cur.astype(str) + '_' + St.eqn.astype(str), drop_first=True).astype(float)
X['reg_me'] = St.reg * (St.eqn == 0)
X['reg_late'] = St.reg * (St.eqn == 1)
for c in CTL:
    X[c + '_0'] = St[c] * (St.eqn == 0)
    X[c + '_1'] = St[c] * (St.eqn == 1)
X = sm.add_constant(X, has_constant='add')
sysres = sm.OLS(St.y.values, X).fit(cov_type='cluster', cov_kwds={'groups': St.ym.astype(str).values})
i, j = list(X.columns).index('reg_me'), list(X.columns).index('reg_late')
V = sysres.cov_params()
bm, bl = sysres.params['reg_me'], sysres.params['reg_late']
vd = V.iloc[i, i] + V.iloc[j, j] - 2 * V.iloc[i, j]
sd = np.sqrt(vd)
print('\n=== C3: B_late - B_me, estimated jointly ===')
print('  B_me %+.3f   B_late %+.3f   cov %+.4f (correlation %+.2f)'
      % (bm, bl, V.iloc[i, j], V.iloc[i, j] / np.sqrt(V.iloc[i, i] * V.iloc[j, j])))
print('  difference %+.3f   se %.3f   t %+.2f   95%% CI [%+.2f, %+.2f]'
      % (bl - bm, sd, (bl - bm) / sd, bl - bm - 1.96 * sd, bl - bm + 1.96 * sd))

# ---------------- C2: the timing restriction, tested jointly ----------------
med = P.w.median()
P['late'] = (P.w <= med).astype(float)
P['s1w_late'] = P.s1w * P.late
r2 = fit(P, 'd_avg_next', ['s1w', 's1w_late'])
print('\n=== C2: joint interaction test, replacing the split ===')
print('  s1w              %+.3f (se %.3f, t %+.2f)' % (r2.params['s1w'], r2.bse['s1w'], r2.params['s1w'] / r2.bse['s1w']))
print('  s1w x late-month %+.3f (se %.3f, t %+.2f)   <- the difference between groups'
      % (r2.params['s1w_late'], r2.bse['s1w_late'], r2.params['s1w_late'] / r2.bse['s1w_late']))
print('  95%% CI for the interaction [%+.2f, %+.2f]'
      % (r2.params['s1w_late'] - 1.96 * r2.bse['s1w_late'], r2.params['s1w_late'] + 1.96 * r2.bse['s1w_late']))

# ---------------- C4: regime split ----------------
print('\n=== C4: by communication regime ===')
for lab, Q in (('pre-1994 (decision not announced)', P[P.post == 0]),
               ('1994+ (announced, noon rate is pre-announcement)', P[P.post == 1])):
    if Q.ym.nunique() < 12:
        print('  %-48s too few months (%d)' % (lab, Q.ym.nunique())); continue
    rm, rl = fit(Q, 'd_me', ['s']), fit(Q, 'd_avg_next', ['s1w'])
    print('  %-48s B_me %+.3f (t %+.2f)   B_late %+.3f (t %+.2f)   [%d months]'
          % (lab, rm.params['s'], rm.params['s'] / rm.bse['s'],
             rl.params['s1w'], rl.params['s1w'] / rl.bse['s1w'], Q.ym.nunique()))

rec = [dict(stat=k, b=v[0], se=v[1], se_dk=v[2]) for k, v in out.items()]
rec += [dict(stat='diff_late_minus_me', b=bl - bm, se=sd),
        dict(stat='interaction_late', b=r2.params['s1w_late'], se=r2.bse['s1w_late']),
        dict(stat='panel', b=len(P), se=P.ym.nunique())]
pd.DataFrame(rec).to_csv(os.path.join(PKG, 'results', 'timing_test3.csv'), index=False)
print('\nwrote results/timing_test3.csv')
