r"""Executes code/TIMING_TEST.md, anchored 2026-09-23T21:05:21Z before this was run.

Does the within-month date of a shock predict how much of its impact response is
displaced into the following month's average-to-average change?

    python timing_test.py <path to One_Quarter_RFS_Replication>

Writes results/timing_test.csv and prints the three criteria stated in the protocol.

One implementation choice the protocol did not fix: the twelve outcomes are in different
units (log points for equities and currencies, percentage points for the yield, index
points for the VIX), so an equal-weighted mean of raw coefficients is not meaningful.
Each outcome is therefore standardized by its own monthly standard deviation before
pooling. Per-series estimates are reported in native units, unstandardized. This choice
is disclosed rather than silent; it affects only the pooled row.
"""
import sys, os, numpy as np, pandas as pd
import statsmodels.api as sm

PKG = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__.strip().splitlines()[2].strip())
D = pd.read_csv(os.path.join(PKG, 'extension', 'daily_panel.csv'), parse_dates=['date'])
D['ym'] = D.date.dt.to_period('M')

LOGS = ['sp500', 'fx_AUD', 'fx_CAD', 'fx_DKK', 'fx_JPY', 'fx_NZD', 'fx_NOK', 'fx_SEK', 'fx_CHF', 'fx_GBP']
LEVS = ['dy10', 'vix']
OUT = LOGS + LEVS

# ---- monthly constructions from the same daily data ----
lvl = D[['ym']].copy()
for c in OUT:
    v = D[c].where(D[c] > 0) if c in LOGS else D[c]   # a zero quote is missing, not a price
    lvl[c] = 100 * np.log(v) if c in LOGS else v
avg = lvl.groupby('ym').mean()
me = lvl.groupby('ym').last()

# ---- admitted months: exactly one announcement, and its position in the month ----
def calendar(shock):
    g = D[['ym', 'date', shock]].copy()
    g['nz'] = g[shock].fillna(0) != 0
    rows = []
    for ym, blk in g.groupby('ym'):
        blk = blk.sort_values('date').reset_index(drop=True)
        hits = blk.index[blk.nz].tolist()
        if len(hits) != 1:
            continue
        Dm = len(blk)
        d = hits[0] + 1                       # 1-indexed position among the month's trading days
        rows.append(dict(ym=ym, s=blk.loc[hits[0], shock], D=Dm, d=d, w=(Dm - d + 1) / Dm))
    return pd.DataFrame(rows).set_index('ym')


def nw(y, X):
    X = sm.add_constant(X, has_constant='add')
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)   # screens inf as well as nan
    if ok.sum() < 40:
        return np.nan, np.nan, int(ok.sum())
    r = sm.OLS(y[ok], X[ok]).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    return r.params.iloc[1], r.bse.iloc[1], int(ok.sum())


rows = []
for shock in ('jk', 'bs'):
    cal = calendar(shock)
    tot = D.groupby('ym')[shock].apply(lambda v: (v.fillna(0) != 0).sum())
    print('%s: %d months with an announcement, %d admitted (exactly one), %d excluded (two or more)'
          % (shock, (tot >= 1).sum(), len(cal), (tot >= 2).sum()))
    for c in OUT:
        dme, davg = me[c].diff(), avg[c].diff()
        f = pd.DataFrame(index=avg.index)
        f['s'] = cal.s.reindex(f.index)
        f['w'] = cal.w.reindex(f.index)
        f = f[f.s.notna()]
        X = pd.DataFrame(index=f.index)
        X['sw'] = f.s * f.w
        X['s1w'] = f.s * (1 - f.w)
        X['s'] = f.s
        for L in (1, 2):
            X['dy%d' % L] = davg.shift(L).reindex(f.index)
            X['ds%d' % L] = f.s.shift(L)
        ctl = ['dy1', 'dy2', 'ds1', 'ds2']
        b_me, se_me, n = nw(dme.reindex(f.index), X[['s'] + ctl])
        b_now, se_now, _ = nw(davg.reindex(f.index), X[['sw'] + ctl])
        b_lat, se_lat, _ = nw(davg.shift(-1).reindex(f.index), X[['s1w'] + ctl])
        sd = davg.std()
        rows.append(dict(shock=shock, outcome=c, n=n, sd=sd,
                         B_me=b_me, se_me=se_me, B_now=b_now, se_now=se_now,
                         B_late=b_lat, se_late=se_lat,
                         t_late=b_lat / se_lat if se_lat else np.nan,
                         z_me=b_me / sd, z_now=b_now / sd, z_late=b_lat / sd))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(PKG, 'results', 'timing_test.csv'), index=False)

print('\n%-5s %-8s %5s %10s %10s %10s %8s' % ('shock', 'outcome', 'n', 'B_me', 'B_now', 'B_late', 't(late)'))
for _, r in R.iterrows():
    print('%-5s %-8s %5d %10.3f %10.3f %10.3f %8.2f'
          % (r.shock, r.outcome, r.n, r.B_me, r.B_now, r.B_late, r.t_late))

print('\n=== pooled, outcomes standardized by their own monthly SD ===')
for shock in ('jk', 'bs'):
    s = R[R.shock == shock]
    zm, zn, zl = s.z_me.mean(), s.z_now.mean(), s.z_late.mean()
    # equal-weighted mean across series; SE across series, which is conservative here
    sel = s.z_late.std(ddof=1) / np.sqrt(len(s))
    print('%s : B_me %+.3f   B_now %+.3f   B_late %+.3f (se across series %.3f, t %+.2f)'
          % (shock, zm, zn, zl, sel, zl / sel))
    print('      ratios to B_me:  now %+.2f   late %+.2f' % (zn / zm, zl / zm))

print('\n=== diagnostic: split at the median w (not a criterion) ===')
for shock in ('jk', 'bs'):
    cal = calendar(shock)
    med = cal.w.median()
    for half, sel_ in (('late-month announcements (w below median)', cal.w <= med),
                       ('early-month announcements (w above median)', cal.w > med)):
        acc = []
        for c in OUT:
            davg = avg[c].diff()
            f = cal[sel_].copy()
            X = pd.DataFrame(index=f.index)
            X['s1w'] = f.s * (1 - f.w)
            for L in (1, 2):
                X['dy%d' % L] = davg.shift(L).reindex(f.index)
                X['ds%d' % L] = f.s.shift(L)
            b, se, n = nw(davg.shift(-1).reindex(f.index), X)
            if np.isfinite(b):
                acc.append(b / davg.std())
        print('%s  %-44s B_late (pooled, SD units) %+.3f  [%d series, median w %.2f]'
              % (shock, half, np.mean(acc) if acc else np.nan, len(acc), med))
