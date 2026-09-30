"""Build the balanced 100-stock panel of OPTIONS_PROTOCOL.md, Section 2. python options/options_build.py"""
from options_lib import *
D = pd.read_parquet('/home/claude/opt/option_stock_week.parquet'); D['week'] = pd.to_datetime(D.week); W = pd.DatetimeIndex(sorted(D.week.unique())); T = len(W)
ok = D.dropna(subset=['cw_last', 'skew30', 'ret']); cov = ok.groupby('permno').week.nunique(); full = cov[cov >= T - 5].index
mc = ok[ok.permno.isin(full)].groupby('permno').mcap.mean().sort_values(ascending=False); top = list(mc.index[:N])
dsi = pd.read_parquet('/mnt/user-data/uploads/Downloads/DII_QJE/crsp_dsi.parquet'); dsi['date'] = pd.to_datetime(dsi.date); dsi = dsi.set_index('date').sort_index()
g = (1 + dsi.vwretd.astype(float)); wk = pd.Series(np.searchsorted(W.values, dsi.index.values, side='left'), index=dsi.index)      # week = first Friday on or after the day
mkt = g.groupby(wk).prod() - 1; mkt = mkt[(mkt.index >= 0) & (mkt.index < T)]; mkt = pd.Series(mkt.values, index=W[mkt.index.values])
P = pd.DataFrame(index=W); info = []
for p in top:
    d = D[D.permno == p].set_index('week').reindex(W); miss = int(d.ret.isna().sum())
    P[f'cw_{p}'] = d.cw_last.astype(float).interpolate(limit_area='inside').bfill().ffill(); P[f'skew_{p}'] = d.skew30.astype(float).interpolate(limit_area='inside').bfill().ffill()
    P[f'y_{p}'] = 100 * (d.ret.astype(float) - mkt.reindex(W)).fillna(0.0); info.append(dict(permno=p, missing_weeks=miss, mean_mcap=float(mc[p])))
P.index.name = 'week'; assert np.isfinite(P.values).all(); P.to_csv(os.path.join(DATA, 'options_panel.csv')); pd.DataFrame(info).to_csv(os.path.join(DATA, 'options_panel_info.csv'), index=False)
n, tr = split_n(T - 2, GAP); print('panel', P.shape, W[0].date(), W[-1].date(), 'eval rows', n, 'from', W[T - n].date(), 'shifts', n - 2 * (max(HS) + 4), 'missing weeks max', max(i['missing_weeks'] for i in info))
