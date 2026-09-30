"""Evaluation-block share of each hold-out cause's sum of squares (after residualizing on two own lags with training-block
coefficients), against the block's share of the observations. python flows/holdout_describe.py"""
from holdout_lib import *
rows = []
for name in ('A', 'B'):
    cfg = HOLDOUTS[name]; A = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0)
    for cause in (['oi_growth', 'nc_short_chg', 'r'] if name == 'A' else ['oi_growth', 'nc_short_chg']):
        F = A[[f'{cause}_{m}' for m in cfg['markets']]].dropna(); T = len(F) - 2; n, tr = split_n(T, cfg['gap'])
        for m in cfg['markets']:
            f = F[f'{cause}_{m}'].values; Z = np.column_stack([np.ones(T), f[1:-1], f[:-2]]); y = f[2:]; u = y - Z @ np.linalg.lstsq(Z[:tr], y[:tr], rcond=None)[0]
            rows.append(dict(holdout=name, cause=cause, market=m, eval_rows=n, T=T, obs_share=round(100 * n / T, 1), ss_share=round(100 * np.sum(u[tr + cfg['gap']:] ** 2) / np.sum(u ** 2), 1)))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'holdout_ssshare.csv'), index=False); print(D.to_string(index=False))
