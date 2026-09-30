"""Single-series description of the flow innovations and outcomes (no statistic relates a flow to a price).
    python flows/describe.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
rows = []
for name in ('set1', 'set2', 'set2nine'):
    W, cfg = load(name); n, tr = split_n(len(W) - 2, cfg['gap'])
    for m in cfg['markets']:
        f = W['f_' + m].values; r = W['r_' + m].values; T = len(f) - 2; Z = np.column_stack([np.ones(T), f[1:-1], f[:-2]]); b = np.linalg.lstsq(Z[:tr], f[2:][:tr], rcond=None)[0]; u = f[2:] - Z @ b
        ev = slice(tr + cfg['gap'], T); k = lambda x: float(((x - x.mean()) ** 4).mean() / x.var() ** 2)
        rows.append(dict(set=name, market=m, T=T, first=W.index[2], last=W.index[-1], eval_from=W.index[2 + tr + cfg['gap']], n_eval=n, flow_sd=f.std(), flow_ar1=np.corrcoef(f[1:], f[:-1])[0, 1], zero_share=float((f == 0).mean()),
                         innovation_kurtosis=k(u), sd_ratio_eval_to_train=u[ev].std() / u[:tr].std(), eval_share_of_ss=float((u[ev] ** 2).sum() / (u ** 2).sum()), eval_share_of_obs=n / T,
                         outcome_sd=r.std(), outcome_kurtosis=k(r), outcome_sd_ratio=r[2:][ev].std() / r[2:][:tr].std()))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'describe.csv'), index=False, float_format='%.5g'); pd.set_option('display.width', 250); print(D.round(3).to_string(index=False))
