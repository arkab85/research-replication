"""Same-period relation between the flow innovation and the outcome, by market (computed AFTER the primary analysis).
It matters for the rotation test: a first stage that has learned to use Y_t is fed noise when the outcome is rotated,
which moves the reference distribution up (the 'impact effect' conservativeness).      python flows/same_period.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
rows = []
for name in ('set1', 'set2'):
    W, cfg = load(name); n, tr = split_n(len(W) - 2, cfg['gap'])
    for m in cfg['markets']:
        u, r, _ = prepare(W['f_' + m].values, W['r_' + m].values, cfg['gap'], 'FR'); ev = slice(tr + cfg['gap'], None)
        rows.append(dict(set=name, market=m, corr_train=np.corrcoef(u[:tr], r[:tr])[0, 1], corr_eval=np.corrcoef(u[ev], r[ev])[0, 1], corr_lead1_eval=np.corrcoef(u[ev][:-1], r[ev][1:])[0, 1], corr_lag1_eval=np.corrcoef(u[ev][1:], r[ev][:-1])[0, 1]))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'same_period.csv'), index=False, float_format='%.4g'); print(D.round(3).to_string(index=False)); print(D.groupby('set')[['corr_train', 'corr_eval', 'corr_lead1_eval', 'corr_lag1_eval']].mean().round(3))
