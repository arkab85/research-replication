"""Descriptive check added after the impact-preserving results (PROTOCOL 11): is the long-horizon pattern for inflows and
outflows in the currency markets a relation between the SIZE of the flow and the SIZE of later returns (a shared volatility
regime), not a relation between the flow and the return?

For each group of markets and horizon h, on the evaluation block: the mean across markets of the correlation of |u_t| with
|r_{t+h}| and of u_t with r_{t+h}, with a p-value from the same common calendar shifts as the main tests (the return block
is rotated against the flow innovation). One-sided for the absolute values, two-sided for the signed ones.
    python flows/volatility_check.py"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
W, cfg = load('set1'); hs = cfg['hs']; gap = cfg['gap']; rows = []; n, tr = split_n(len(W) - 2, gap); lo = max(hs) + 4
E = {m: tuple(a[tr + gap:] for a in prepare(W['f_' + m].values, W['r_' + m].values, gap, 'FR')[:2]) for m in cfg['markets']}
def c(a, b, h): return np.corrcoef(a[:-h], b[h:])[0, 1]
for g, ms in [('all twelve', cfg['markets'])] + list(cfg['groups'].items()):
    for h in hs:
        for kind, fa, fb, two in (('|flow| with |later return|', np.abs, np.abs, False), ('flow with later return', lambda x: x, lambda x: x, True)):
            obs = np.mean([c(fa(E[m][0]), fb(E[m][1]), h) for m in ms]); ref = np.array([np.mean([c(fa(E[m][0]), fb(np.roll(E[m][1], k)), h) for m in ms]) for k in range(lo, n - lo)])
            p = (1 + np.sum(np.abs(ref - ref.mean()) >= abs(obs - ref.mean()))) / (len(ref) + 1) if two else (1 + np.sum(ref >= obs)) / (len(ref) + 1)
            rows.append(dict(group=g, h=h, relation=kind, mean_correlation=obs, mean_under_rotation=ref.mean(), p_rotation=p))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'volatility_check.csv'), index=False, float_format='%.4g'); pd.set_option('display.width', 200)
print(D.pivot_table(index=['relation', 'group'], columns='h', values='mean_correlation').round(3).to_string()); print(); print(D.pivot_table(index=['relation', 'group'], columns='h', values='p_rotation').round(3).to_string())
