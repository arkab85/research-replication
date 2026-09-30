"""The nineteen-currency pooled p-values at a bootstrap size that pins them down.

The paper's default is 499 wild draws, whose Monte Carlo standard error at p near 0.05 is about one
percentage point. That is why the six-month value moved between 0.038 and 0.058 across runs: ordinary
bootstrap noise, not a failure to reproduce. This recomputes the pooled p-values with many more draws
and across several seeds, so the reported number is stable to the printed digit.

    python code/currency_seed_stability.py [B_wild] [n_seeds]
"""
import sys, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *

BW = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
BP = 1499
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

D = load_panel()
rows = []
for name, pairs, kw in (('19 currencies, 1974-2001', [('rr', 'fx_' + c) for c in FLOAT9 + EURO10], dict(end='2001-12')),
                        ('9 floaters, 1974-2001', [('rr', 'fx_' + c) for c in FLOAT9], dict(end='2001-12'))):
    for h in HS:
        us = [unit(sample(D, x, y, **kw), h) for x, y in pairs]
        pw, pp = [], []
        for s in range(SEEDS):
            P, a, b = pooled(us, BW, BP, seed=s)
            pw.append(a); pp.append(b)
        rows.append(dict(panel=name, h=h, P_N=P, n_units=len(us),
                         wild_mean=np.mean(pw), wild_min=np.min(pw), wild_max=np.max(pw),
                         paired_mean=np.mean(pp), paired_min=np.min(pp), paired_max=np.max(pp),
                         B_wild=BW, B_paired=BP, seeds=SEEDS))
        print('  %-26s h=%d  wild %.4f [%.4f, %.4f]   paired %.4f [%.4f, %.4f]' % (
            name, h, np.mean(pw), np.min(pw), np.max(pw), np.mean(pp), np.min(pp), np.max(pp)), flush=True)

T = pd.DataFrame(rows)
save(T.round(5), 'currency_seed_stability.csv')
print('\nwritten to results/currency_seed_stability.csv')
print('\nMonte Carlo standard error of a p-value near 0.05:')
for B in (499, BW):
    print('   B = %5d  ->  %.4f' % (B, (0.05 * 0.95 / B) ** 0.5))
