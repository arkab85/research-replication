"""Figure 3: the propositions at work (from theory_demos_results.json)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent.parent
d = json.load(open(P / 'theory_demos_results.json'))
fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# A
rows = d['A_predictability']['rows']
b = np.array([r['bound'] for r in rows]); h = np.array([r['reverse_hsic']['mean'] for r in rows])
lo = np.array([r['reverse_hsic']['q05'] for r in rows]); hi = np.array([r['reverse_hsic']['q95'] for r in rows])
a = ax[0, 0]
a.loglog(b, b, 'k--', lw=1, label=r'Bound $v_X/\ell_u^2=1-\varphi^2$')
a.errorbar(b, h, yerr=[h - lo, hi - h], fmt='o', color='0.2', ms=4, capsize=2, label=r'Estimated $H_b$')
for r in rows:
    a.annotate(fr"$\varphi$={r['phi']}", (r['bound'], r['reverse_hsic']['mean']), textcoords='offset points', xytext=(4, -10), fontsize=7)
a.set_xlabel(r'Driver innovation share $1-\varphi^2$'); a.set_ylabel('Reverse dependence (kernel units)')
a.set_title('A. Predictability bound', fontsize=10); a.legend(fontsize=7, frameon=False, loc='lower right')

# B
rows = d['B_state_omission']['rows']
n = np.array([r['n'] for r in rows])
m1 = np.array([r['dii_state_conditioned']['mean'] for r in rows])
l1 = np.array([r['dii_state_conditioned']['q05'] for r in rows]); u1 = np.array([r['dii_state_conditioned']['q95'] for r in rows])
m0 = np.array([r['pooled_component']['mean'] for r in rows])
popv = [r['dii'] for r in d['C_state_error_population'] if r['sigma'] == 1.0 and r['q'] == 0.0][0]
a = ax[0, 1]
a.fill_between(n, l1, u1, color='0.85', label='State-conditioned DII, 5-95%')
a.plot(n, m1, 'o-', color='0.1', ms=4, label='State-conditioned DII, mean')
a.plot(n, m0, 's--', color='0.5', ms=4, label='Pooled: each component (DII = 0)')
a.axhline(popv, ls=':', color='k', lw=1, label='Population DII with state')
a.set_xscale('log'); a.set_xlabel('Sample size'); a.set_ylabel('Kernel units')
a.set_title('B. State omission erases DII', fontsize=10); a.legend(fontsize=7, frameon=False)

# C
a = ax[1, 0]
for sig, mk in [(1.0, 'o-'), (2.0, 's--')]:
    rr = sorted([r for r in d['C_state_error_population'] if r['sigma'] == sig], key=lambda r: r['q'])
    a.plot([r['q'] for r in rr], [r['dii'] for r in rr], mk, color='0.2', ms=4, label=fr'$\sigma={sig:g}$')
a.axhline(0, color='k', lw=.6)
a.set_xlabel('State misclassification probability $q$'); a.set_ylabel('Population DII')
a.set_title('C. Imperfect states', fontsize=10); a.legend(fontsize=7, frameon=False)

# D
rows = d['D_incorporation']['population_by_h']
hh = [r['h'] for r in rows]
a = ax[1, 1]
a.plot(hh, [r['dii_inclusive']['mean'] for r in rows], 'o-', color='0.1', ms=4, label=r'Impact-inclusive $P_{t+h}-P_{t-1}$')
a.plot(hh, [r['dii_post_impact']['mean'] for r in rows], 's--', color='0.5', ms=4, label=r'Post-impact $P_{t+h}-P_t$')
a.axhline(0, color='k', lw=.6)
a.set_xlabel('Horizon $h$'); a.set_ylabel('DII (kernel units)')
a.set_title('D. Martingale price, persistent footprint', fontsize=10); a.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(P / 'dii_theory_demos.pdf'); fig.savefig(P / 'dii_theory_demos.png', dpi=130)
print('figure written')
