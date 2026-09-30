"""Figure: why the null configuration matters (frozen results only; no re-estimation)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

P = Path(__file__).resolve().parent.parent
d = json.load(open(P / 'full_study_results.json'))
rows = [r for r in d['summary'] if r['design'] == 'regular' and r['version'] == 'feasible']
rows.sort(key=lambda r: (r['n'], r['allocation'] != 'equal'))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
labels, x = [], 0
for r in rows:
    for k, (key, col, name) in enumerate([('wild', '0.55', 'Degenerate-null (wild) only'),
                                          ('intersection', '0.1', 'Joint intersection')]):
        rate = 100 * r['rates'][key]['rate']
        lo, hi = (100 * v for v in r['rates'][key]['wilson95'])
        ax[0].bar(x + 0.38 * k, rate, 0.36, color=col, label=name if x == 0 else None)
        ax[0].errorbar(x + 0.38 * k, rate, yerr=[[rate - lo], [hi - rate]], color='k', lw=0.8, capsize=2)
    labels.append(f"n={r['n']}\n{'equal' if r['allocation']=='equal' else 'large'} training")
    x += 1
ax[0].axhline(5, ls='--', color='k', lw=0.8)
ax[0].set_xticks([i + 0.19 for i in range(len(labels))]); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel('Rejection rate (%) at nominal 5%')
ax[0].set_title('A. Simulation: true equal-positive null', fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc='upper left')

f = pd.read_csv(P / 'finance/dii_all_results.csv')
f = f[f.specification == 'primary'].reset_index(drop=True)
names = {'FX': 'Yen/USD', 'GS10': 'Treasury', 'CREDIT': 'Credit', 'VIX': 'VIX'}
cell = [f"{names.get(a, a)} h={h}" for a, h in zip(f.asset, f.h)]
y = range(len(f))
ax[1].scatter(f.p_wild, y, marker='o', facecolors='none', edgecolors='0.4', label='Degenerate-null (wild) only')
ax[1].scatter(f.p_intersection, y, marker='s', color='0.1', label='Joint intersection')
for i in y:
    ax[1].plot([f.p_wild[i], f.p_intersection[i]], [i, i], color='0.75', lw=0.8, zorder=0)
ax[1].axvline(0.05, ls='--', color='k', lw=0.8)
ax[1].set_yticks(list(y)); ax[1].set_yticklabels(cell, fontsize=7); ax[1].invert_yaxis()
ax[1].set_xlabel('p-value'); ax[1].set_xlim(0, 0.85)
ax[1].set_title('B. Monetary-shock illustration: 12 primary cells', fontsize=10)
ax[1].legend(fontsize=8, frameon=False, loc='lower right')
fig.tight_layout()
fig.savefig(P / 'dii_calibration.pdf'); fig.savefig(P / 'dii_calibration.png', dpi=150)
lo = min(r['rates']['wild']['rate'] for r in rows); hi = max(r['rates']['wild']['rate'] for r in rows)
print(f'wild-only range {100*lo:.1f}-{100*hi:.1f}; wild<0.05 cells: {(f.p_wild<0.05).sum()}; joint<0.05: {(f.p_intersection<0.05).sum()}')
