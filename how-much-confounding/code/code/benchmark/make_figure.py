"""Figure 1: exposure needed for each rotation to be the truth, raw and conditioned on
three proxy regimes (lower confidence bounds sqrt of certified cell bounds), with the recursive
orderings, the heteroskedasticity interval (equity) and the range of point embedding floors.
Writes 3_LaTeX_Source/figures/exposure_by_rotation.pdf.
Run: python benchmark/make_figure.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results')
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), '3_LaTeX_Source', 'figures')
BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#1f1f1e', '#8a8983'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def steps(mid, y):
    return mid, y


def panel(ax, deg_mid, raw, adj, lines, band, shade=None, title=''):
    if shade is not None:
        ax.axvspan(*shade, color='#e8e7e1', lw=0, zorder=0)
    ax.axhspan(*band, color='#cde2fb', alpha=.55, lw=0, zorder=0)
    ax.plot(deg_mid, raw, color=BLUE, lw=2, label='Raw certificate', zorder=3)
    ax.plot(deg_mid, adj, color=ORANGE, lw=2, ls='--', label='Conditioned on VIX regimes', zorder=3)
    for x, lab in lines:
        ax.axvline(x, color=MUTED, lw=1, ls=':', zorder=1)
        ax.text(x, ax.get_ylim()[1] if False else 0.0, '', color=INK)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel('Rotation (degrees)')
    ax.grid(axis='y', color='#ecebe6', lw=.8)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    axs = axs[::-1]  # equity left, monetary policy right
    # FOMC, iid, one-degree cells; display on [-45, 45)
    n = np.load(os.path.join(RES, 'fomc_raw_b1.npz'))
    raw = np.sqrt(n['cell_lower']); a = json.load(open(os.path.join(RES, 'fomc_adjusted.json')))['block1']['adjusted_grid']['k3']
    adj = np.array(a['cell_lower_sqrt']); mid = np.arange(90) + .5
    mid = np.where(mid >= 45, mid - 90, mid); o = np.argsort(mid)
    nf = json.load(open(os.path.join(RES, 'nu_floors_fomc.json')))['block1']
    pts = np.array([nf[f'k{k}']['orders_pair_floor_point'][1] for k in (2, 3, 5)])
    fomc_ang = json.load(open(os.path.join(RES, 'fomc.json')))['raw']['block1']['recursive']['angles_deg'][1]
    panel(axs[0], mid[o], raw[o], adj[o], [(0, 'Rates first'), (fomc_ang, 'Stocks first')], (pts.min(), pts.max()),
          title='Monetary-policy surprises')
    axs[0].annotate('rates first', (0, .245), ha='center', fontsize=8, color=INK)
    axs[0].annotate('stocks first', (fomc_ang, .245), ha='center', fontsize=8, color=INK)
    axs[0].set_xlim(-45, 45)
    # Equity, blocks of ten, three-degree cells
    e = json.load(open(os.path.join(RES, 'equity_benchmark.json')))['raw']
    ea = json.load(open(os.path.join(RES, 'equity_adjusted_b10.json')))['adjusted_grid']
    g1 = json.load(open(os.path.join(RES, 'equity_grid1_b10.json')))
    raw = np.array(g1['raw']['cell_lower_sqrt']); adj = np.array(g1['cond']['cell_lower_sqrt']); mid = np.arange(90) + .5
    nfe = json.load(open(os.path.join(RES, 'nu_floors_equity.json')))['block10']
    hi = int(round(e['het']['k2']['angle'] / 3))
    pts = [nfe[f'k{k}']['orders_pair_floor_point'][j] for k in (2, 3, 5) for j in range(2)]
    ang2 = e['block10']['angles_deg'][1]
    panel(axs[1], mid, raw, adj, [(0, ''), (ang2, '')], (min(pts), max(pts)), shade=(min(g1['ar']['k2']['set']) - .5, max(g1['ar']['k2']['set']) + .5),
          title='Returns and volatility')
    axs[1].annotate('returns first', (1, .245), ha='left', fontsize=8, color=INK)
    axs[1].annotate('VIX first', (ang2, .245), ha='center', fontsize=8, color=INK)
    axs[1].annotate('robust\nrotation set', (np.mean(g1['ar']['k2']['set']), .012), ha='center', fontsize=7.5, color=INK)
    axs[1].set_xlim(0, 90)
    axs[1].set_ylabel('Exposure needed (lower bound)')
    for ax in axs:
        ax.set_ylim(0, .27)
    h, l = axs[1].get_legend_handles_labels()
    from matplotlib.patches import Patch
    h.append(Patch(color='#cde2fb', alpha=.55)); l.append('Embedding floor at the rejected orderings')
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'exposure_by_rotation.pdf'))
    fig.savefig('/tmp/claude-0/-home-claude/c02c7242-61ad-5b4c-9415-a595f8c70298/scratchpad/fig.png', dpi=150)


if __name__ == '__main__':
    main()
