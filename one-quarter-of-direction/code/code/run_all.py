"""Reproduces Tables 2-9 and Figures 2-5 of 'One Quarter of Direction'.

    python code/run_all.py                 # everything, bootstrap sizes as in the paper (about 30-60 minutes)
    python code/run_all.py --quick         # bootstrap sizes divided by ten: a fifteen-minute smoke test
    python code/run_all.py --tables 2 3    # selected tables only

Point estimates (DII, pooled statistics, n*rho_f, local projections) are deterministic given data/analysis_panel.csv.
Bootstrap p-values are Monte Carlo quantities; seeds are fixed here (seed=0 throughout) so reruns are identical."""
import sys, argparse, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_lib import *

ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ap.add_argument('--tables', nargs='*', default=[str(k) for k in range(2, 10)])
ap.add_argument('--no-figures', action='store_true'); A = ap.parse_args()
def B(n): return max(19, n // 10) if A.quick else n
D = load_panel(); _cache = {}
def U(x, y, h, **kw):
    key = (x, y, h, tuple(sorted(kw.items())))
    if key not in _cache: _cache[key] = unit(sample(D, x, y, **{k: v for k, v in kw.items() if k in ('start', 'end', 'xshift')}), h, **{k: v for k, v in kw.items() if k in ('p', 'gap')})
    return _cache[key]
def units(pairs, h, **kw): return [U(x, y, h, **kw) for x, y in pairs]
def byh(pairs, **kw): return {h: units(pairs, h, **kw) for h in HS}


def table2():
    rows = []
    for x, y in PAIRS18:
        for h in HS:
            u = U(x, y, h); pw, pp = cell_tests(u, B(499), B(199))
            rows.append(dict(shock=SHOCK_NAME[x], x=x, outcome=y, h=h, T=u['T'], nT=u['nT'], DII=u['obs'], p_wild=pw, p_paired=pp, n_rho_f=u['diag']))
    return save(pd.DataFrame(rows), 'table2_pairs.csv')


SETS3 = {'eighteen pairs': PAIRS18, 'fifteen pairs excl. EBP': [q for q in PAIRS18 if q[1] != 'ebp'],
         'twelve pairs excl. Bauer-Swanson and EBP': [q for q in PAIRS18 if q[0] != 'bs' and q[1] != 'ebp'],
         'nine price and volatility pairs': [q for q in PV if q[0] != 'bs']}
def table3():
    rows = []
    for name, pairs in SETS3.items():
        for h in HS:
            us = units(pairs, h); P, pw, pp = pooled(us, *((B(4999), B(499)) if h == 3 else (B(499), B(149))))
            rows.append(dict(set=name, h=h, positive=sum(u['obs'] > 0 for u in us), of=len(us), P_N=P, p_wild=pw, p_paired=pp))
    t = pd.DataFrame(rows); pers = t.groupby('set', sort=False)[['p_wild', 'p_paired']].max().reset_index(); pers.insert(1, 'h', 'persistence (max over h)')
    return save(pd.concat([t, pers], ignore_index=True), 'table3_pooled.csv')


def table4():
    a = pd.DataFrame([dict(shock=SHOCK_NAME[x], **{f'h{h}': np.mean([U(x, y, h)['obs'] for y in ('sp500', 'dy10', 'vix')]) for h in HS}) for x in ('rr', 'jk', 'bs', 'oil')])
    save(a, 'table4A_family_means.csv'); rows = []
    for name, drop in (('all four shocks (12 pairs)', None), ('drop narrative', 'rr'), ('drop Jarocinski-Karadi', 'jk'), ('drop Bauer-Swanson (the nine pairs)', 'bs'), ('drop oil', 'oil')):
        pairs = [q for q in PV if q[0] != drop]; bw, bp = (B(4999), B(999)) if drop is None else (B(2999), B(299)); Uh = byh(pairs)
        P3, pw3, pp3 = pooled(Uh[3], bw, bp); H, pwH, ppH = pooled_contrast(Uh, HUMP, bw, bp)
        rows.append(dict(sample=name, P_N3=P3, p_wild=pw3, p_paired=pp3, hump=H, hump_p_wild=pwH, hump_p_paired=ppH))
    return save(pd.DataFrame(rows), 'table4B_leave_one_out.csv')


def table5():
    rows = []
    for name, pairs in (('nine price and volatility pairs', SETS3['nine price and volatility pairs']), ('all eighteen rule-based pairs', PAIRS18)):
        Uh = byh(pairs)
        for stat, c in (('P_N(3)', {3: 1.0}), ('hump H', HUMP), ('P_N(3)-P_N(1)', D31), ('P_N(3)-P_N(6)', D36)):
            v, pw, pp = pooled_contrast(Uh, c, B(4999), B(499)); rows.append(dict(set=name, statistic=stat, value=v, p_wild=pw, p_paired=pp))
    for k in (12, 24, -12, -24):                    # placebo calendars: each shock moved k months, 18 pairs, h=3, wild only
        v, pw, _ = pooled(units(PAIRS18, 3, xshift=k), B(999), 0); rows.append(dict(set='placebo calendar, 18 pairs, h=3', statistic=f'shift {k:+d} months', value=v, p_wild=pw, p_paired=np.nan))
    return save(pd.DataFrame(rows), 'table5_hump_placebo.csv')


def table6():
    cells, pools = [], []; fx = lambda c: 'fx_' + c
    for x in ('rr', 'jk', 'bs'):
        for h in HS:
            for c in FLOAT9:
                u = U(x, fx(c), h); pw, pp = cell_tests(u, B(299), B(99)); cells.append(dict(shock=SHOCK_NAME[x], x=x, currency=c, h=h, nT=u['nT'], DII=u['obs'], p_wild=pw, p_paired=pp))
            us = units([(x, fx(c)) for c in FLOAT9], h); P, pw, pp = pooled(us, B(499), B(149)); pools.append(dict(panel=SHOCK_NAME[x] + ', 9 floaters', h=h, positive=sum(u['obs'] > 0 for u in us), of=9, P_N=P, p_wild=pw, p_paired=pp))
    for name, pairs, kw in (('All 27 pairs', [(x, fx(c)) for x in ('rr', 'jk', 'bs') for c in FLOAT9], {}),
                            ('19 currencies, 1974-2001', [('rr', fx(c)) for c in FLOAT9 + EURO10], dict(end='2001-12')),
                            ('9 floaters, 1974-2001', [('rr', fx(c)) for c in FLOAT9], dict(end='2001-12'))):
        for h in HS:
            us = units(pairs, h, **kw); P, pw, pp = pooled(us, B(499), B(149)); pools.append(dict(panel=name, h=h, positive=sum(u['obs'] > 0 for u in us), of=len(us), P_N=P, p_wild=pw, p_paired=pp))
    save(pd.DataFrame(cells), 'table6_currency_cells.csv'); return save(pd.DataFrame(pools), 'table6_currency_pooled.csv')


def lp_single(x, y, h, L=2):
    n = len(y); t = np.arange(L, n - h); Z = np.column_stack([np.ones(len(t)), x[t]] + [x[t - l] for l in range(1, L + 1)] + [y[t - l] for l in range(1, L + 1)])
    return np.linalg.lstsq(Z, y[t + h], rcond=None)[0][1]
def lp_panel_dk(frames, h, L=2, power=1):
    """Panel local projection with currency fixed effects and Driscoll-Kraay standard errors (Bartlett kernel)."""
    Zs, Ys, Ts = [], [], []
    for d in frames:
        x = d['x'].values; y = d['y'].values; n = len(y); t = np.arange(L, n - h)
        Z = np.column_stack([x[t]] + [x[t - l] for l in range(1, L + 1)] + [y[t - l] for l in range(1, L + 1)]); yy = y[t + h] ** power
        Zs.append(Z - Z.mean(0)); Ys.append(yy - yy.mean()); Ts.append(d.index.values[t])
    Z = np.vstack(Zs); Y = np.concatenate(Ys); dates = np.concatenate(Ts); b = np.linalg.lstsq(Z, Y, rcond=None)[0]; e = Y - Z @ b
    g = pd.DataFrame(Z * e[:, None]).groupby(dates).sum().sort_index().values; Tn = len(g); m = int(np.floor(4 * (Tn / 100) ** (2 / 9))); S = g.T @ g
    for l in range(1, m + 1): G = g[l:].T @ g[:-l]; S += (1 - l / (m + 1)) * (G + G.T)
    XXi = np.linalg.inv(Z.T @ Z); V = XXi @ S @ XXi; return b[0], b[0] / np.sqrt(V[0, 0])
def table7():
    """Two versions. 'as published': the Australian and New Zealand dollars, the pound and the Irish punt are left in the
    H.10 quotation, US$ per unit, so that for these four a dollar appreciation is NEGATIVE; this reproduces the published
    Table 7 exactly and contradicts its note. 'corrected': all nineteen as foreign currency per dollar, as the note says.
    The index is exactly invariant to the sign of the outcome, so Tables 6 and 9 are unaffected; an average across
    currencies (the baskets) and a regression coefficient (the local projections) are not.
    The all-nineteen basket averages the currencies available in each month, so it continues to 2019 on the nine floaters."""
    sets = {'all 19 basket': FLOAT9 + EURO10, 'European basket': EURO10, 'non-European basket': ['AUD', 'CAD', 'NZD', 'JPY']}; rb, rl = [], []
    for ver in ('as published', 'corrected'):
        P = D.copy()
        if ver == 'as published':
            for c in ('AUD', 'NZD', 'GBP', 'IEP'): P['fx_' + c] = -P['fx_' + c]
        for name, cs in sets.items():
            P['bk'] = P[['fx_' + c for c in cs]].mean(axis=1, skipna=True)
            for h in HS:
                u = unit(sample(P, 'rr', 'bk'), h); pw, pp = cell_tests(u, B(499), B(199)); rb.append(dict(version=ver, basket=name, h=h, T=u['T'], DII=u['obs'], p_wild=pw, p_paired=pp))
        fr = [sample(P, 'rr', 'fx_' + c) for c in FLOAT9 + EURO10]; frc = [sample(P, 'rr', 'fx_' + c, end='2001-12') for c in FLOAT9 + EURO10]
        for h in (0, 1, 3, 6):
            b = np.array([lp_single(d['x'].values, d['y'].values, h) for d in fr]); bp, tp = lp_panel_dk(fr, h); bc, tc = lp_panel_dk(frc, h); _, t2 = lp_panel_dk(fr, h, power=2); _, t2c = lp_panel_dk(frc, h, power=2); _, t3 = lp_panel_dk(fr, h, power=3)
            rl.append(dict(version=ver, h=h, mean_beta=b.mean(), dispersion=b.std(ddof=1) / np.sqrt(len(b)), positive=int((b > 0).sum()), of=len(b), panel_beta=bp, panel_t_DK=tp, panel_beta_common=bc, panel_t_common=tc, sq_t=t2, sq_t_common=t2c, cube_t=t3))
    save(pd.DataFrame(rb), 'table7_baskets.csv'); return save(pd.DataFrame(rl), 'table7_local_projections.csv')


def table8(R=40, T=551, seed=0):
    rng = np.random.RandomState(seed); rr = sample(D, 'rr', 'sp500')['x'].values[:T]; rows = []
    designs = {'dense': (lambda: rng.normal(size=T), (0, .2, .4, .6, .8, 1.0)), 'sparse (p=0.15)': (lambda: rng.normal(size=T) * (rng.rand(T) < .15), (0, .1, .2, .3, .5, 1.0, 1.5)),
               'actual narrative shock': (lambda: rr, (0, .1, .2, .3, .5, 1.0, 1.5))}
    for name, (draw, grid) in designs.items():
        for s in grid:
            v, sh = [], []
            for _ in range(R if not A.quick else 8):
                X = draw(); x2 = X ** 2 - np.mean(X ** 2); e = rng.normal(size=T); Y = np.zeros(T)
                for t in range(1, T): Y[t] = 0.3 * Y[t - 1] + (s * x2[t - 3] if t >= 3 else 0.0) + e[t]
                v.append(unit(pd.DataFrame({'x': X, 'y': Y}, index=np.arange(T).astype(str)), 3)['obs']); sh.append(np.var(s * x2) / np.var(Y))
            rows.append(dict(design=name, s=s, share=np.mean(sh), mean_DII3=np.mean(v)))
    return save(pd.DataFrame(rows), 'table8_calibration.csv')


def table9():
    fx = lambda c: 'fx_' + c; rows = []
    # The published 19-currency rows use each currency's own sample (floaters to 2019, legacy currencies to 2001), not the
    # 1974-2001 common window of Table 6; the common-window version is reported as well so that like is compared with like.
    c19 = [('rr', fx(c)) for c in FLOAT9 + EURO10]
    specs = (('Domestic price and volatility pairs, h=3', SETS3['nine price and volatility pairs'], 3, {}), ('Narrative x 19 currencies, h=1', c19, 1, {}), ('Narrative x 19 currencies, h=6', c19, 6, {}),
             ('All 27 currency pairs, h=6', [(x, fx(c)) for x in ('rr', 'jk', 'bs') for c in FLOAT9], 6, {}),
             ('Narrative x 19 currencies, h=1, common window 1974-2001', c19, 1, dict(end='2001-12')), ('Narrative x 19 currencies, h=6, common window 1974-2001', c19, 6, dict(end='2001-12')))
    for name, pairs, h, kw in specs:
        for p in (1, 2, 3):
            us = units(pairs, h, p=p, **kw); P, pw, pp = pooled(us, B(299), B(99)); rows.append(dict(result=name, lags=p, positive=sum(u['obs'] > 0 for u in us), of=len(us), P_N=P, p_wild=pw, p_paired=pp))
    return save(pd.DataFrame(rows), 'table9_lag_robustness.csv')


def figures():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rd = lambda f: pd.read_csv(os.path.join(RES, f)) if os.path.exists(os.path.join(RES, f)) else None
    t3 = rd('table3_pooled.csv')
    if t3 is not None:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for s, st in (('eighteen pairs', 'o-'), ('nine price and volatility pairs', 's--')):
            g = t3[(t3['set'] == s) & (t3['h'].astype(str).isin(['1', '3', '6']))]; ax.plot(g['h'].astype(int), g['P_N'], st, label=s, color='k')
            for _, r in g.iterrows(): ax.annotate(f"{r['p_wild']:.2f}, {r['p_paired']:.2f}", (int(r['h']), r['P_N']), textcoords='offset points', xytext=(4, 5), fontsize=7)
        ax.axhline(0, lw=.5, color='k'); ax.set_xticks([1, 3, 6]); ax.set_xlabel('horizon h (months after the shock)'); ax.set_ylabel('pooled index'); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(RES, 'figure2_horizon_profile.pdf')); plt.close(fig)
    t6 = rd('table6_currency_cells.csv')
    if t6 is not None:
        g = t6[t6['x'] == 'rr']; fig, axs = plt.subplots(3, 3, figsize=(7, 5.5), sharex=True, sharey=True)
        for ax, c in zip(axs.ravel(), FLOAT9): q = g[g['currency'] == c]; ax.bar(q['h'].astype(str), q['DII'], color='.4'); ax.axhline(0, lw=.5, color='k'); ax.set_title(c, fontsize=8)
        fig.suptitle('Narrative shock, nine floating currencies: index by horizon', fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(RES, 'figure3_floaters.pdf')); plt.close(fig)
    lp, t6p = rd('table7_local_projections.csv'), rd('table6_currency_pooled.csv')
    if lp is not None and t6p is not None:
        lp = lp[lp['version'] == 'as published']                       # the published Figure 4; see table7() for the corrected version
        fig, ax = plt.subplots(1, 2, figsize=(7, 3)); ax[0].errorbar(lp['h'], lp['mean_beta'], yerr=lp['dispersion'], fmt='o-', color='k'); ax[0].axhline(0, lw=.5, color='k'); ax[0].set_title('mean local projection (percent per unit shock)', fontsize=8)
        q = t6p[t6p['panel'] == '19 currencies, 1974-2001']; ax[1].plot(q['h'], q['P_N'], 's-', color='k'); ax[1].axhline(0, lw=.5, color='k'); ax[1].set_title('pooled index, 19 currencies', fontsize=8)
        for a in ax: a.set_xlabel('h')
        fig.tight_layout(); fig.savefig(os.path.join(RES, 'figure4_lp_vs_index.pdf')); plt.close(fig)
    t8 = rd('table8_calibration.csv')
    if t8 is not None:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for (nm, g), st in zip(t8.groupby('design', sort=False), ('o-', 's--', '^:')): ax.plot(g['share'], g['mean_DII3'], st, color='k', label=nm)
        ax.set_xlabel('share of outcome variance explained by the channel'); ax.set_ylabel('mean index at h=3'); ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(RES, 'figure5_calibration.pdf')); plt.close(fig)


if __name__ == '__main__':
    t0 = time.time()
    for k in A.tables:
        t1 = time.time(); globals()['table' + k](); print(f'[table {k} done in {time.time() - t1:.0f}s]')
    if not A.no_figures: figures()
    print(f'\nall done in {(time.time() - t0) / 60:.1f} minutes; outputs in {RES}')
