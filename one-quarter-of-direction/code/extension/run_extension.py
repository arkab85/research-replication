"""Runs Parts A, B, C of extension/PROTOCOL.md.   python extension/run_extension.py [A] [B] [C] [--quick]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
import engine
def _cgram_fast(Z):
    """Same matrix as engine.cgram (H K H written as K - row means - column means + grand mean); O(n^2) instead of O(n^3)."""
    Z = np.asarray(Z, float); Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    sq = (Z ** 2).sum(1); d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0); med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    K = np.exp(-d2 / (2 * (med / 2.0) + 1e-12)); r = K.mean(1, keepdims=True); return K - r - r.T + K.mean()
import numpy as np
_z = np.random.RandomState(0).normal(size=(60, 3)); assert np.allclose(engine.cgram(_z), _cgram_fast(_z), atol=1e-12); engine.cgram = _cgram_fast
import core_lib as L; from core_lib import pd, unit, cell_tests, pooled, pooled_contrast, HUMP, FLOAT9, SHOCK_NAME
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True); QUICK = '--quick' in sys.argv; SHIFT_ONLY = '--shift-only' in sys.argv; parts = [a for a in sys.argv[1:] if a in 'ABC'] or ['A', 'B', 'C']
def B(n): return max(19, n // 10) if QUICK else n
def save(df, name): df.to_csv(os.path.join(OUT, name), index=False, float_format='%.6g'); print(f'\n== {name}'); print(df.to_string(index=False), flush=True)
def smp(D, x, y, start, end): return pd.concat([D[x].rename('x'), D[y].rename('y')], axis=1).loc[start:end].dropna()
def cells_and_pools(units_by, hs, Bc, Bp, tag, sets):
    """units_by[(x, y, h)] -> unit. Writes one cell table and one pooled table."""
    if SHIFT_ONLY: return
    rows = [dict(x=x, outcome=y, h=h, T=u['T'], nT=u['nT'], n_rows=u['n'], test_from=u['dates'][0], test_to=u['dates'][-1], DII=u['obs'], n_rho_f=u['diag'], **dict(zip(('p_wild', 'p_paired'), cell_tests(u, *Bc)))) for (x, y, h), u in units_by.items()]
    save(pd.DataFrame(rows), f'{tag}_cells.csv'); rows = []
    for name, keys in sets.items():
        for h in hs:
            us = [units_by[(x, y, h)] for x, y in keys]; P, pw, pp = pooled(us, *Bp); rows.append(dict(set=name, h=h, positive=sum(u['obs'] > 0 for u in us), of=len(us), P_N=P, p_wild=pw, p_paired=pp))
        if len(hs) == 3:
            c = {hs[0]: -0.5, hs[1]: 1.0, hs[2]: -0.5}; Hs, pw, pp = pooled_contrast({h: [units_by[(x, y, h)] for x, y in keys] for h in hs}, c, *Bp); rows.append(dict(set=name, h='hump', positive=np.nan, of=len(keys), P_N=Hs, p_wild=pw, p_paired=pp))
    if rows: save(pd.DataFrame(rows), f'{tag}_pooled.csv')


def shift_report(ds, hs, sets, tag, Bs, p=2, gap=6):
    """Circular-shift test (code/sparsity_check.py), valid under sparse shocks: cells and pooled sets from one set of shared rotations.
    ds: {(x, y): sample}. Added after size_under_sparsity.py showed the wild and paired tests over-reject when the shock is sparse."""
    import sparsity_check as SC; keys = list(ds); rows = []
    for h in hs:
        obs, M = SC.shift_matrix([ds[k] for k in keys], h, Bs, 0, p, gap)
        for j, (x, y) in enumerate(keys): o, pv, m = SC.shift_p_from(obs, M, [j]); rows.append(dict(kind='cell', name=f'{x} x {y}', h=h, of=1, stat=o, mean_under_rotation=m, p_shift=pv))
        for name, ks in sets.items(): o, pv, m = SC.shift_p_from(obs, M, [keys.index(k) for k in ks]); rows.append(dict(kind='pooled', name=name, h=h, of=len(ks), stat=o, mean_under_rotation=m, p_shift=pv))
        print(f'  shift h={h} done', flush=True)
    save(pd.DataFrame(rows).sort_values(['kind', 'name', 'h']), f'{tag}_shift.csv')


def part_A():
    D = pd.read_csv(os.path.join(HERE, 'monthly_panel_ext.csv'), index_col=0); D.index = D.index.astype(str); base = L.load_panel(); D['ebp'] = base['ebp'].reindex(D.index)
    ymap = {'spread': 'spread_ext', 'dollar': 'dollar_ext'}; new = [('jk', 'spread'), ('jk', 'dollar'), ('bs', 'spread'), ('bs', 'dollar')]; pairs22 = L.PAIRS18 + new
    U = {(x, y, h): unit(smp(D, x, ymap.get(y, y), L.START[x], '2019-12'), h) for x, y in pairs22 for h in L.HS}
    ext = [q for q in pairs22 if q[1] in ymap]
    cells_and_pools({k: v for k, v in U.items() if (k[0], k[1]) in ext}, L.HS, (B(499), B(199)), None, 'A_spread_dollar_to_2019', {})
    rows = []
    for name, keys in () if SHIFT_ONLY else (('22 rule-based pairs (spread and dollar to 2019-08)', pairs22), ('18 original pairs, spread and dollar to 2019-08', L.PAIRS18), ('7 spread and dollar pairs', ext)):
        for h in L.HS:
            us = [U[(x, y, h)] for x, y in keys]; P, pw, pp = pooled(us, B(4999), B(499)); rows.append(dict(set=name, h=h, positive=sum(u['obs'] > 0 for u in us), of=len(us), P_N=P, p_wild=pw, p_paired=pp))
        Hs, pw, pp = pooled_contrast({h: [U[(x, y, h)] for x, y in keys] for h in L.HS}, HUMP, B(4999), B(499)); rows.append(dict(set=name, h='hump', positive=np.nan, of=len(keys), P_N=Hs, p_wild=pw, p_paired=pp))
    if rows: save(pd.DataFrame(rows), 'A_pooled.csv')
    shift_report({(x, y): smp(D, x, ymap.get(y, y), L.START[x], '2019-12') for x, y in pairs22}, L.HS, {'22 rule-based pairs': pairs22, '18 original pairs (spread, dollar to 2019-08)': L.PAIRS18, '7 spread and dollar pairs': ext}, 'A', B(499))


def part_B():
    D = pd.read_csv(os.path.join(HERE, 'monthly_panel_ext.csv'), index_col=0); D.index = D.index.astype(str)
    U = {(x, c, h): unit(smp(D, x + '_ext', 'fx_' + c, {'jk': '1990-01', 'bs': '1988-02'}[x], '2024-01'), h) for x in ('jk', 'bs') for c in FLOAT9 for h in L.HS}
    cells_and_pools(U, L.HS, (B(299), B(99)), (B(499), B(149)), 'B_floaters_to_2024', {'High-freq. MP to 2024-01, 9 floaters': [('jk', c) for c in FLOAT9], 'Bauer-Swanson (2023 update) to 2023-12, 9 floaters': [('bs', c) for c in FLOAT9], 'both shocks, 18 pairs': [(x, c) for x in ('jk', 'bs') for c in FLOAT9]})
    shift_report({(x, c): smp(D, x + '_ext', 'fx_' + c, {'jk': '1990-01', 'bs': '1988-02'}[x], '2024-01') for x in ('jk', 'bs') for c in FLOAT9}, L.HS, {'JK x 9 floaters': [('jk', c) for c in FLOAT9], 'BS x 9 floaters': [('bs', c) for c in FLOAT9], 'both shocks, 18 pairs': [(x, c) for x in ('jk', 'bs') for c in FLOAT9]}, 'B', B(499))


def part_C():
    D = pd.read_csv(os.path.join(HERE, 'daily_panel.csv'), index_col=0); D.index = D.index.astype(str); hs = (21, 63, 126); dom = ['sp500', 'dy10', 'vix']; fx = ['fx_' + c for c in FLOAT9]; U = {}
    for x, st in (('jk', '1990-01-02'), ('bs', '1988-02-04')):
        for y in dom + fx:
            d = smp(D, x, y, st, '2019-12-31')
            for h in hs: U[(x, y, h)] = unit(d, h, p=2, gap=126); print(f'  built {x} x {y} h={h}: T={len(d)} n={U[(x, y, h)]["n"]} DII={U[(x, y, h)]["obs"]:+.5f}', flush=True)
    SETS = {'6 domestic pairs': [(x, y) for x in ('jk', 'bs') for y in dom], '18 currency pairs': [(x, y) for x in ('jk', 'bs') for y in fx],
                                                         'JK x 9 floaters': [('jk', y) for y in fx], 'BS x 9 floaters': [('bs', y) for y in fx], 'all 24 pairs': [(x, y) for x in ('jk', 'bs') for y in dom + fx]}
    if not os.path.exists(os.path.join(OUT, 'C_daily_shift.csv')) or '--redo-shift' in sys.argv: shift_report({(x, y): smp(D, x, y, st, '2019-12-31') for x, st in (('jk', '1990-01-02'), ('bs', '1988-02-04')) for y in dom + fx}, hs, SETS, 'C_daily', B(199), p=2, gap=126)
    cells_and_pools(U, hs, (B(299), B(99)), (B(499), B(149) if '--c-full' in sys.argv else 0), 'C_daily', SETS)   # pooled paired only with --c-full (about 90 minutes); see PROTOCOL.md, departure 4


if __name__ == '__main__':
    for p in parts: t = time.time(); globals()['part_' + p](); print(f'[part {p} done in {(time.time() - t) / 60:.1f} min]', flush=True)
