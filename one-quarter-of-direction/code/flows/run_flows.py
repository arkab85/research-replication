"""Runs flows/PROTOCOL.md.

    python flows/run_flows.py check                          # fast routines equal the engine's, on simulated data
    python flows/run_flows.py size  set1 R part nparts       # Step 0a: actual flows, simulated outcomes with no channel
    python flows/run_flows.py power set1 R part nparts       # Step 0b: the same with an asymmetric channel of known strength
    python flows/run_flows.py collect set1                   # rejection rates from the part files
    python flows/run_flows.py apply set1                     # the real outcomes; run once, after Step 0
"""
import sys, os, time, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')      # before numpy loads
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
import sparsity_check as SC
DIRECTIONS = ('FR', 'RF', 'FR+', 'FR-')


def with_outcomes(panel, markets, R_):
    P = panel.copy()
    for i, m in enumerate(markets): P['r_' + m] = R_[:, i]
    return P


def tests(U, hs, S, boot=True, B_w=199, B_p=99, seed=0, p=2, step=1):
    """Every statistic in S under the three calibrations. The paired draws are made once and shared by the statistics."""
    obs, M, K = rotate(U, hs, p, step); rows = []; D = paired_draws(U, hs, B_p, seed) if boot else None
    for name, w in S.items():
        o, mu, pr = rot_p(obs, M, K, w); row = dict(statistic=name, value=o, mean_under_rotation=mu, p_rotation=pr, min_p=1 / (K + 1), rotations=K)
        if boot: _, row['p_wild'], _ = RL.pooled_contrast(U, w, B_w, 0, seed); row['p_paired'] = paired_p(U, D, w)
        rows.append(row)
    return rows, (obs, M, K)


def asym_rows(keep, S):
    """Two-sided contrast between inflows and outflows; the two halves are rotated by the same calendar shifts."""
    (op, Mp, K), (om, Mm, K2) = keep['FR+'], keep['FR-']; assert K == K2; out = []
    for st, w in S.items():
        o = sum(w[h] * (op[h].mean() - om[h].mean()) for h in w); b = sum(w[h] * (Mp[h].mean(1) - Mm[h].mean(1)) for h in w)
        out.append(dict(direction='FR+ minus FR-', statistic=st, value=o, mean_under_rotation=b.mean(), p_rotation=(1 + np.sum(np.abs(b - b.mean()) >= abs(o - b.mean()))) / (K + 1), min_p=1 / (K + 1), rotations=K))
    return out


def mode_check():
    rng = np.random.RandomState(1); W, cfg = load('set1'); R_ = sim_outcomes(len(W), len(cfg['markets']), rng); P = with_outcomes(W, cfg['markets'], R_)
    for p, tol in ((2, 1e-12), (4, 1e-5)):
        U = make_units(P, cfg['markets'][:2], (1, 13), cfg['gap'], 'FR', p=p); worst = 0.0
        for h in (1, 13):
            for u in U[h]:
                for off in (0, 17, 60): worst = max(worst, abs(stat(u['s1'], u['Xte'], np.roll(u['Yte'], off), h, p) - SC.stat(u['s1'], u['Xte'], np.roll(u['Yte'], off), h, p)))
                worst = max(worst, abs(u['obs'] - stat(u['s1'], u['Xte'], u['Yte'], h, p)))
        # With more regressors a diagonal squared distance can round to 1e-15 instead of 0 depending on memory layout; the engine's
        # median over positive distances then moves by one order statistic. Observed and rotated values share one routine (rotate()).
        print(f'largest difference between the fast statistic and the engine, {p} lags:', worst); assert worst < tol
    Z = rng.normal(size=(190, 6)); a = rng.standard_t(4, 190); g = max(np.abs(fast_cgram(Z) - engine_cgram(Z)).max(), np.abs(fast_cgram(a) - engine_cgram(a)).max())
    h0 = float(np.sum(engine_cgram(a) * engine_cgram(Z))) / 189 ** 2; print('centering by means against H K H: largest entry difference', g, '; HSIC', h0, 'vs', hsic2(a, Z)); assert g < 1e-12
    U = make_units(P, cfg['markets'][:2], (1, 13), cfg['gap'], 'FR'); w = {1: 0.5, 13: 0.5}; _, _, pp = RL.pooled_contrast(U, w, 1, 39, 5); mine = paired_p(U, paired_draws(U, (1, 13), 39, 5), w)
    print('paired bootstrap p-value, core_lib.pooled_contrast against the fast routine (same draws):', pp, mine); assert abs(pp - mine) < 1e-12
    obs, M, K = rotate(U, (1, 13), 2); print('common calendar shifts:', K, '(n =', len(U[1][0]['Xte']), ')'); assert K == len(U[1][0]['Xte']) - 2 * (13 + 4)


def mode_size(name, R, part, nparts, power=False):
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); T = len(W); rows = []; t0 = time.time(); step = 3 if len(hs) > 3 else 1
    designs = [(0.0, 1)] if not power else [(s, k) for s in (0.01, 0.02, 0.05, 0.10) for k in (1, 4)]
    n, tr = split_n(T - 2, cfg['gap']); ev = slice(2 + tr + cfg['gap'], T); chan = {}     # ev: the rows of the full-length arrays on which the index is computed
    for m in mk:                                                          # channel g(e) = e + 0.6 (e^2 - 1) of the own-lag flow innovation, scaled where the test looks
        f = W['f_' + m].values; Z = np.column_stack([np.ones(T - 2), f[1:-1], f[:-2]]); e = f[2:] - Z @ np.linalg.lstsq(Z, f[2:], rcond=None)[0]
        e = e / e[tr + cfg['gap']:].std(); g = np.concatenate([[0, 0], e + 0.6 * (e ** 2 - 1)])
        for k in (1, 4):
            c = np.zeros(T)
            for j in range(1, k + 1): c[j:] += g[:-j]
            chan[m, k] = (c - c[ev].mean()) / c[ev].std()
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(1000 + rep); base = sim_outcomes(T, len(mk), rng)
        for share, spread in designs:
            R_ = base.copy(); se = sf = 0.0
            if share > 0:
                b = np.sqrt(share / (1 - share))
                for i, m in enumerate(mk): R_[:, i] += b * chan[m, spread]
                se = float(np.mean([np.var(b * chan[m, spread][ev]) / np.var(R_[ev, i]) for i, m in enumerate(mk)])); sf = float(np.mean([np.var(b * chan[m, spread]) / np.var(R_[:, i]) for i, m in enumerate(mk)]))
            P = with_outcomes(W, mk, R_); keep = {}
            for d in (DIRECTIONS if not power else ('FR',)):
                U = make_units(P, mk, hs, cfg['gap'], d); out, keep[d] = tests(U, hs, S, boot=not power, B_w=199, B_p=59, seed=rep, step=step)
                for r in out: rows.append(dict(rep=rep, share=share, spread=spread, share_eval=se, share_full=sf, direction=d, **r))
            if not power:
                for r in asym_rows(keep, S): rows.append(dict(rep=rep, share=share, spread=spread, share_eval=se, share_full=sf, **r))
        print(f'rep {rep} done, {time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f'{"power" if power else "size"}_{name}_part{part}.csv'), index=False)


def mode_ip(name, R, part, nparts, power=False):
    """Step 0c (PROTOCOL 10): actual flows; simulated outcomes WITH a same-period relation to the flow innovation (correlation
    0.4 in each block) and no relation at any lead or lag; plain and impact-preserving rotation side by side."""
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); T = len(W); rows = []; t0 = time.time(); step = 3 if len(hs) > 3 else 1
    designs = [(0.0, 1)] if not power else [(s, k) for s in (0.05, 0.10) for k in (1, 4)]; n, tr = split_n(T - 2, cfg['gap']); ev = slice(2 + tr + cfg['gap'], T); c = 0.4 / np.sqrt(1 - 0.16); same = {}; chan = {}
    for m in mk:
        f = W['f_' + m].values; Z = np.column_stack([np.ones(T - 2), f[1:-1], f[:-2]]); e = f[2:] - Z @ np.linalg.lstsq(Z, f[2:], rcond=None)[0]
        es = e.copy(); es[:tr + cfg['gap']] /= e[:tr].std(); es[tr + cfg['gap']:] /= e[tr + cfg['gap']:].std(); same[m] = np.concatenate([[0, 0], c * es])     # same-period term, scaled block by block
        ee = e / e[tr + cfg['gap']:].std(); g = np.concatenate([[0, 0], ee + 0.6 * (ee ** 2 - 1)])
        for k in (1, 4):
            ch = np.zeros(T)
            for j in range(1, k + 1): ch[j:] += g[:-j]
            chan[m, k] = (ch - ch[ev].mean()) / ch[ev].std()
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(5000 + rep); base = sim_outcomes(T, len(mk), rng)
        for share, spread in designs:
            R_ = base.copy(); se = 0.0
            for i, m in enumerate(mk):
                R_[:, i] += same[m]
                if share > 0: R_[:, i] += np.sqrt(share / (1 - share) * np.var(R_[ev, i])) * chan[m, spread]
            if share > 0: se = float(np.mean([share for _ in mk]))
            P = with_outcomes(W, mk, R_); keep = {}; keep_ip = {}
            for d in (DIRECTIONS if not power else ('FR',)):
                U = make_units(P, mk, hs, cfg['gap'], d); a = rotate(U, hs, 2, step); b = rotate(U, hs, 2, step, ip=True); keep[d] = a; keep_ip[d] = b
                for st, w in S.items():
                    o, mu, pr = rot_p(*a, w); _, mu2, pr2 = rot_p(*b, w); rows.append(dict(rep=rep, share=share, spread=spread, share_eval=se, direction=d, statistic=st, value=o, mean_under_rotation=mu, mean_under_rotation_ip=mu2, p_rotation=pr, p_rotation_ip=pr2, rotations=a[2]))
            if not power:
                for r1, r2 in zip(asym_rows(keep, S), asym_rows(keep_ip, S)): rows.append(dict(rep=rep, share=share, spread=spread, share_eval=se, direction=r1['direction'], statistic=r1['statistic'], value=r1['value'], mean_under_rotation=r1['mean_under_rotation'], mean_under_rotation_ip=r2['mean_under_rotation'], p_rotation=r1['p_rotation'], p_rotation_ip=r2['p_rotation'], rotations=r1['rotations']))
        print(f'rep {rep} done, {time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f'{"powerip" if power else "sizeip"}_{name}_part{part}.csv'), index=False)


def mode_apply_ip(name):
    """PROTOCOL 10, real data, once: plain and impact-preserving rotation p-values side by side (no bootstraps)."""
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); dates = pd.to_datetime(W.index).values; out = []; keep = {}; keep_ip = {}; draws = []
    specs = [('baseline', {}, mk, W, DIRECTIONS), ('lags=1', dict(p=1), mk, W, ('FR', 'RF')), ('lags=4', dict(p=4), mk, W, ('FR', 'RF')), ('raw units', dict(raw=True), mk, W, ('FR', 'RF'))]
    for spec, kw, ms, panel, dirs in specs:
        for d in dirs:
            U = make_units(panel, ms, hs, cfg['gap'], d, dates=dates, **kw); a = rotate(U, hs, kw.get('p', 2)); b = rotate(U, hs, kw.get('p', 2), ip=True)
            if spec == 'baseline':
                keep[d] = a; keep_ip[d] = b; w = S['A']      # reference draws of the mean index, for the figure
                for scheme, (o_, M_, K_) in (('plain', a), ('impact-preserving', b)): draws.append(pd.DataFrame(dict(set=name, direction=d, scheme=scheme, observed=sum(w[h] * o_[h].mean() for h in w), draw=sum(w[h] * M_[h].mean(1) for h in w))))
            groups = {'all': list(range(len(ms)))}; groups.update({g: [ms.index(m) for m in gm] for g, gm in cfg['groups'].items()} if spec == 'baseline' else {})
            for g, cols in groups.items():
                pv = {}
                for st, w in S.items():
                    o, mu, pr = rot_p(*a, w, cols); _, mu2, pr2 = rot_p(*b, w, cols); pv[st] = pr2
                    out.append(dict(set=name, spec=spec, group=g, direction=d, markets=len(cols), statistic=st, value=o, mean_under_rotation=mu, p_rotation=pr, mean_under_rotation_ip=mu2, p_rotation_ip=pr2, min_p=1 / (a[2] + 1), rotations=a[2],
                                    same_period_slope=float(np.mean([np.linalg.lstsq(np.column_stack([np.ones(len(u['Ste'])), u['Ste']]), u['Yte'], rcond=None)[0][1] for u in [U[hs[0]][i] for i in cols]]))))
                hp = holm({k: v for k, v in pv.items() if k.startswith('P(')})
                for r in out[-len(S):]: r['p_rotation_ip_holm'] = hp.get(r['statistic'], np.nan)
            print(spec, d, 'done', flush=True)
    for r1, r2 in zip(asym_rows(keep, S), asym_rows(keep_ip, S)): out.append(dict(set=name, spec='baseline', group='all', direction=r1['direction'], markets=len(mk), statistic=r1['statistic'], value=r1['value'], mean_under_rotation=r1['mean_under_rotation'], p_rotation=r1['p_rotation'], mean_under_rotation_ip=r2['mean_under_rotation'], p_rotation_ip=r2['p_rotation'], min_p=r1['min_p'], rotations=r1['rotations']))
    pd.concat(draws).to_csv(os.path.join(RES, f'rotation_draws_{name}.csv'), index=False, float_format='%.6g')
    D = pd.DataFrame(out); D.to_csv(os.path.join(RES, f'pooled_ip_{name}.csv'), index=False, float_format='%.6g'); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    print(D[(D.spec == 'baseline') & (D.group == 'all')][['direction', 'statistic', 'value', 'mean_under_rotation', 'p_rotation', 'mean_under_rotation_ip', 'p_rotation_ip', 'p_rotation_ip_holm', 'min_p']].to_string(index=False))


def mode_apply_std(name, B_w=999):
    """PROTOCOL 11, real data: the cause standardized by its own trailing volatility (24 months; 104 weeks), the cheap remedy
    of cheaper_remedies.py. Plain and impact-preserving rotation, and the pooled wild bootstrap for comparison."""
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); win = 24 if len(hs) == 3 else 104; P = W.copy(); out = []
    for m in mk: P['f_' + m] = W['f_' + m] / W['f_' + m].rolling(win).std().shift(1)
    P = P.dropna(); dates = pd.to_datetime(P.index).values
    for d in ('FR', 'RF'):
        U = make_units(P, mk, hs, cfg['gap'], d, dates=dates); a = rotate(U, hs, 2); b = rotate(U, hs, 2, ip=True)
        for st, w in S.items():
            o, mu, pr = rot_p(*a, w); _, mu2, pr2 = rot_p(*b, w); _, pw, _ = RL.pooled_contrast(U, w, B_w, 0, 7)
            out.append(dict(set=name, direction=d, statistic=st, value=o, mean_under_rotation=mu, p_rotation=pr, mean_under_rotation_ip=mu2, p_rotation_ip=pr2, p_wild=pw, min_p=1 / (a[2] + 1), rotations=a[2],
                            positive=(int((a[0][int(st[2:-1])] > 0).sum()) if st.startswith('P(') else np.nan), eval_share_of_ss=float(np.mean([(u['Xte'] ** 2).sum() for u in U[hs[0]]]) / np.mean([(prepare(P['f_' + u['market']].values, P['r_' + u['market']].values, cfg['gap'], d)[0] ** 2).sum() for u in U[hs[0]]]))))
        print(d, 'done', flush=True)
    D = pd.DataFrame(out); D.to_csv(os.path.join(RES, f'pooled_std_{name}.csv'), index=False, float_format='%.6g'); pd.set_option('display.width', 250); print(D.drop(columns=['set', 'rotations']).to_string(index=False))


def mode_collect(name):
    for kind in ('size', 'power', 'sizeip', 'powerip'):
        fs = glob.glob(os.path.join(RES, f'{kind}_{name}_part*.csv'))
        if not fs: continue
        D = pd.concat([pd.read_csv(f) for f in fs]); cols = [c for c in ('p_rotation', 'p_rotation_ip', 'p_wild', 'p_paired') if c in D and D[c].notna().any()]
        g = D.groupby(['share', 'spread', 'direction', 'statistic'], sort=False); out = g[cols].agg(lambda s: 100 * np.mean(s.dropna() <= 0.05)).round(1)
        out['reps'] = g['rep'].nunique(); out['mean_value'] = g['value'].mean(); out['mean_under_rotation'] = g['mean_under_rotation'].mean(); out['share_eval'] = g['share_eval'].mean(); out['rotations'] = g['rotations'].max()
        out = out.reset_index(); out.to_csv(os.path.join(RES, f'{kind}_{name}.csv'), index=False); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
        print(f'\n== {kind} {name}: rejection rates at 5 percent'); print(out.to_string(index=False))


def mode_apply(name, B_w=999, B_p=499):
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); dates = pd.to_datetime(W.index).values; pooled = []; cells = []; keep = {}
    n, tr = split_n(len(W) - 2, cfg['gap']); print(name, 'T =', len(W) - 2, 'evaluation block n =', n, 'from', W.index[2 + tr + cfg['gap']], 'to', W.index[-1], flush=True)
    for d in DIRECTIONS:
        U = make_units(W, mk, hs, cfg['gap'], d, dates=dates); out, (obs, M, K) = tests(U, hs, S, boot=True, B_w=B_w, B_p=B_p, seed=7); keep[d] = (obs, M, K)
        hp = holm({r['statistic']: r['p_rotation'] for r in out if r['statistic'].startswith('P(')})
        for r in out:
            isP = r['statistic'].startswith('P('); pooled.append(dict(set=name, spec='baseline', direction=d, markets=len(mk), positive=(int((obs[int(r['statistic'][2:-1])] > 0).sum()) if isP else np.nan),
                                                                       p_rotation_holm=hp.get(r['statistic'], np.nan), holm_floor=(len(hs) / (K + 1) if isP else np.nan), **r))
        for h in hs:
            for i, u in enumerate(U[h]):
                pw, pp = RL.cell_tests(u, 499, 99, seed=11); xall = prepare(W['f_' + u['market']].values, W['r_' + u['market']].values, cfg['gap'], d)[0]
                cells.append(dict(set=name, direction=d, market=u['market'], h=h, index=obs[h][i], p_rotation=(1 + np.sum(M[h][:, i] >= obs[h][i])) / (K + 1), p_wild=pw, p_paired=pp, block_share_of_cause_ss=float((u['Xte'] ** 2).sum() / (xall ** 2).sum())))
        for g, ms in cfg['groups'].items():
            c = [mk.index(m) for m in ms]
            for st, w in S.items(): o, mu, pr = rot_p(obs, M, K, w, c); pooled.append(dict(set=name, spec='group:' + g, direction=d, markets=len(ms), statistic=st, value=o, mean_under_rotation=mu, p_rotation=pr, min_p=1 / (K + 1), rotations=K))
        print(d, 'done', flush=True)
    for r in asym_rows(keep, S): pooled.append(dict(set=name, spec='baseline', markets=len(mk), **r))
    specs = [('lags=1', dict(p=1), mk, W), ('lags=4', dict(p=4), mk, W), ('raw units', dict(raw=True), mk, W)]
    fx = [m for m in mk if 'rw_' + m in W.columns]
    if fx: specs.append(('currencies, next-day noon rate', {}, fx, W.assign(**{'r_' + m: W['rw_' + m] for m in fx})))
    for spec, kw, ms, panel in specs:
        for d in ('FR', 'RF'):
            U = make_units(panel, ms, hs, cfg['gap'], d, dates=dates, **kw); obs, M, K = rotate(U, hs, kw.get('p', 2))
            for st, w in S.items(): o, mu, pr = rot_p(obs, M, K, w); pooled.append(dict(set=name, spec=spec, direction=d, markets=len(ms), statistic=st, value=o, mean_under_rotation=mu, p_rotation=pr, min_p=1 / (K + 1), rotations=K))
        print(spec, 'done', flush=True)
    pd.DataFrame(pooled).to_csv(os.path.join(RES, f'pooled_{name}.csv'), index=False, float_format='%.6g'); pd.DataFrame(cells).to_csv(os.path.join(RES, f'cells_{name}.csv'), index=False, float_format='%.6g')
    P = pd.DataFrame(pooled); pd.set_option('display.width', 250); print(P[P.spec == 'baseline'][['direction', 'statistic', 'value', 'mean_under_rotation', 'positive', 'p_rotation', 'p_rotation_holm', 'p_wild', 'p_paired', 'min_p']].to_string(index=False))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check': mode_check()
    elif mode in ('size', 'power'): mode_size(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), power=(mode == 'power'))
    elif mode in ('sizeip', 'powerip'): mode_ip(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), power=(mode == 'powerip'))
    elif mode == 'collect': mode_collect(sys.argv[2])
    elif mode == 'apply': mode_apply(sys.argv[2])
    elif mode == 'apply_ip': mode_apply_ip(sys.argv[2])
    elif mode == 'apply_std': mode_apply_std(sys.argv[2])
