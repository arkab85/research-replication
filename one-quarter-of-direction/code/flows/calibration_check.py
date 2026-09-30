"""The companion paper's DII inference (fixed bandwidth, polynomial spaces, joint block bootstrap with common evaluation
and training multipliers) on the designs of this paper. Added after the primary analysis (PROTOCOL.md, Section 12).

    python flows/calibration_check.py size  <set> R part nparts     actual flows, simulated independent outcomes
    python flows/calibration_check.py collect <set>
    python flows/calibration_check.py apply <set>                   real data, all four cause definitions, once
    python flows/calibration_check.py shocks                        the nine and twelve price/volatility pairs on the common
                                                                   1990-2019 calendar, monthly-average and month-end prices
"""
import sys, os, glob, itertools, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'companion_dii')); sys.path.insert(0, os.path.join(ROOT, 'code'))
from dii_joint_nuisance import prepare_direction, operator_matrices
from dii_training_aware import count_weights
from dii_confidence import dyadic_block_length
from flows_lib import load, SETS, stats_for, prepare, split_n, sim_outcomes, holm, RES
import core_lib as RL
B_DRAWS = 999; SEED = 20260913          # the companion application's draw count and seed


def poly(z, degree=2):
    """Total-degree polynomial with cross terms, as in the companion paper's application."""
    z = np.atleast_2d(z); return np.column_stack([np.ones(len(z))] + [np.prod(z[:, ix], axis=1) for d in range(1, degree + 1) for ix in itertools.combinations_with_replacement(range(z.shape[1]), d)])


def trim(N):
    """Largest N' <= N divisible by its own dyadic block length."""
    while N % dyadic_block_length(N): N -= 1
    return N


def designs(X, Y, hs, gap, p=2):
    """Common origins t in [p, T - max(h)); for each horizon the forward (target Y_{t+h}) and reverse (target X_t) designs with the
    same conditioning set as engine.build. Training = the last m origins before the gap, evaluation = the last n origins."""
    T = len(X); t = np.arange(p, T - max(hs)); To = len(t); n, tr = split_n(To, gap); n = trim(n); m = trim(tr); rows = np.r_[t[tr - m:tr], t[tr + gap + (split_n(To, gap)[0] - n):]]
    assert len(rows) == m + n; te = np.arange(m, m + n); out = {}
    C = np.column_stack([Y[rows]] + [X[rows - i] for i in range(1, p + 1)] + [Y[rows - i] for i in range(1, p + 1)])
    for h in hs: out[h] = dict(f=(Y[rows + h], np.column_stack([X[rows], C])), b=(X[rows], np.column_stack([Y[rows + h], C])))
    return out, m, te


def joint_draws(units, B=B_DRAWS, seed=SEED):
    """units: {key: (target_f, regs_f, target_b, regs_b, m, te)}, all with the same m and te. Returns per key the two HSIC
    components, the DII, and the B joint draws of the quadratic and linear calibrations (companion dii_joint_nuisance,
    with the draw arrays kept so that contrasts across keys can be formed from the common multipliers)."""
    k0 = next(iter(units)); m = units[k0][4]; n = len(units[k0][5]); rng = np.random.default_rng(seed); W = count_weights(rng, n, B); WT = count_weights(rng, m, B); out = {}
    for key, (tf, zf, tb, zb, m_, te) in units.items():
        assert m_ == m and len(te) == n; comp = []
        for target, regs in ((tf, zf), (tb, zb)):
            d = prepare_direction(target, regs, m, te, poly); A, M, J = operator_matrices(d); db = WT @ d['score'] / m; H = float(A.sum() / n ** 2)
            quad = np.maximum(np.sum((W @ A) * W, axis=1) / n ** 2 + 2 * np.sum((W @ M) * db, axis=1) / n + np.sum((db @ J) * db, axis=1), 0); lin = 2 * (W @ A.sum(0) / n ** 2 + db @ M.mean(0)); comp.append((H, quad, lin))
        (Hf, qf, lf), (Hb, qb, lb) = comp; out[key] = dict(forward=Hf, reverse=Hb, dii=Hb - Hf, quad=qb - qf, lin=lb - lf)
    return out, n, m


def pooled(out, keys, w):
    """Contrast sum_h w[h] * mean_k DII(k, h) with its joint draws; upper-tail p-values."""
    D = sum(w[h] * np.mean([out[k, h]['dii'] for k in keys]) for h in w); q = sum(w[h] * np.mean([out[k, h]['quad'] for k in keys], axis=0) for h in w); l = sum(w[h] * np.mean([out[k, h]['lin'] for k in keys], axis=0) for h in w)
    pq = (1 + np.count_nonzero(q >= D - 1e-12)) / (len(q) + 1); pl = (1 + np.count_nonzero(l >= D - 1e-12)) / (len(l) + 1); return dict(value=D, p_quadratic=pq, p_linear=pl, p_intersection=max(pq, pl))


def run_panel(P, mk, hs, gap, direction, B=B_DRAWS, seed=SEED):
    units = {}
    for mkt in mk:
        X, Y, _ = prepare(P['f_' + mkt].values, P['r_' + mkt].values, gap, direction); dz, m, te = designs(X, Y, hs, gap)
        for h in hs: units[mkt, h] = (dz[h]['f'][0], dz[h]['f'][1], dz[h]['b'][0], dz[h]['b'][1], m, te)
    return joint_draws(units, B, seed)


def mode_size(name, R, part, nparts):
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); rows = []
    for rep in range(part, R, nparts):
        base = sim_outcomes(len(W), len(mk), np.random.RandomState(1000 + rep)); P = W.copy()
        for i, m_ in enumerate(mk): P['r_' + m_] = base[:, i]
        for d in ('FR', 'RF', 'FR+', 'FR-'):
            out, n, m = run_panel(P, mk, hs, cfg['gap'], d, B=399, seed=rep)
            for st, w in S.items(): rows.append(dict(rep=rep, direction=d, statistic=st, n=n, m=m, **pooled(out, mk, w)))
        print('rep', rep, flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'v_size_{name}_part{part}.csv'), index=False)


def mode_sizeip(name, R, part, nparts):
    """As mode_size, with a same-period term giving the simulated outcome a correlation of 0.4 with the actual flow innovation
    (block-by-block scaling as in run_flows.mode_ip) and no relation at any lead or lag."""
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); T = len(W); rows = []; n, tr = split_n(T - 2, cfg['gap']); c = 0.4 / np.sqrt(1 - 0.16); same = {}
    for m_ in mk:
        f = W['f_' + m_].values; Z = np.column_stack([np.ones(T - 2), f[1:-1], f[:-2]]); e = f[2:] - Z @ np.linalg.lstsq(Z, f[2:], rcond=None)[0]
        es = e.copy(); es[:tr + cfg['gap']] /= e[:tr].std(); es[tr + cfg['gap']:] /= e[tr + cfg['gap']:].std(); same[m_] = np.concatenate([[0, 0], c * es])
    for rep in range(part, R, nparts):
        base = sim_outcomes(T, len(mk), np.random.RandomState(5000 + rep)); P = W.copy()
        for i, m_ in enumerate(mk): P['r_' + m_] = base[:, i] + same[m_]
        for d in ('FR', 'RF', 'FR+', 'FR-'):
            out, n_, m = run_panel(P, mk, hs, cfg['gap'], d, B=399, seed=rep)
            for st, w in S.items(): rows.append(dict(rep=rep, direction=d, statistic=st, n=n_, m=m, **pooled(out, mk, w)))
        print('rep', rep, flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'v_sizeip_{name}_part{part}.csv'), index=False)


def mode_collect(name, kind='size'):
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, f'v_{kind}_{name}_part*.csv'))]); g = D.groupby(['direction', 'statistic'], sort=False)
    out = g[['p_quadratic', 'p_linear', 'p_intersection']].agg(lambda s: 100 * np.mean(s <= 0.05)).round(1); out['reps'] = g.rep.nunique(); out['mean_value'] = g.value.mean(); out['n'] = g.n.max(); out['m'] = g.m.max()
    out = out.reset_index(); out.to_csv(os.path.join(RES, f'v_{kind}_{name}.csv'), index=False); pd.set_option('display.width', 200); print(out.to_string(index=False))


def mode_apply(name):
    W, cfg = load(name); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); rows = []; cells = []
    for d in ('FR', 'RF', 'FR+', 'FR-'):
        out, n, m = run_panel(W, mk, hs, cfg['gap'], d); pv = {}
        for st, w in S.items(): r = pooled(out, mk, w); pv[st] = r['p_intersection']; rows.append(dict(set=name, spec='all', direction=d, markets=len(mk), statistic=st, n=n, m=m, **r))
        hp = holm({k: v for k, v in pv.items() if k.startswith('P(')})
        for r in rows[-len(S):]: r['p_intersection_holm'] = hp.get(r['statistic'], np.nan)
        for g, ms in cfg['groups'].items():
            for st, w in S.items(): rows.append(dict(set=name, spec=g, direction=d, markets=len(ms), statistic=st, n=n, m=m, **pooled(out, ms, w)))
        for (mkt, h), v in out.items(): cells.append(dict(set=name, direction=d, market=mkt, h=h, forward_hsic=v['forward'], reverse_hsic=v['reverse'], dii=v['dii'], **{k: pooled(out, [mkt], {h: 1.0})[k] for k in ('p_quadratic', 'p_linear', 'p_intersection')}))
        print(d, 'done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'v_pooled_{name}.csv'), index=False, float_format='%.6g'); pd.DataFrame(cells).to_csv(os.path.join(RES, f'v_cells_{name}.csv'), index=False, float_format='%.6g')
    R_ = pd.DataFrame(rows); pd.set_option('display.width', 220); print(R_[R_.spec == 'all'][['direction', 'statistic', 'value', 'p_quadratic', 'p_linear', 'p_intersection', 'p_intersection_holm', 'n', 'm']].to_string(index=False))


def mode_shocks():
    """Nine and twelve price/volatility pairs on the common calendar January 1990 to December 2019 (the high-frequency shock's
    sample), monthly-average and month-end outcomes, horizons 1, 3, 6, gap 6; the hump and the three-month index."""
    XR = os.path.join(ROOT, 'extension', 'data_raw'); D = RL.load_panel(); E = D.copy()
    j = json.load(open(os.path.join(XR, 'GSPC_yahoo.json')))['chart']['result'][0]; sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna()
    h15 = pd.read_csv(os.path.join(XR, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h15.columns if c.endswith('Y10_N.B')][0]; y10 = pd.to_numeric(h15[c10], errors='coerce'); y10.index = pd.to_datetime(h15.iloc[:, 0]); y10 = y10.dropna()
    v = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'VIX_History.csv')); vix = pd.Series(v['CLOSE'].values, index=pd.to_datetime(v['DATE'], format='%m/%d/%Y'))
    def me(s): mm = s.groupby(s.index.to_period('M')).last(); mm.index = mm.index.astype(str); return mm
    E['sp500'] = (100 * np.log(me(sp)).diff()).reindex(E.index); E['dy10'] = me(y10).diff().reindex(E.index); E['vix'] = (100 * np.log(me(vix)).diff()).reindex(E.index)
    hs = (1, 3, 6); S = {'P(1)': {1: 1.}, 'P(3)': {3: 1.}, 'P(6)': {6: 1.}, 'A': {1: 1 / 3, 3: 1 / 3, 6: 1 / 3}, 'hump': dict(RL.HUMP)}; rows = []; cells = []
    for tag, Pn in (('monthly averages', D), ('month-end values', E)):
        units = {}; idx = None
        for x, y in RL.PV: idx = RL.sample(Pn, x, y, start='1990-01').index if idx is None else idx.intersection(RL.sample(Pn, x, y, start='1990-01').index)
        for x, y in RL.PV:
            d = RL.sample(Pn, x, y, start='1990-01').loc[idx]; X = (d['x'].values - d['x'].values.mean()) / d['x'].values.std(); Y = (d['y'].values - d['y'].values.mean()) / d['y'].values.std(); dz, m, te = designs(X, Y, hs, 6)
            for h in hs: units[(x, y), h] = (dz[h]['f'][0], dz[h]['f'][1], dz[h]['b'][0], dz[h]['b'][1], m, te)
        out, n, m = joint_draws(units); nine = [q for q in RL.PV if q[0] != 'bs']
        for setname, keys in (('nine price and volatility pairs', nine), ('twelve price and volatility pairs', RL.PV)):
            for st, w in S.items(): rows.append(dict(outcomes=tag, set=setname, pairs=len(keys), statistic=st, n=n, m=m, **pooled(out, keys, w)))
        for (k, h), v_ in out.items(): cells.append(dict(outcomes=tag, shock=k[0], outcome=k[1], h=h, forward_hsic=v_['forward'], reverse_hsic=v_['reverse'], dii=v_['dii'], **{c: pooled(out, [k], {h: 1.0})[c] for c in ('p_quadratic', 'p_linear', 'p_intersection')}))
        print(tag, 'done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'holdout', 'results', 'v_shocks.csv'), index=False, float_format='%.6g'); pd.DataFrame(cells).to_csv(os.path.join(ROOT, 'holdout', 'results', 'v_shocks_cells.csv'), index=False, float_format='%.6g')
    pd.set_option('display.width', 220); print(pd.DataFrame(rows)[['outcomes', 'set', 'statistic', 'value', 'p_quadratic', 'p_linear', 'p_intersection', 'n', 'm']].to_string(index=False))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'size': mode_size(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif mode == 'collect': mode_collect(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'size')
    elif mode == 'sizeip': mode_sizeip(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif mode == 'apply': mode_apply(sys.argv[2])
    elif mode == 'shocks': mode_shocks()
