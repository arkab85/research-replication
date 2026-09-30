r"""Executes code/REMEDY_COMPARISON.md, anchored 2026-09-24T15:18:40Z before this was run.

Size AND power for every practical remedy, on identical designs and identical simulated outcomes.
Three treatments of the cause (baseline, trailing-volatility standardized, mid-sample block) x two
channel strengths (none, a tenth of evaluation-block return variance) x two calibrations (dependent
wild bootstrap, enumerated rotation).

    python flows/remedy_comparison.py R part nparts [set1|set2]
    python flows/remedy_comparison.py collect [set1|set2]
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from engine import core_q

NAME = sys.argv[-1] if sys.argv[-1] in ('set1', 'set2') else 'set1'
WIN = 24 if NAME == 'set2' else 104
SHARES = (0.0, 0.10)
G = lambda u: u + 0.6 * (u ** 2 - 1)          # asymmetric channel, as in the paper's power section


def unit_at(X, Y, h, a, n, gap, p=2):
    """flows_lib.unit with the evaluation block at [a, a+n) and regressions fitted on both flanks."""
    parts = [build(X[:max(a - gap, 0)], Y[:max(a - gap, 0)], h, p)] if a - gap > 40 else []
    if len(X) - (a + n + gap) > 40:
        parts.append(build(X[a + n + gap:], Y[a + n + gap:], h, p))
    dtr = pd.concat(parts, ignore_index=True)
    dtr.attrs = parts[0].attrs
    s1 = Stage1().fit(dtr)
    dte = build(X[a:a + n], Y[a:a + n], h, p)
    ef, eb = s1.resid(dte)
    return dict(s1=s1, dte=dte, dates=np.arange(a + p, a + p + len(dte)), Q=core_q(ef, eb, dte),
                obs=dii(ef, eb, dte), n=len(dte), Xte=X[a:a + n])


if sys.argv[1] == 'collect':
    fs = glob.glob(os.path.join(RES, f'remedy_comparison_{NAME}_part*.csv'))
    D = pd.concat([pd.read_csv(f) for f in fs])
    g = D.groupby(['design', 'share', 'statistic'], sort=False)
    out = pd.DataFrame(dict(
        reps=g.rep.nunique(),
        wild=100 * g.p_wild.apply(lambda s: np.mean(s <= .05)),
        rotation=100 * g.p_rot.apply(lambda s: np.mean(s <= .05)),
        mean_index=g.value.mean(),
        ss_share=g.ss_share.mean())).reset_index()
    out.insert(0, 'set', NAME)
    out.to_csv(os.path.join(RES, f'remedy_comparison_{NAME}.csv'), index=False, float_format='%.5g')
    print(out.to_string(index=False))
    n = int(out.reps.min())
    print('\nMonte Carlo se at a 5%% rate with %d reps: %.1f points' % (n, 100 * (.05 * .95 / n) ** .5))
    sys.exit()

R, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W, cfg = load(NAME)
mk, hs, gap = cfg['markets'], cfg['hs'], cfg['gap']
S = {'A': stats_for(hs)['A'], 'P(1)': {1: 1.0}}
Wst = W.copy()
for m in mk:
    Wst['f_' + m] = W['f_' + m] / W['f_' + m].rolling(WIN).std().shift(1)
Wst = Wst.dropna()

rows = []
for rep in range(part, R, nparts):
    base = sim_outcomes(len(W), len(mk), np.random.RandomState(1000 + rep))
    for share in SHARES:
        for design, panel in (('baseline', W), ('standardized', Wst), ('mid-sample', W)):
            P = panel.copy()
            off = len(W) - len(panel)
            for i, m in enumerate(mk):
                P['r_' + m] = base[off:, i]
            U = {h: [] for h in hs}
            ss = []
            for m in mk:
                X, Y, Sp = prepare(P['f_' + m].values, P['r_' + m].values, gap, 'FR')
                n, tr = split_n(len(X), gap)
                a = tr + gap if design != 'mid-sample' else (len(X) - n) // 2
                if share > 0:                       # asymmetric channel over lags 1..4
                    ch = np.zeros_like(Y)
                    for L in range(1, 5):
                        ch[L:] += G(X[:-L]) / 4.0
                    blk = slice(a, a + n)
                    v = np.var(Y[blk])
                    c = np.std(ch[blk])
                    if c > 0:
                        Y = Y + ch * np.sqrt(share * v) / c
                # the evaluation-block arrays rotate() needs, attached exactly as make_units does
                for h in hs:
                    u = (unit_at(X, Y, h, a, n, gap) if design == 'mid-sample'
                         else RL.unit(pd.DataFrame({'x': X, 'y': Y}), h, 2, gap))
                    u['Xte'], u['Yte'], u['Ste'] = X[a:a + n], Y[a:a + n], Sp[a:a + n]
                    u['market'], u['h'], u['p'] = m, h, 2
                    U[h].append(u)
                ss.append((X[a:a + n] ** 2).sum() / (X ** 2).sum())
            # the rotation reference set is built once per design and shared by both statistics,
            # exactly as run_flows.py does it
            obs, M, K = rotate(U, hs, 2, 1)
            for st, w in S.items():
                o, mu, pr = rot_p(obs, M, K, w)
                _, pw, _ = RL.pooled_contrast(U, w, 199, 0, rep)
                rows.append(dict(rep=rep, design=design, share=share, statistic=st,
                                 value=o, p_wild=pw, p_rot=pr, min_p=1.0 / (K + 1),
                                 rotations=K, ss_share=np.mean(ss)))
    print('rep %d done' % rep, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'remedy_comparison_{NAME}_part{part}.csv'), index=False)
