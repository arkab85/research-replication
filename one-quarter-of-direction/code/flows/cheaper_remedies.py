"""Do cheaper remedies restore the size of the default (wild) bootstrap? Monthly eight-country flow panel, actual flows,
simulated independent outcomes (as in Step 0a). Three designs:
  baseline      end-of-sample evaluation block, flows as they are
  standardized  each flow divided by its own standard deviation over the previous 24 months
  mid-sample    same block length, placed in the middle of the sample; the residualizing regressions are fitted on both flanks
Pooled statistics: mean index over horizons and the one-month index. Reported: rejection rate of the pooled wild bootstrap
at 5 percent, the mean index with no channel, and the evaluation block's share of the cause's sum of squares.
The weekly panel uses a trailing window of 104 weeks.
    python flows/cheaper_remedies.py R part nparts [set2|set1]      |      python flows/cheaper_remedies.py collect [set2|set1]"""
import sys, os, glob; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from engine import core_q


def unit_at(X, Y, h, a, n, gap, p=2):
    """core_lib.unit with the evaluation block at rows [a, a+n) and the regressions fitted on both flanks."""
    parts = [build(X[:max(a - gap, 0)], Y[:max(a - gap, 0)], h, p)] if a - gap > 40 else []
    if len(X) - (a + n + gap) > 40: parts.append(build(X[a + n + gap:], Y[a + n + gap:], h, p))
    dtr = pd.concat(parts, ignore_index=True); dtr.attrs = parts[0].attrs; s1 = Stage1().fit(dtr); dte = build(X[a:a + n], Y[a:a + n], h, p); ef, eb = s1.resid(dte)
    return dict(s1=s1, dte=dte, dates=np.arange(a + p, a + p + len(dte)), Q=core_q(ef, eb, dte), obs=dii(ef, eb, dte), n=len(dte), Xte=X[a:a + n])


NAME = sys.argv[-1] if sys.argv[-1] in ('set1', 'set2') else 'set2'; WIN = 24 if NAME == 'set2' else 104
if sys.argv[1] == 'collect':
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, f'cheaper_remedies_{NAME}_part*.csv'))]); g = D.groupby(['design', 'statistic'], sort=False)
    out = pd.DataFrame(dict(reps=g.rep.nunique(), wild_rejection=100 * g.p_wild.apply(lambda s: np.mean(s <= 0.05)), mean_index=g.value.mean(), eval_share_of_ss=g.ss_share.mean())).reset_index()
    out.insert(0, 'set', NAME); out.to_csv(os.path.join(RES, f'cheaper_remedies_{NAME}.csv'), index=False, float_format='%.5g'); print(out.to_string(index=False)); sys.exit()
R, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); W, cfg = load(NAME); mk = cfg['markets']; hs = cfg['hs']; gap = cfg['gap']; S = {'A': stats_for(hs)['A'], 'P(1)': {1: 1.0}}; rows = []
Wst = W.copy()
for m in mk: Wst['f_' + m] = W['f_' + m] / W['f_' + m].rolling(WIN).std().shift(1)
Wst = Wst.dropna()
for rep in range(part, R, nparts):
    base = sim_outcomes(len(W), len(mk), np.random.RandomState(1000 + rep))
    for design, panel in (('baseline', W), ('standardized', Wst), ('mid-sample', W)):
        P = panel.copy(); off = len(W) - len(panel)
        for i, m in enumerate(mk): P['r_' + m] = base[off:, i]
        U = {h: [] for h in hs}; ss = []
        for m in mk:
            X, Y, _ = prepare(P['f_' + m].values, P['r_' + m].values, gap, 'FR'); n, tr = split_n(len(X), gap); a = tr + gap if design != 'mid-sample' else (len(X) - n) // 2
            for h in hs: U[h].append(unit_at(X, Y, h, a, n, gap) if design == 'mid-sample' else {**RL.unit(pd.DataFrame({'x': X, 'y': Y}), h, 2, gap), 'Xte': X[a:]})
            ss.append((X[a:a + n] ** 2).sum() / (X ** 2).sum())
        for st, w in S.items():
            val, pw, _ = RL.pooled_contrast(U, w, 199, 0, rep); rows.append(dict(rep=rep, design=design, statistic=st, value=val, p_wild=pw, ss_share=np.mean(ss)))
    print('rep', rep, flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'cheaper_remedies_{NAME}_part{part}.csv'), index=False)
