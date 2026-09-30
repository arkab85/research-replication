r"""Corrected remedy comparison. Supersedes remedy_comparison.py, which was confounded.

WHAT WAS WRONG. The first version built the simulated channel inside the loop over remedies, from
each remedy's own transformed cause, and let the evaluation block differ across remedies (195 weeks
for the raw flows, 182 for the standardized ones, because the 104-week rolling window drops rows).
So it varied the alternative and the sample alongside the procedure, and the reported rise in power
from rescaling did not isolate rescaling. An external reader found this.

WHAT THIS DOES INSTEAD. Per replication and per market, the outcome is built ONCE: a simulated
baseline plus, under the alternative, a channel driven by the RAW flow innovation. Every remedy is
then applied to that identical outcome, on identical rows, with the same evaluation block. The
remedies differ only in how the cause is treated and how the test is calibrated, which is the
comparison the paper needs.

    python flows/remedy_comparison2.py R part nparts [set1|set2]
    python flows/remedy_comparison2.py collect [set1|set2]
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from engine import core_q

NAME = sys.argv[-1] if sys.argv[-1] in ('set1', 'set2') else 'set1'
WIN = 24 if NAME == 'set2' else 104
SHARES = (0.0, 0.10)
G = lambda u: u + 0.6 * (u ** 2 - 1)


def unit_at(X, Y, h, a, n, gap, p=2):
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
    fs = glob.glob(os.path.join(RES, f'remedy2_{NAME}_part*.csv'))
    D = pd.concat([pd.read_csv(f) for f in fs])
    g = D.groupby(['design', 'share', 'statistic'], sort=False)
    out = pd.DataFrame(dict(
        reps=g.rep.nunique(),
        wild=100 * g.p_wild.apply(lambda s: np.mean(s <= .05)),
        rotation=100 * g.p_rot.apply(lambda s: np.mean(s <= .05)),
        n_eval=g.n_eval.median(),
        ss_share=g.ss_share.mean())).reset_index()
    out.insert(0, 'set', NAME)
    out.to_csv(os.path.join(RES, f'remedy2_{NAME}.csv'), index=False, float_format='%.5g')
    print(out.to_string(index=False))
    n = int(out.reps.min())
    se = 100 * (.05 * .95 / n) ** .5
    print('\n%d replications; Monte Carlo se at a 5%% rate %.1f points' % (n, se))
    print('evaluation block identical across remedies:',
          out.groupby('design').n_eval.median().nunique() <= 2)
    sys.exit()

R, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
W0, cfg = load(NAME)
mk, hs, gap = cfg['markets'], cfg['hs'], cfg['gap']
S = {'A': stats_for(hs)['A'], 'P(1)': {1: 1.0}}

# --- one common row set, so every remedy sees the same sample -------------------
Wst0 = W0.copy()
for m in mk:
    Wst0['f_' + m] = W0['f_' + m] / W0['f_' + m].rolling(WIN).std().shift(1)
keep = Wst0.dropna().index
W = W0.loc[keep].reset_index(drop=True)          # raw flows, common rows
Wst = Wst0.loc[keep].reset_index(drop=True)      # standardized flows, same rows

rows = []
for rep in range(part, R, nparts):
    base = sim_outcomes(len(W), len(mk), np.random.RandomState(1000 + rep))
    for share in SHARES:
        # ---- build each market's outcome ONCE, from the RAW cause ----
        outcome = {}
        for i, m in enumerate(mk):
            r = base[:, i].copy()
            if share > 0:
                Xr, Yr, _ = prepare(W['f_' + m].values, r, gap, 'FR')
                n, tr = split_n(len(Xr), gap)
                a = tr + gap
                ch = np.zeros_like(Yr)
                for L in range(1, 5):
                    ch[L:] += G(Xr[:-L]) / 4.0
                v, c = np.var(Yr[a:a + n]), np.std(ch[a:a + n])
                if c > 0:
                    add = ch * np.sqrt(share * v) / c
                    r = r.copy()
                    r[len(r) - len(add):] += add     # prepare() drops leading rows; align at the end
            outcome[m] = r
        # ---- every remedy on that identical outcome ----
        for design, panel in (('baseline', W), ('standardized', Wst), ('mid-sample', W)):
            U = {h: [] for h in hs}
            ss, nn = [], []
            for m in mk:
                X, Y, Sp = prepare(panel['f_' + m].values, outcome[m], gap, 'FR')
                n, tr = split_n(len(X), gap)
                a = tr + gap if design != 'mid-sample' else (len(X) - n) // 2
                for h in hs:
                    u = (unit_at(X, Y, h, a, n, gap) if design == 'mid-sample'
                         else RL.unit(pd.DataFrame({'x': X, 'y': Y}), h, 2, gap))
                    u['Xte'], u['Yte'], u['Ste'] = X[a:a + n], Y[a:a + n], Sp[a:a + n]
                    u['market'], u['h'], u['p'] = m, h, 2
                    U[h].append(u)
                ss.append((X[a:a + n] ** 2).sum() / (X ** 2).sum())
                nn.append(n)
            obs, M, K = rotate(U, hs, 2, 1)
            for st, w in S.items():
                o, mu, pr = rot_p(obs, M, K, w)
                _, pw, _ = RL.pooled_contrast(U, w, 199, 0, rep)
                rows.append(dict(rep=rep, design=design, share=share, statistic=st,
                                 value=o, p_wild=pw, p_rot=pr, rotations=K,
                                 n_eval=int(np.median(nn)), ss_share=np.mean(ss)))
    print('rep %d done' % rep, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'remedy2_{NAME}_part{part}.csv'), index=False)
