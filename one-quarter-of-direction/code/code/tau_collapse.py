"""Does one diagnostic predict both failures? tau = share of evaluation pairs with |X_i - X_j| <= |g_i - g_j|, the share
for which the first-stage error exceeds the gap between two values of the regressand. Theorem 1 gives a threshold at
p_2 = 1/2 for an ATOM; the claim under test is that the operative quantity is tau, that p_2 <= tau always, and that a
CONTINUOUS cause with a quiet evaluation block fails at the same threshold with no atom at all.

Designs, all with Y independent of X (AR(1), no channel), T = 540, h = 3, two lags, end-of-sample split:
  atom      X = B V, P(B = 0) = pi0, constant scale        (pi0 sweeps tau up through ties)
  drift     X = s_t V, s_t falling from 1 to a             (a sweeps tau up with no atom at all)
  tails     X = V from Student-t(nu), constant scale       (nu sweeps tau up through leverage)
Reported per design: mean tau, mean p_2, mean n*HSIC of the backward residual, the median-heuristic bandwidth in units of
the first-stage error, and the rejection rate of the dependent wild bootstrap and of the rotation test at 5 percent.
    python code/tau_collapse.py R part nparts        |        python code/tau_collapse.py collect
"""
import sys, os, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'flows'))
import flows_lib as FL                                   # installs the O(n^2) centering, checked against the engine
from core_lib import *
import sparsity_check as SC
T = 540; H = 3; P = 2

DESIGNS = ([('atom', p) for p in (0.0, 0.3, 0.5, 0.6, 0.65, 0.71, 0.8, 0.9)]
           + [('drift', a) for a in (1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015)]
           + [('tails', v) for v in (50.0, 8.0, 4.0, 3.0, 2.5, 2.1)])


def draw(kind, par, rng):
    v = rng.normal(size=T) if kind != 'tails' else rng.standard_t(par, size=T) / np.sqrt(par / (par - 2))
    if kind == 'atom': x = v * (rng.uniform(size=T) > par)
    elif kind == 'drift': x = v * np.linspace(1.0, par, T)
    else: x = v
    y = np.zeros(T); e = rng.normal(size=T)
    for t in range(1, T): y[t] = 0.3 * y[t - 1] + e[t]
    return x, y


def one(kind, par, rep):
    rng = np.random.RandomState(20260922 + 977 * rep); x, y = draw(kind, par, rng); d = pd.DataFrame({'x': x, 'y': y})
    u = unit(d, H); dte = u['dte']; s1 = u['s1']
    Xv = dte['X_t'].values; ghat = Xv - s1.resid(dte)[1]                         # fitted backward mean = X - residual
    dx = np.abs(Xv[:, None] - Xv[None, :]); dg = np.abs(ghat[:, None] - ghat[None, :]); iu = np.triu_indices(len(Xv), 1)
    tau = float(np.mean(dx[iu] <= dg[iu])); p2 = float(np.mean(dx[iu] == 0))
    eb = s1.resid(dte)[1]; d2 = (eb[:, None] - eb[None, :]) ** 2; med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    err = np.std(ghat - np.mean(ghat)) if np.std(ghat) > 0 else 1.0                # scale of the fitted function's variation
    hs = len(dte) * hsic2(eb, dte[dte.attrs['b']].values)
    _, pw, _ = wild_p(u['Q'], u['n'], 199, np.random.RandomState(rep))
    s1r, Xr, Yr = SC.split(d, H); obs = SC.stat(s1r, Xr, Yr, H); lo = H + P + 2; ref = np.array([SC.stat(s1r, Xr, np.roll(Yr, k), H) for k in range(lo, len(Xr) - lo)])
    pr = (1 + np.sum(ref >= obs)) / (len(ref) + 1)
    return dict(design=kind, par=par, rep=rep, tau=tau, p2=p2, nhsic=hs, bw_over_err=np.sqrt(med) / err, p_wild=pw, p_rot=pr, eval_sd=float(np.std(Xv)), n=len(dte))


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'tau_collapse_part*.csv'))]); g = D.groupby(['design', 'par'], sort=False)
        out = pd.DataFrame(dict(reps=g.rep.nunique(), tau=g.tau.mean(), p2=g.p2.mean(), n_hsic=g.nhsic.mean(), bw_over_err=g.bw_over_err.mean(),
                                wild=100 * g.p_wild.apply(lambda s: np.mean(s <= .05)), rotation=100 * g.p_rot.apply(lambda s: np.mean(s <= .05)))).reset_index()
        save(out.round(4), 'tau_collapse.csv'); return_ = None
    else:
        R, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); rows = []
        for i, (kind, par) in enumerate(DESIGNS):
            for rep in range(part, R, nparts): rows.append(one(kind, par, rep))
            print(kind, par, 'done', flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'tau_collapse_part{part}.csv'), index=False)
