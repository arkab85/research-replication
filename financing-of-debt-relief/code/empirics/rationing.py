"""Cell-level statistics for the fixed-ranking (rationing) prediction and the random-rationing benchmark.

Fixed ranking: for a decision cell with index r and funded share m, choose the cell intercept a so that
mean(sigmoid(a + b r)) = m, holding the latent weight b at its pre-period estimate. Random rationing:
the issuer selects, along the same ranking, the options it would fund at its own pre-period funded
share m0, and then funds a random fraction min(1, m/m0) of them; the cell's covariance of exercise with
r is that fraction times the fixed-ranking covariance at m0. If m >= m0 the benchmark equals the
fixed-ranking prediction.

The level response is the within-cell covariance of exercise with r over the variance of r (a
linear-probability slope with cell fixed effects); the relative response divides it by the funded
share. Both aggregate covariances and variances across the cells of a period."""
import numpy as np, pandas as pd
from felogit import sig
from config import PERIODS

def solve_alpha(r, b, m):
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sig(mid + b * r).mean() > m: hi = mid
        else: lo = mid
    return (lo + hi) / 2

def fixed_p(r, b, m):
    return np.full_like(r, m) if m in (0.0, 1.0) else sig(solve_alpha(r, b, m) + b * r)

def issuer_pre_share(g):
    pre = g[g.period == 'pre']
    return pre.groupby('issuer_id').x.mean().to_dict(), float(pre.x.mean())

def cell_stats(g, b, m0=None, m0_default=None):
    out = []
    for c, gc in g.groupby('cell', sort=False):
        r = gc.r.values; x = gc.x.values; m = x.mean(); rc = r - r.mean()
        p = fixed_p(r, b, m); covp = (rc * p).sum()
        if m0 is not None:
            s0 = m0.get(gc.issuer_id.iat[0], m0_default)
            covr = (m / s0) * (rc * fixed_p(r, b, s0)).sum() if (m < s0 and s0 > 0) else covp
        else:
            covr = np.nan
        out.append((gc.issuer_id.iat[0], gc.ym.iat[0], gc.period.iat[0], len(r), x.sum(), (rc * x).sum(), covp, covr, (rc * rc).sum()))
    return pd.DataFrame(out, columns=['issuer', 'ym', 'period', 'n', 'nx', 'cov', 'covp', 'covr', 'var'])

def agg(cs):
    m = cs.nx.sum() / cs.n.sum(); V = cs['var'].sum()
    lvl = cs['cov'].sum() / V; lvlp = cs.covp.sum() / V; lvlr = cs.covr.sum() / V
    return dict(m=m, level=lvl, level_pred=lvlp, level_random=lvlr, rel=lvl / m, rel_pred=lvlp / m, rel_random=lvlr / m)

def prepare(d, by_forbearance=False):
    d = d.copy()
    d['cell'] = d.issuer_id.astype(str) + '_' + d.ym.astype(str)
    if by_forbearance:
        d['cell'] = d.cell + '_' + d.forbear.astype(int).astype(str)
    d['period'] = 'early'
    for p, a, b in PERIODS:
        d.loc[d.ym.between(a, b), 'period'] = p
    return d
