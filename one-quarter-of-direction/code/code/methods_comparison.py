"""Granger causality, transfer entropy and the kernel directional index on the same designs.
Design fixed in advance in code/METHODS_COMPARISON.md and anchored before this was run.

    python code/methods_comparison.py size      (experiment 1: no channel by construction)
    python code/methods_comparison.py direction (experiment 2: a channel known to exist, both directions)
"""
import sys, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'flows'))
import numpy as np, pandas as pd
from flows_lib import *
from engine import dii, core_q, wild_p                       # flows_lib re-exports only build/Stage1/sieve
from scipy import stats as sps

P_LAGS = 2
RESDIR = os.path.join(os.path.dirname(HERE), 'results')


# ---------------------------------------------------------------- competitors
def granger_p(x, y, h, p=P_LAGS):
    """Does x_t and its lags improve prediction of y_{t+h} given y's own lags? Asymptotic F."""
    T = len(x); lo = p
    rows_y, Zr, Zf = [], [], []
    for t in range(lo, T - h):
        yl = [y[t - k] for k in range(p)]
        xl = [x[t - k] for k in range(p)]
        rows_y.append(y[t + h]); Zr.append([1.0] + yl); Zf.append([1.0] + yl + xl)
    Y = np.asarray(rows_y); Zr = np.asarray(Zr); Zf = np.asarray(Zf)
    if len(Y) < Zf.shape[1] + 5: return np.nan

    def rss(Z):
        b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        e = Y - Z @ b
        return float(e @ e)
    r0, r1 = rss(Zr), rss(Zf)
    df1 = Zf.shape[1] - Zr.shape[1]; df2 = len(Y) - Zf.shape[1]
    if df2 <= 0 or r1 <= 0: return np.nan
    F = ((r0 - r1) / df1) / (r1 / df2)
    return float(sps.f.sf(F, df1, df2))


def _bin(v, k):
    """Equal-frequency bins, ties broken by rank."""
    r = sps.rankdata(v, method='ordinal')
    return np.minimum((r - 1) * k // len(v), k - 1).astype(int)


def _entropy_mm(counts, n):
    """Plug-in entropy with the Miller-Madow bias correction."""
    pnz = counts[counts > 0] / n
    return float(-(pnz * np.log(pnz)).sum() + (len(pnz) - 1) / (2 * n))


def transfer_entropy(x, y, h, k=4, p=1):
    """TE_{x->y} = H(y_{t+h}|y_t) - H(y_{t+h}|y_t,x_t), plug-in on equal-frequency bins."""
    T = len(x); lo = p
    yf = _bin(y, k); xf = _bin(x, k)
    a = np.array([yf[t + h] for t in range(lo, T - h)])
    b = np.array([yf[t] for t in range(lo, T - h)])
    c = np.array([xf[t] for t in range(lo, T - h)])
    n = len(a)
    def H(*cols):
        idx = np.zeros(n, dtype=np.int64)
        for col in cols: idx = idx * k + col
        return _entropy_mm(np.bincount(idx, minlength=k ** len(cols)).astype(float), n)
    return H(a, b) + H(b, c) - H(b) - H(a, b, c)


def te_p(x, y, h, B=199, seed=0, k=4):
    """Permutation p-value for transfer entropy: permute the cause, keep the outcome intact."""
    obs = transfer_entropy(x, y, h, k=k)
    rng = np.random.RandomState(seed)
    null = np.array([transfer_entropy(rng.permutation(x), y, h, k=k) for _ in range(B)])
    return float((1 + np.sum(null >= obs)) / (B + 1))


# ---------------------------------------------------------------- the kernel index
def dii_pvals(x, y, h, seed=0, B=199, step=3):
    """The index on one pair: default wild bootstrap, and the rotation calibration."""
    d = pd.DataFrame({'x': x, 'y': y})
    T = len(d); n = int(np.floor(T / np.log(T))); gap = 6; tr = T - n - gap
    s1 = Stage1().fit(build(d['x'].values[:tr], d['y'].values[:tr], h, P_LAGS))
    dte = build(d['x'].values[tr + gap:], d['y'].values[tr + gap:], h, P_LAGS)
    ef, eb = s1.resid(dte)
    obs = dii(ef, eb, dte)
    Q = core_q(ef, eb, dte)
    _, pw, _ = wild_p(Q, len(dte), B, np.random.RandomState(seed))
    Xr = d['x'].values[tr + gap:]; Yr = d['y'].values[tr + gap:]
    lo = h + P_LAGS + 2
    ref = []
    for kk in range(lo, len(Xr) - lo, step):
        dd = build(Xr, np.roll(Yr, kk), h, P_LAGS)
        e1, e2 = s1.resid(dd); ref.append(dii(e1, e2, dd))
    ref = np.asarray(ref)
    pr = float((1 + np.sum(ref >= obs)) / (len(ref) + 1)) if len(ref) else np.nan
    return obs, float(pw), pr


METHODS = ('granger', 'granger_abs', 'transfer_entropy', 'dii_wild', 'dii_rotation')


def run_pair(x, y, h, seed=0):
    out = {'granger': granger_p(x, y, h),
           'granger_abs': granger_p(np.abs(x), np.abs(y), h),
           'transfer_entropy': te_p(x, y, h, seed=seed)}
    _, pw, pr = dii_pvals(x, y, h, seed=seed)
    out['dii_wild'] = pw; out['dii_rotation'] = pr
    return out


# ---------------------------------------------------------------- experiment 1: size
def experiment_size(R=200):
    W, cfg = load('set1'); mk = cfg['markets']; gap = cfg['gap']
    rows = []
    for rep in range(R):
        rng = np.random.RandomState(7000 + rep)
        base = sim_outcomes(len(W), len(mk), rng)
        for j, m in enumerate(mk):
            u, _, _ = prepare(W['f_' + m].values, W['r_' + m].values, gap, 'FR')
            y = base[:len(u), j]
            r = run_pair(u, y, 4, seed=rep)
            rows.append(dict(rep=rep, market=m, **r))
        if (rep + 1) % 10 == 0:
            print('  rep %d/%d' % (rep + 1, R), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RESDIR, 'methods_size.csv'), index=False)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESDIR, 'methods_size.csv'), index=False)
    print('\n=== Experiment 1: rejection of a TRUE null at nominal 5%% (%d reps x %d markets) ===' % (R, len(mk)))
    for mth in METHODS:
        s = D[mth].dropna()
        print('  %-18s %5.1f%%   (n=%d)' % (mth, 100 * np.mean(s <= .05), len(s)))


# ---------------------------------------------------------------- experiment 2: direction
def experiment_direction():
    import build_flows_data as B
    W, cfg = load('set1'); mk = cfg['markets']; hs = cfg['hs']; gap = cfg['gap']
    cal = pd.DatetimeIndex(pd.to_datetime(W.index)); Pc = B.cash()
    rows = []
    for m in mk:
        s = Pc[m]
        dd = (s.diff() if m in B.UST else 100 * np.log(s).diff()).dropna()
        wk = pd.Series(np.searchsorted(cal.values, dd.index.values, side='left'), index=dd.index)
        rv = (dd ** 2).groupby(wk).sum(); rv = rv[(rv.index > 0) & (rv.index < len(cal))]
        ret = dd.groupby(wk).sum(); ret = ret.reindex(rv.index)
        v = np.log(rv.values + 1e-12); r = ret.values
        ok = np.isfinite(v) & np.isfinite(r)
        v, r = v[ok], r[ok]
        for h in hs:
            fwd = run_pair(r, v, h, seed=hash((m, h)) % 10000)          # returns -> volatility (true)
            bwd = run_pair(v, r, h, seed=hash((m, h, 'b')) % 10000)     # volatility -> returns (reverse)
            for mth in METHODS:
                rows.append(dict(market=m, h=h, method=mth, p_forward=fwd[mth], p_reverse=bwd[mth]))
        print('  %s done' % m, flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESDIR, 'methods_direction.csv'), index=False)
    D['rej_f'] = D.p_forward <= .05; D['rej_b'] = D.p_reverse <= .05
    D['outcome'] = np.where(D.rej_f & ~D.rej_b, 'resolves',
                     np.where(D.rej_f & D.rej_b, 'detects, not resolved',
                       np.where(~D.rej_f & D.rej_b, 'WRONG DIRECTION', 'misses')))
    print('\n=== Experiment 2: returns -> volatility, both directions, %d market-horizons each ===' % (len(mk) * len(hs)))
    tab = D.groupby(['method', 'outcome']).size().unstack(fill_value=0)
    print(tab.to_string())


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'direction'
    if mode == 'size': experiment_size(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else: experiment_direction()
