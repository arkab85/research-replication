"""Four-variable monetary-policy application: the surprise vector of Jarocinski (2024, JME),
(MP1, 2-year, 10-year Treasury yield surprises, S&P 500), 30-minute windows, from the
Jarocinski-Karadi surprise database updated by the authors (data/fomc_surprises_jk.csv),
announcements with all four surprises available.  Proxy: VIX close on the last trading day
before the announcement.

Reports, for iid and moving-block bootstraps (blocks ceil(sqrt n)):
  * the 24 recursive orderings: raw and regime-conditioned (k = 3) maximum pairwise operator norms,
    joint .95 critical values over orderings and pairs, and breakdown budgets;
  * volatility floors (k = 2, 3, 5) with the boundary pretest, and a Wald test of isotropy;
  * the heteroskedasticity-identified rotation (joint diagonalisation of the three regime
    covariances), re-estimated in every draw, with column-angle intervals;
  * at that rotation: raw and conditioned maximum pairwise norms and their lower confidence bounds
    (a lower bound on the exposure of the data-generating model if the rotation is the true one),
    and the co-skewness Wald statistic (12 moments);
  * the independence-minimising rotation (point estimate) and its angles to the identified one.
Writes results/fomc4.json.  Run: OPENBLAS_NUM_THREADS=1 python benchmark/fomc4.py
"""
import os, sys, json, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.linalg import logm
from scipy.optimize import minimize
from scipy.stats import chi2
from dcore import (whiten, rotation, skew, pair_sq_norms, pair_devs, ordering_rotations, min_dependence_rotation,
                   match_columns, coskew_moments, isotropy_moments)
from acore import quantile_bins, qtile
from bcore import rho_proxy_boot, boot_index

OUT = os.path.join(HERE, 'results')
VARS = ['MP1', 'TFUT02', 'TFUT10', 'SP500']
K = 3
B = int(os.environ.get("BLIMIT", 499))
SEED = 2026092710


def load():
    d = pd.read_csv(os.path.join(HERE, 'data', 'fomc_surprises_jk.csv'), parse_dates=['start'])
    d = d.dropna(subset=VARS).reset_index(drop=True)
    vix = pd.read_csv(os.path.join(os.path.dirname(HERE), 'vix-daily.csv'), parse_dates=['DATE']).set_index('DATE')['CLOSE']
    pos = vix.index.searchsorted(d.start.dt.normalize(), side='left') - 1
    assert (pos >= 0).all() and (vix.index[pos] < d.start.dt.normalize()).all()
    return d, d[VARS].values, vix.values[pos]


def theta_of(R):
    A = np.real(logm(R)); d = R.shape[0]; iu = np.triu_indices(d, 1)
    return A[iu]


def jd(z, bins, k, t0):
    d = z.shape[1]; n = len(z); pb = np.bincount(bins, minlength=k) / n
    Ms = [np.cov(z[bins == j].T, bias=True) for j in range(k)]; iu = np.triu_indices(d, 1)

    def f(t):
        R = rotation(t, d)
        return sum(w * ((R.T @ M @ R)[iu] ** 2).sum() for w, M in zip(pb, Ms))
    r = minimize(f, t0, method='BFGS')
    R = rotation(r.x, d)
    return R, np.array([np.diag(R.T @ M @ R) for M in Ms]), r.fun


def het_rotation(z, bins, k, starts=30, seed=1):
    d = z.shape[1]; p = d * (d - 1) // 2; rng = np.random.default_rng(seed); best = None
    for s in range(starts):
        R, D, f = jd(z, bins, k, rng.uniform(-np.pi, np.pi, p) if s else np.zeros(p))
        if best is None or f < best[2]:
            best = (R, D, f)
    R, D, f = best
    o = np.argsort(-np.abs(R[:, 0]))  # canonical order: not needed, alignment handles labels
    return R, D, f


def run(u, W, block, rng, R_het, Qi):
    n, d = u.shape
    z, L = whiten(u)
    orders = ordering_rotations(L, d); keys = list(orders)
    bins = quantile_bins(W, K); one = np.zeros(n, int)
    E0 = {pi: z @ Q for pi, Q in orders.items()}
    raw0 = {pi: pair_sq_norms(E) for pi, E in E0.items()}
    cnd0 = {pi: pair_sq_norms(E, bins, K) for pi, E in E0.items()}
    Eh = z @ R_het
    rawh0, cndh0 = pair_sq_norms(Eh), pair_sq_norms(Eh, bins, K)
    cs0 = coskew_moments(Eh)
    th0 = theta_of(R_het)
    Dr = np.empty((B, len(keys), 6)); Dc = np.empty((B, len(keys), 6))
    Hr = np.empty((B, 6)); Hc = np.empty((B, 6)); CS = np.empty((B, len(cs0))); ANG = np.empty((B, d))
    t0 = time.time()
    for b in range(B):
        idx = boot_index(n, block, rng)
        ub = u[idx] - u[idx].mean(0); zb, Lb = whiten(ub); Wb = W[idx]; bb = quantile_bins(Wb, K)
        ob = ordering_rotations(Lb, d)
        for m, pi in enumerate(keys):
            Eb = zb @ ob[pi]
            Dr[b, m] = pair_devs(Eb, np.zeros(n, int), E0[pi], one, 1, raw0[pi])
            Dc[b, m] = pair_devs(Eb, bb, E0[pi], bins, K, cnd0[pi])
        Rb, _, _ = jd(zb, bb, K, th0)
        Rb, ang = match_columns(Rb, R_het); ANG[b] = ang
        Ehb = zb @ Rb
        Hr[b] = pair_devs(Ehb, np.zeros(n, int), Eh, one, 1, rawh0)
        Hc[b] = pair_devs(Ehb, bb, Eh, bins, K, cndh0)
        CS[b] = coskew_moments(Ehb)
        if b % 50 == 0:
            print(block, b, round(time.time() - t0), flush=True)
    qr = qtile(Dr.max(axis=(1, 2)), .95); qc = qtile(Dc.max(axis=(1, 2)), .95)
    brk = lambda v, q: float(np.sqrt(max(np.sqrt(v).max() - q, 0)))
    res = dict(block=block, B=B, q_raw=qr, q_cond=qc, orderings={})
    for pi in keys:
        res['orderings'][''.join(map(str, pi))] = dict(max_norm=float(np.sqrt(raw0[pi]).max()), breakdown=brk(raw0[pi], qr),
                                                     max_norm_cond=float(np.sqrt(cnd0[pi]).max()), breakdown_cond=brk(cnd0[pi], qc))
    qh, qhc = qtile(Hr.max(1), .95), qtile(Hc.max(1), .95)
    Vcs = np.cov(CS.T)
    res['het'] = dict(max_norm=float(np.sqrt(rawh0).max()), exposure_lower=brk(rawh0, qh), q=qh,
                      max_norm_cond=float(np.sqrt(cndh0).max()), residual_exposure_lower=brk(cndh0, qhc), q_cond=qhc,
                      column_angle_q95=np.quantile(ANG, .95, axis=0).tolist(),
                      coskew_wald=float(cs0 @ np.linalg.solve(Vcs, cs0)), coskew_df=len(cs0),
                      coskew_p=float(chi2.sf(cs0 @ np.linalg.solve(Vcs, cs0), len(cs0))))
    res['seconds'] = round(time.time() - t0, 1)
    return res


def isotropy_test(u, W, k, block, rng, Bi=999):
    n = len(u); z, _ = whiten(u); b = quantile_bins(W, k)
    m = isotropy_moments(z, b, k).mean(0)
    mb = []
    for _ in range(Bi):
        idx = boot_index(n, block, rng); ub = u[idx] - u[idx].mean(0); zb, _ = whiten(ub)
        mb.append(isotropy_moments(zb, quantile_bins(W[idx], k), k).mean(0))
    V = np.cov(np.array(mb).T); st = float(m @ np.linalg.solve(V, m))
    return dict(k=k, wald=st, df=len(m), p=float(chi2.sf(st, len(m))))


def main():
    d, u, W = load(); u = u - u.mean(0); n = len(u)
    z, L = whiten(u)
    out = dict(n=n, first=str(d.start.min()), last=str(d.start.max()), vars=VARS, seed=SEED,
               corr=np.corrcoef(u.T).round(3).tolist(), kurtosis_whitened=(z ** 4).mean(0).tolist())
    bins = quantile_bins(W, K)
    R_het, D, f = het_rotation(z, bins, K)
    Qi, fi = min_dependence_rotation(z, starts=8)
    Qi_m, ang_i = match_columns(Qi, R_het)
    sd = u.std(0)
    out['het_point'] = dict(k=K, regime_variances=D.tolist(), objective=f, impact=(L @ R_het).tolist(),
                            impact_sd_units=((L @ R_het) / sd[:, None]).tolist())
    out['min_dependence'] = dict(sum_sq_norms=fi, max_norm=float(np.sqrt(pair_sq_norms(z @ Qi)).max()),
                                 angles_to_het=ang_i.tolist(), impact_sd_units=((L @ Qi_m) / sd[:, None]).tolist())
    for k2 in (5,):
        R5, D5, _ = het_rotation(z, quantile_bins(W, k2), k2)
        R5m, a5 = match_columns(R5, R_het)
        out['het_point'][f'k{k2}_angles_to_k3'] = a5.tolist(); out['het_point'][f'k{k2}_regime_variances'] = D5.tolist()
    print('het', np.round(D, 2).tolist(), 'min-dep angles to het', np.round(ang_i, 1), flush=True)
    rng = np.random.default_rng(SEED)
    blocks = (1, int(np.ceil(np.sqrt(n))))
    out['floors'] = {f'k{k}_block{bl}': rho_proxy_boot(u, W, k, 999, bl, rng) for k in (2, 3, 5) for bl in blocks}
    out['isotropy'] = {f'k{k}_block{bl}': isotropy_test(u, W, k, bl, rng) for k in (2, 3, 5) for bl in blocks}
    print('floors', {k: round(v['rho'], 3) for k, v in out['floors'].items()}, 'iso', {k: round(v['p'], 3) for k, v in out['isotropy'].items()}, flush=True)
    for bl in blocks:
        out[f'block{bl}'] = r = run(u, W, bl, rng, R_het, Qi)
        s = sorted(r['orderings'].items(), key=lambda kv: kv[1]['breakdown'])
        print(bl, 'orderings not rejected raw:', [k for k, v in s if v['breakdown'] == 0], 'cond:', [k for k, v in r['orderings'].items() if v['breakdown_cond'] == 0], flush=True)
        print(bl, 'het', {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in r['het'].items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, os.environ.get('FOMC4_OUT', 'fomc4.json')), 'w'), indent=1)


if __name__ == '__main__':
    main()
