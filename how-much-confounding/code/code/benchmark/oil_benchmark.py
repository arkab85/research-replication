"""Oil application: exposure floor from an internal (lagged-innovation) proxy, isotropy test
and the heteroskedasticity-identified rotation with its bootstrap interval.

Proxy: exponentially weighted moving average (weight .94) of lagged whitened squared
innovations ||z_{t-j}||^2/2 of the production--price block; it is a function of past
innovations only.  The VAR is regenerated and refitted in every bootstrap draw, with
12-month moving blocks as in the main oil analysis.

Run: OPENBLAS_NUM_THREADS=1 python benchmark/oil_benchmark.py
Writes benchmark/results/oil_benchmark.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index, rho_proxy_boot, isotropy_test, het_angle
from svarcore import var_fit, var_regenerate

OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
CODE = os.path.dirname(HERE)
SEED = 2026092703
B_BENCH, B_HET = 999, 499
LAM = .94


def ewma(u, lam=LAM):
    S = np.cov(u.T)
    m2 = np.einsum('ti,ij,tj->t', u, np.linalg.inv(S), u) / u.shape[1]
    e = np.ones(len(u))
    for t in range(1, len(u)):
        e[t] = lam * e[t - 1] + (1 - lam) * m2[t - 1]
    return e


def elasticity(L, adeg):
    A = L @ rot(np.deg2rad(adeg)); out = []
    for k in range(2):
        if A[0, k] * A[1, k] > 0 and abs(A[1, k]) > 1e-12:
            out.append(A[0, k] / A[1, k])
    return min(out, key=abs) if out else np.nan


def main():
    y = np.loadtxt(os.path.join(CODE, 'oil_extended.txt')); p = 24; cols = [0, 2]; block = 12
    Bc, U = var_fit(y, p); Uc = U - U.mean(0); u = Uc[:, cols]; n = len(u)
    W = ewma(u)
    rng = np.random.default_rng(SEED)
    res = dict(n=n, seed=SEED, proxy='EWMA(.94) of lagged ||z||^2/2', block=block)
    res['benchmark'] = {f'k{k}': rho_proxy_boot(u, W, k, B_BENCH, block, rng) for k in (2, 3, 5)}
    res['isotropy'] = {f'k{k}': isotropy_test(u, W, k, B_BENCH, block, rng) for k in (2, 3, 5)}
    print('floor', {k: (round(v['rho'], 3), round(v['lower'], 3)) for k, v in res['benchmark'].items()})
    print('isotropy', {k: round(v['p'], 3) for k, v in res['isotropy'].items()}, flush=True)
    het = {}
    for k in (2, 3, 5):
        a, L, Dv = het_angle(u, W, k); a = (a + 45) % 90 - 45
        ab = np.empty(B_HET); eb = np.empty(B_HET)
        for b in range(B_HET):
            idx = boot_index(n, block, rng)
            ys = var_regenerate(Bc, y[:p], Uc[idx]); _, ub = var_fit(ys, p)
            ub = ub - ub.mean(0); ub = ub[:, cols]
            ang, Lb, _ = het_angle(ub, ewma(ub), k); ang = (ang - a + 45) % 90 - 45 + a
            ab[b] = ang; eb[b] = elasticity(Lb, ang)
        het[f'k{k}'] = dict(angle=a, elasticity=float(elasticity(L, a)), ci95=np.percentile(ab, [2.5, 97.5]).tolist(),
                            elasticity_ci95=np.nanpercentile(eb, [2.5, 97.5]).tolist(),
                            share_nan=float(np.isnan(eb).mean()), bin_variances=Dv.tolist())
        print('het', k, round(a, 1), np.round(het[f'k{k}']['ci95'], 1), flush=True)
    res['het'] = het
    with open(os.path.join(OUT, 'oil_benchmark.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
