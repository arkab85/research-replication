"""Two experiments aimed at the Finance DE/AE's core concerns.

P. Balanced-panel averages and factor adjustment (Theorem 1(iii) with fixed N).
   Common driver X_t ~ N(0,1); N units, Y_{i,t+1} = theta (X_t^2-1) + f_{t+1} + e_{i,t+1},
   observed common factor f independent of X. theta=0 is a double-independence null;
   theta>0 gives positive DII in every unit. Contrast: equal-weight average of unit DIIs,
   with or without the factor in both regressor vectors. Common bootstrap paths.
R. Heavy tails: the baseline equal-positive ('regular') and double-null designs with
   Student-t innovations (df = 5, 3; unit variance) versus Gaussian, fitted quadratic
   means, joint procedure. df=3 violates the eighth-moment condition of Theorem 1.
Seed 20260921. Output: fan_pelger_results.json.
"""
from pathlib import Path
import json, sys
import numpy as np

S = Path(__file__).resolve().parent
P = S.parent
sys.path.insert(0, str(S))
from dii_joint_nuisance import prepare_direction, operator_matrices  # noqa: E402
from dii_training_aware import count_weights  # noqa: E402

SEED = 20260921
m, gap, n, B = 204, 39, 120, 399
te = np.arange(m + gap, m + gap + n)


def poly2(z):
    z = np.atleast_2d(z)
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            cols.append(z[:, i] * z[:, j])
    return np.column_stack(cols)


def prepare_extra(target, regressors, extra, m, te, basis):
    """prepare_direction with additional mean columns (e.g. an observed factor) that
    enter the fitted mean but not the regressor kernel. Their coefficients' training
    influence is propagated exactly like other regression coefficients; the residual
    is invariant to affine standardization of the extra columns."""
    target = np.asarray(target); z = np.asarray(regressors); extra = np.atleast_2d(np.asarray(extra).T).T
    sy = target[:m].std()
    ty = (target - target[:m].mean()) / sy
    zz = (z - z[:m].mean(0)) / z[:m].std(0)
    Pm = np.column_stack([basis(zz), extra]); beta = np.linalg.lstsq(Pm[:m], ty[:m], rcond=None)[0]
    u = ty - Pm @ beta; Q = Pm[:m].T @ Pm[:m] / m
    if np.linalg.matrix_rank(Q) != Q.shape[0]:
        raise ValueError('Singular design')
    score_beta = (Pm[:m] * u[:m, None]) @ np.linalg.inv(Q)
    score_scale = np.column_stack([.5 * (ty[:m] ** 2 - 1), .5 * (zz[:m] ** 2 - 1)])
    d = dict(u=u[te], z=zz[te], P=Pm[te], score=np.column_stack([score_beta, score_scale, zz[:m]]), p=Pm.shape[1])
    return d, beta[-extra.shape[1]:] * sy


def contrast_test(pairs, weights, rng):
    """p-values for sum_j w_j (H_b,j - H_f,j) with common evaluation/training paths."""
    W = count_weights(rng, n, B); WT = count_weights(rng, m, B)
    D = 0.0; q = np.zeros(B); l = np.zeros(B)
    for (fwd, rev), w in zip(pairs, weights):
        for d, sgn in [(fwd, -1.0), (rev, 1.0)]:
            A, M, J = operator_matrices(d)
            db = WT @ d['score'] / m
            H = float(A.sum() / n ** 2)
            quad = np.sum((W @ A) * W, axis=1) / n ** 2 + 2 * np.sum((W @ M) * db, axis=1) / n + np.sum((db @ J) * db, axis=1)
            lin = 2 * (W @ A.sum(0) / n ** 2 + db @ M.mean(0))
            D += sgn * w * H; q += sgn * w * np.maximum(quad, 0); l += sgn * w * lin
    pq = (1 + np.count_nonzero(q >= D - 1e-12)) / (B + 1)
    pl = (1 + np.count_nonzero(l >= D - 1e-12)) / (B + 1)
    return D, max(pq, pl)


rng = np.random.default_rng(SEED)
out = {'settings': dict(m=m, gap=gap, n=n, draws=B, seed=SEED)}

# ---------------------------------------------------------------- P
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
Prow = []
L = m + gap + n
for theta in [0.0, 0.35]:
    for N in [1, 4, 16]:
        modes = ['unadjusted', 'factor_adjusted', 'defactored_known', 'defactored_estimated']
        rej = {k: 0 for k in modes}; dsum = {k: [] for k in modes}
        for r in range(reps):
            x = rng.normal(size=L); f = rng.normal(size=L)
            Y = theta * (x ** 2 - 1)[:, None] + f[:, None] + rng.normal(size=(L, N))
            for mode in rej:
                pairs = []
                for i in range(N):
                    if mode == 'unadjusted':
                        fwd = prepare_direction(Y[:, i], x[:, None], m, te, poly2)
                        rev = prepare_direction(x, Y[:, i][:, None], m, te, poly2)
                    elif mode == 'factor_adjusted':
                        fwd = prepare_direction(Y[:, i], np.column_stack([x, f]), m, te, poly2)
                        rev = prepare_direction(x, np.column_stack([Y[:, i], f]), m, te, poly2)
                    elif mode == 'defactored_known':
                        yt = Y[:, i] - f
                        fwd = prepare_direction(yt, x[:, None], m, te, poly2)
                        rev = prepare_direction(x, yt[:, None], m, te, poly2)
                    else:
                        fwd, lam = prepare_extra(Y[:, i], x[:, None], f, m, te, poly2)
                        yt = Y[:, i] - lam[0] * f
                        rev = prepare_direction(x, yt[:, None], m, te, poly2)
                    pairs.append((fwd, rev))
                D, p = contrast_test(pairs, [1.0 / N] * N, rng)
                rej[mode] += p <= 0.05; dsum[mode].append(D)
        for mode in rej:
            Prow.append(dict(theta=theta, N=N, mode=mode, reps=reps, reject=rej[mode] / reps,
                             mean_dii=float(np.mean(dsum[mode]))))
        print('P', theta, N, {k: v / reps for k, v in rej.items()}, flush=True)
out['panel'] = Prow

# ---------------------------------------------------------------- R
def basis_state(z):
    """Quadratic in the continuous coordinate, linear in the binary state, and their product."""
    z = np.atleast_2d(z)
    return np.column_stack([np.ones(len(z)), z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 0] * z[:, 1]])


def tdraw(df, size):
    if df is None:
        return rng.normal(size=size)
    return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))


Rrow = []
for design in ['regular', 'double']:
    for df in [None, 5, 3]:
        rej = 0
        for r in range(300):
            N_ = L + 1
            e = tdraw(df, N_); c = rng.choice([-1., 1.], size=L)
            if design == 'regular':
                s = 1 + 0.65 * c; x = s * e[1:]; y = s * e[:-1]
            else:
                x = e[1:]; y = e[:-1]
            fwd = prepare_direction(y, np.column_stack([x, c]), m, te, basis_state)
            rev = prepare_direction(x, np.column_stack([y, c]), m, te, basis_state)
            D, p = contrast_test([(fwd, rev)], [1.0], rng)
            rej += p <= 0.05
        Rrow.append(dict(design=design, df='Gaussian' if df is None else df, reps=300, reject=rej / 300))
        print('R', design, df, rej / 300, flush=True)
out['heavy_tails'] = Rrow
json.dump(out, open(P / 'fan_pelger_results.json', 'w'), indent=1)
print('done')
