"""Generic nuisance derivatives for the joint DII bootstrap by finite differences.

The analytic operator_matrices() in dii_joint_nuisance.py covers fixed polynomial
means, target scales, and regressor centers/scales with unit bandwidth. This module
computes the same objects (A, M, J) numerically from a feature map
    e -> (residual features, regressor features)
so that two extensions use the identical bootstrap formula:
  1. an estimated factor loading that enters the reverse regressor
     (defactored outcome Y - lambda f used as a regressor), and
  2. a fixed bandwidth multiplier c for both kernels.
Conventions follow verify_joint_nuisance.py: M is the row-mean derivative of the
cross operator and J the mixed second derivative of <C(a), C(b)>.
"""
import numpy as np
from full_study import center


def _cross(fa, fb, c):
    ua, za = fa; ub, zb = fb
    K = np.exp(-.5 * ((ua[:, None] - ub[None, :]) / c) ** 2)
    L = np.exp(-.5 * np.sum(((za[:, None, :] - zb[None, :, :]) / c) ** 2, axis=2))
    return center(K) * center(L)


def fd_matrices(features, q, c=1.0, eps=1e-4):
    zero = np.zeros(q); eye = np.eye(q) * eps
    f0 = features(zero)
    A = _cross(f0, f0, c)
    n = A.shape[0]
    fp = [features(e) for e in eye]; fm = [features(-e) for e in eye]
    M = np.column_stack([(_cross(f0, fp[k], c) - _cross(f0, fm[k], c)).sum(1) / (2 * eps * n) for k in range(q)])
    J = np.zeros((q, q))
    for a in range(q):
        for b in range(a, q):
            v = (_cross(fp[a], fp[b], c).sum() - _cross(fp[a], fm[b], c).sum()
                 - _cross(fm[a], fp[b], c).sum() + _cross(fm[a], fm[b], c).sum()) / (4 * eps ** 2 * n * n)
            J[a, b] = J[b, a] = v
    return A, M, J


def standard_direction(target, regressors, m, te, basis):
    """Same estimand and score as prepare_direction, with a feature map for fd_matrices."""
    target = np.asarray(target); z = np.atleast_2d(np.asarray(regressors).T).T
    ty = (target - target[:m].mean()) / target[:m].std()
    zz = (z - z[:m].mean(0)) / z[:m].std(0)
    P = basis(zz); beta = np.linalg.lstsq(P[:m], ty[:m], rcond=None)[0]
    u = ty - P @ beta; Q = P[:m].T @ P[:m] / m
    if np.linalg.matrix_rank(Q) != Q.shape[0]:
        raise ValueError('Singular design')
    score = np.column_stack([(P[:m] * u[:m, None]) @ np.linalg.inv(Q),
                             .5 * (ty[:m] ** 2 - 1), .5 * (zz[:m] ** 2 - 1), zz[:m]])
    p, k = P.shape[1], zz.shape[1]
    ue, Pe, ze = u[te], P[te], zz[te]

    def features(e):
        return (ue - Pe @ e[:p]) * np.exp(-e[p]), (ze - e[p + 1 + k:]) * np.exp(-e[p + 1:p + 1 + k])
    return dict(features=features, q=p + 1 + 2 * k, score=score, u=ue, z=ze)


def defactored_pair(y, x, f, m, te, basis):
    """Forward: Y on basis(X) plus f (f in the mean, X in the kernel).
    Reverse: X on basis(Y - lambda_hat f), with lambda_hat from the forward training fit.
    The reverse feature map refits standardization and OLS at a perturbed lambda, and the
    reverse score gains the training influence of lambda_hat. Returns (forward, reverse)."""
    y = np.asarray(y); x = np.asarray(x); f = np.asarray(f)
    sy = y[:m].std(); ty = (y - y[:m].mean()) / sy
    xs = (x - x[:m].mean()) / x[:m].std()
    Pf = np.column_stack([basis(xs[:, None]), f]); bf = np.linalg.lstsq(Pf[:m], ty[:m], rcond=None)[0]
    uf = ty - Pf @ bf; Qf = Pf[:m].T @ Pf[:m] / m
    sbf = (Pf[:m] * uf[:m, None]) @ np.linalg.inv(Qf)
    score_f = np.column_stack([sbf, .5 * (ty[:m] ** 2 - 1), .5 * (xs[:m] ** 2 - 1), xs[:m]])
    pf = Pf.shape[1]; ufe, Pfe, xe = uf[te], Pf[te], xs[te]

    def feat_f(e):
        return (ufe - Pfe @ e[:pf]) * np.exp(-e[pf]), ((xe - e[pf + 2]) * np.exp(-e[pf + 1]))[:, None]
    fwd = dict(features=feat_f, q=pf + 3, score=score_f)

    lam = bf[-1] * sy
    infl_lam = sy * sbf[:, -1]   # lambda = s_Y * beta_f; the scale effects cancel, leaving the raw-unit OLS influence
    tx = (x - x[:m].mean()) / x[:m].std()

    def reverse_at(l):
        yt = y - l * f
        zz = (yt - yt[:m].mean()) / yt[:m].std()
        P = basis(zz[:, None]); b = np.linalg.lstsq(P[:m], tx[:m], rcond=None)[0]
        return tx - P @ b, P, zz, b

    ur, Pr, zr, br = reverse_at(lam)
    Qr = Pr[:m].T @ Pr[:m] / m
    score_r = np.column_stack([(Pr[:m] * ur[:m, None]) @ np.linalg.inv(Qr),
                               .5 * (tx[:m] ** 2 - 1), .5 * (zr[:m] ** 2 - 1), zr[:m], infl_lam])
    pr = Pr.shape[1]; cache = {}

    def feat_r(e):
        key = round(float(e[-1]), 12)
        if key not in cache:
            u_, P_, z_, _ = reverse_at(lam + e[-1]); cache[key] = (u_[te], P_[te], z_[te])
        u_, P_, z_ = cache[key]
        return (u_ - P_ @ e[:pr]) * np.exp(-e[pr]), ((z_ - e[pr + 2]) * np.exp(-e[pr + 1]))[:, None]
    rev = dict(features=feat_r, q=pr + 4, score=score_r, lam=lam, reverse_at=reverse_at)
    return fwd, rev


def contrast_fd(pairs, weights, rng, m, n, B, c=1.0):
    """Joint p-value for sum_j w_j (H_b,j - H_f,j) using fd_matrices; common paths."""
    from dii_training_aware import count_weights
    W = count_weights(rng, n, B); WT = count_weights(rng, m, B)
    D = 0.0; q = np.zeros(B); l = np.zeros(B)
    for (fwd, rev), w in zip(pairs, weights):
        for d, sgn in [(fwd, -1.0), (rev, 1.0)]:
            A, M, J = fd_matrices(d['features'], d['q'], c)
            db = WT @ d['score'] / m
            H = float(A.sum() / n ** 2)
            quad = np.sum((W @ A) * W, axis=1) / n ** 2 + 2 * np.sum((W @ M) * db, axis=1) / n + np.sum((db @ J) * db, axis=1)
            lin = 2 * (W @ A.sum(0) / n ** 2 + db @ M.mean(0))
            D += sgn * w * H; q += sgn * w * np.maximum(quad, 0); l += sgn * w * lin
    pq = (1 + np.count_nonzero(q >= D - 1e-12)) / (B + 1)
    pl = (1 + np.count_nonzero(l >= D - 1e-12)) / (B + 1)
    return D, max(pq, pl)
