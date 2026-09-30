"""Checks for dii_fd_nuisance.py.
1. fd_matrices reproduces the analytic operator_matrices on the standard estimand.
2. For the defactored pair, the quadratic form d' J d along a training-weight
   perturbation equals the second difference of a full weighted refit (forward OLS
   with the loading, reverse standardization and OLS at the refitted loading).
3. The same holds with bandwidth multipliers 0.5 and 2 on the standard estimand.
"""
from pathlib import Path
import json, sys
import numpy as np
S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
from dii_joint_nuisance import prepare_direction, operator_matrices
from dii_fd_nuisance import fd_matrices, standard_direction, defactored_pair, _cross


def basis(z):
    z = np.atleast_2d(z)
    return np.column_stack([np.ones(len(z)), z[:, 0], z[:, 0] ** 2])


rng = np.random.default_rng(919)
m, n = 96, 48
te = np.arange(m, m + n)
x = rng.normal(size=m + n); f = rng.normal(size=m + n)
y = .5 * (x ** 2 - 1) + .8 * f + rng.normal(size=m + n)
out = {}

# 1
d = prepare_direction(y, x[:, None], m, te, basis); A, M, J = operator_matrices(d)
s = standard_direction(y, x[:, None], m, te, basis); A2, M2, J2 = fd_matrices(s['features'], s['q'])
out['standard_max_error_M'] = float(np.max(abs(M - M2))); out['standard_max_error_J'] = float(np.max(abs(J - J2)))
assert np.allclose(s['score'], d['score']) and out['standard_max_error_M'] < 1e-6 and out['standard_max_error_J'] < 1e-6, out

# 2
fwd, rev = defactored_pair(y, x, f, m, te, basis)
v = rng.normal(size=m); v -= v.mean(); ep = 1e-4


def refit(sign):
    w = 1 + sign * ep * v
    def wmean(a): return np.average(a[:m], axis=0, weights=w)
    def wsd(a): return np.sqrt(np.average((a[:m] - wmean(a)) ** 2, axis=0, weights=w))
    ty = (y - wmean(y)) / wsd(y); xs = (x - wmean(x)) / wsd(x)
    Pf = np.column_stack([basis(xs[:, None]), f])
    bf = np.linalg.solve(Pf[:m].T @ (w[:, None] * Pf[:m]), Pf[:m].T @ (w * ty[:m]))
    uf = ty - Pf @ bf; lam = bf[-1] * wsd(y)
    tx = (x - wmean(x)) / wsd(x); yt = y - lam * f; zz = (yt - wmean(yt)) / wsd(yt)
    Pr = basis(zz[:, None]); br = np.linalg.solve(Pr[:m].T @ (w[:, None] * Pr[:m]), Pr[:m].T @ (w * tx[:m]))
    ur = tx - Pr @ br
    return (uf[te], xs[te][:, None]), (ur[te], zz[te][:, None])


def inner(a, b, c=1.0): return _cross(a, b, c).sum() / n ** 2


pl, mi = refit(1), refit(-1)
for k, dd in [(0, fwd), (1, rev)]:
    A_, M_, J_ = fd_matrices(dd['features'], dd['q'])
    direction = v @ dd['score'] / m
    analytic = direction @ J_ @ direction
    refitn = (inner(pl[k], pl[k]) + inner(mi[k], mi[k]) - 2 * inner(pl[k], mi[k])) / (4 * ep ** 2)
    rel = abs(refitn - analytic) / max(abs(refitn), 1e-12)
    out[f"defactored_{'forward' if k == 0 else 'reverse'}_refit_rel_error"] = float(rel)
    assert rel < 1e-3, (k, refitn, analytic)

# 3
for c in [0.5, 2.0]:
    s = standard_direction(y, x[:, None], m, te, basis)
    A_, M_, J_ = fd_matrices(s['features'], s['q'], c)
    def refit_std(sign):
        w = 1 + sign * ep * v
        def wmean(a): return np.average(a[:m], axis=0, weights=w)
        def wsd(a): return np.sqrt(np.average((a[:m] - wmean(a)) ** 2, axis=0, weights=w))
        ty = (y - wmean(y)) / wsd(y); xs = (x - wmean(x)) / wsd(x); P = basis(xs[:, None])
        b = np.linalg.solve(P[:m].T @ (w[:, None] * P[:m]), P[:m].T @ (w * ty[:m]))
        return (ty - P @ b)[te], xs[te][:, None]
    p_, m_ = refit_std(1), refit_std(-1)
    direction = v @ s['score'] / m
    refitn = (inner(p_, p_, c) + inner(m_, m_, c) - 2 * inner(p_, m_, c)) / (4 * ep ** 2)
    rel = abs(refitn - direction @ J_ @ direction) / max(abs(refitn), 1e-12)
    out[f'bandwidth_{c}_refit_rel_error'] = float(rel)
    assert rel < 1e-3, (c, rel)

(S.parent / 'fd_nuisance_verification.json').write_text(json.dumps(out, indent=2))
print(out)
