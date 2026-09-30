"""Deterministic checks for the studentised whitening region and its projection."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wcore import chol2, jac_chol2, lrv_bartlett, lrv_blocks, order_quantile
from whiten_region import studentized_region, contains, entry_bounds, elasticity_cells, rot
from fastop import op_inner_fast
from certified_revision import op_inner
from popc import pop_norm_two_point
rng = np.random.default_rng(20260926)
out = {}
# 1. fast operator inner product equals the certified implementation
err = 0
for n1, n2 in ((50, 50), (80, 120), (300, 300)):
    E = rng.normal(size=(n1, 2)); F = rng.standard_t(4, size=(n2, 2))
    err = max(err, abs(op_inner(E, F) - op_inner_fast(E, F)))
assert err < 1e-12; out['fast_operator_max_abs_error'] = err
# 2. Cholesky Jacobian against central finite differences
S = np.array([[1.3, .4], [.4, .9]]); l = chol2(S); J = jac_chol2(l); h = 1e-6; Jn = np.zeros((3, 3))
for j, (a, b) in enumerate(((0, 0), (1, 0), (1, 1))):
    E = np.zeros((2, 2)); E[a, b] = E[b, a] = 1
    Jn[:, j] = (chol2(S + h * E) - chol2(S - h * E)) / (2 * h)
assert np.max(abs(J - Jn)) < 1e-8; out['cholesky_jacobian_fd_error'] = float(np.max(abs(J - Jn)))
# 3. block estimators reduce to the sample covariance with ell=1
X = rng.normal(size=(200, 3))
assert np.allclose(lrv_bartlett(X, 1), lrv_blocks(X, 1)); out['ell1_identity'] = 'pass'
# 4. order-statistic quantile
assert order_quantile(np.arange(1, 200), .975) == 195; out['order_quantile_B199'] = 195
# 5. projection: sampled members of the ellipsoid and angles in a cell lie inside certified bounds
y = np.cumsum(rng.normal(size=(301, 2)), 0) * .1 + rng.normal(size=(301, 2))
reg = studentized_region(y, 1, [0, 1], 1, 199, rng)
TH = np.linspace(0, np.pi / 2, 31); bounds = elasticity_cells(reg, TH)
Lc = np.linalg.cholesky(reg['V']); viol = 0; checked = 0
for _ in range(4000):
    j = int(rng.integers(30)); t = rng.uniform(TH[j], TH[j + 1])
    v = rng.normal(size=3); v *= rng.uniform() ** (1 / 3) / np.linalg.norm(v)
    lv = reg['l'] + np.sqrt(reg['c'] / reg['n']) * Lc @ v
    assert contains(reg, lv)
    L = np.array([[lv[0], 0], [lv[1], lv[2]]]); A = L @ rot(t)
    lo, hi = entry_bounds(reg, TH[j], TH[j + 1])
    assert np.all(A >= lo - 1e-12) and np.all(A <= hi + 1e-12)
    for k in range(2):
        if A[0, k] * A[1, k] > 0:
            e = A[0, k] / A[1, k]; checked += 1
            assert bounds[j, 0] - 1e-12 <= e <= bounds[j, 1] + 1e-12
out['projection_draws'] = 4000; out['elasticity_checks'] = checked
# 6. population operator value used in the recursive-test study
out['pop_norm_delta_0.4'] = pop_norm_two_point(.4); out['pop_norm_delta_0'] = pop_norm_two_point(0.)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_whitening.json'), 'w'), indent=2)
print(out)
