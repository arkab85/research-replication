"""Experiments for the two extensions in dii_fd_nuisance.py (seed 20260922).
L: panel pooling with a training-estimated factor loading whose uncertainty is
   propagated in both directions (same design as Table 5).
W: bandwidth multipliers c in {0.5, 1, 2} (fixed after training standardization):
   size at the equal-positive and double-independence nulls, power against a
   quadratic exposure.
Writes fd_experiments_results.json after each cell.
"""
from pathlib import Path
import json, sys
import numpy as np
S = Path(__file__).resolve().parent
P = S.parent
sys.path.insert(0, str(S))
from dii_fd_nuisance import defactored_pair, standard_direction, contrast_fd  # noqa: E402

m, gap, n, B = 204, 39, 120, 399
te = np.arange(m + gap, m + gap + n); L = m + gap + n
OUT = P / 'fd_experiments_results.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {'settings': dict(m=m, gap=gap, n=n, draws=B, seed=20260922), 'loading': [], 'bandwidth': []}
part = sys.argv[1]
import time
T0 = time.time(); BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
res.setdefault('partial', {})


def basis1(z):
    z = np.atleast_2d(z)
    return np.column_stack([np.ones(len(z)), z[:, 0], z[:, 0] ** 2])


def basis_state(z):
    z = np.atleast_2d(z)
    return np.column_stack([np.ones(len(z)), z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 0] * z[:, 1]])


def save():
    OUT.write_text(json.dumps(res, indent=1))


if part == 'loading':
    for theta in [0.0, 0.35]:
        for N, reps in [(1, 200), (4, 200), (16, 100)]:
            if any(r['theta'] == theta and r['N'] == N for r in res['loading']):
                continue
            key = f'L_{theta}_{N}'
            rng = np.random.default_rng([20260922, 1, int(100 * theta), N])   # per-cell seed
            rej, start = res['partial'].get(key, [0, 0])
            for r in range(reps):
                x = rng.normal(size=L); f = rng.normal(size=L)
                Y = theta * (x ** 2 - 1)[:, None] + f[:, None] + rng.normal(size=(L, N))
                pairs_rng_state = None
                if r < start:
                    from dii_training_aware import count_weights
                    count_weights(rng, n, B); count_weights(rng, m, B)   # advance the stream identically
                    continue
                pairs = [defactored_pair(Y[:, i], x, f, m, te, basis1) for i in range(N)]
                D, p = contrast_fd(pairs, [1.0 / N] * N, rng, m, n, B)
                rej += int(p <= 0.05)
                res['partial'][key] = [rej, r + 1]; save()
                if time.time() - T0 > BUDGET:
                    sys.exit(3)
            res['loading'].append(dict(theta=theta, N=N, reps=reps, reject=rej / reps)); save()
            print('L', theta, N, rej / reps, flush=True)

if part == 'bandwidth':
    for c in [0.5, 1.0, 2.0]:
        for design, reps in [('regular', 300), ('double', 300), ('quadratic', 200)]:
            if any(r['c'] == c and r['design'] == design for r in res['bandwidth']):
                continue
            key = f'W_{c}_{design}'
            rng = np.random.default_rng([20260922, 2, int(10 * c), ['regular', 'double', 'quadratic'].index(design)])   # per-cell seed
            rej, start = res['partial'].get(key, [0, 0])
            for r in range(reps):
                if design == 'quadratic':
                    x = rng.normal(size=L); y = 0.5 * (x ** 2 - 1) + rng.normal(size=L)
                    fw = standard_direction(y, x[:, None], m, te, basis1); rv = standard_direction(x, y[:, None], m, te, basis1)
                else:
                    e = rng.normal(size=L + 1); st = rng.choice([-1., 1.], size=L)
                    sc = 1 + 0.65 * st if design == 'regular' else np.ones(L)
                    x = sc * e[1:]; y = sc * e[:-1]
                    fw = standard_direction(y, np.column_stack([x, st]), m, te, basis_state)
                    rv = standard_direction(x, np.column_stack([y, st]), m, te, basis_state)
                if r < start:
                    from dii_training_aware import count_weights
                    count_weights(rng, n, B); count_weights(rng, m, B)
                    continue
                D, p = contrast_fd([(fw, rv)], [1.0], rng, m, n, B, c)
                rej += int(p <= 0.05)
                res['partial'][key] = [rej, r + 1]; save()
                if time.time() - T0 > BUDGET:
                    sys.exit(3)
            res['bandwidth'].append(dict(c=c, design=design, reps=reps, reject=rej / reps)); save()
            print('W', c, design, rej / reps, flush=True)
