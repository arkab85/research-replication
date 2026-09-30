"""Calibration of the recursive-ordering test and of the proxy-adjusted recursive test.

NOT part of the pre-specified protocols in code/whitening: designed after the applications,
and reported as such.  Designs mirror the recursive-test protocol (true first-ordered structure
A = [[1, 0], [.5, 1]], demeaned unit-exponential eta, n = 600) with persistent two-state common
volatility (stay probability .95, delta = .4, rho = 0.371) and a predetermined continuous proxy
    W_t = zeta_t + kappa * xi_t,  xi_t ~ N(0, 1),
where zeta_t = +-1 is the volatility state, so kappa = 0 reveals the state (residual budget 0 with
k = 2 regimes) and kappa > 0 leaves residual exposure.  The true residual budget
rho^{|W_k} = {E Var(sigma | W_k)}^{1/2} is computed by simulation (10^6 draws).

For each replication: raw and adjusted (k = 2) operator norms at the true ordering (angle 0) and
the data-dependent second ordering, iid bootstrap (the design has no VAR dynamics, so innovations
are resampled in moving blocks of ten to respect the persistent state), 199 draws, joint .95
critical value; reports the breakdown lower bound at the true ordering.
Run: OPENBLAS_NUM_THREADS=1 python benchmark/adjusted_calibration.py KAPPA
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index
from acore import op_inner_cond, quantile_bins, dev, qtile

N, B, BLOCK, K = 600, 199, 10, 2
REPS = int(os.environ.get("REPS", 200))
DELTA, STAY = .4, .95
A = np.array([[1., 0.], [.5, 1.]])


DESIGN = dict(vol='twostate', eta='exp', n=N, k=K, block=BLOCK)
LN_PHI = .95
LN_SD = float(np.sqrt(np.log(1 + (DELTA / np.sqrt(1 + DELTA ** 2)) ** 2 / (1 - (DELTA / np.sqrt(1 + DELTA ** 2)) ** 2)) * 1.0))


def vol_path(rng, n):
    if DESIGN['vol'] == 'twostate':
        zeta = np.empty(n); zeta[0] = rng.choice([-1., 1.])
        for t in range(1, n):
            zeta[t] = zeta[t - 1] if rng.random() < STAY else -zeta[t - 1]
        return zeta, (1 + DELTA * zeta) / np.sqrt(1 + DELTA ** 2)
    # log-normal AR(1) log-volatility with stationary sd LN_SD: sigma = exp(h - var h), E sigma^2 = 1
    h = np.empty(n); h[0] = rng.standard_normal() * LN_SD
    e = rng.standard_normal(n) * LN_SD * np.sqrt(1 - LN_PHI ** 2)
    for t in range(1, n):
        h[t] = LN_PHI * h[t - 1] + e[t]
    return h / LN_SD, np.exp(h - LN_SD ** 2)


def draw_eta(rng, n):
    if DESIGN['eta'] == 'exp':
        return rng.exponential(size=(n, 2)) - 1
    x = rng.standard_t(5, size=(n, 2)); return x / np.sqrt(5 / 3)


def simulate(rng, kappa, n=None):
    n = n or DESIGN['n']
    state, sig = vol_path(rng, n)
    eps = sig[:, None] * draw_eta(rng, n)
    W = state + kappa * rng.standard_normal(n)
    return eps @ A.T, W, sig


def true_residual_budget(kappa, rng):
    m = 10 ** 6
    if DESIGN['vol'] == 'twostate':
        state = rng.choice([-1., 1.], m); sig = (1 + DELTA * state) / np.sqrt(1 + DELTA ** 2)
    else:
        h = rng.standard_normal(m) * LN_SD; state = h / LN_SD; sig = np.exp(h - LN_SD ** 2)
    W = state + kappa * rng.standard_normal(m); k = DESIGN['k']
    b = np.searchsorted(np.quantile(W, np.arange(1, k) / k), W)
    return float(np.sqrt(sum((b == j).mean() * sig[b == j].var() for j in range(k)))), float(sig.std())


def second_angle(L):
    return float(np.arctan2(-L[1, 0], L[1, 1]) % (np.pi / 2))


def one(rng, kappa):
    u, W, _ = simulate(rng, kappa)
    u = u - u.mean(0); n = len(u)
    z, L = whiten(u)
    ang = [0., second_angle(L)]
    E0 = [z @ rot(t) for t in ang]
    K = DESIGN['k']; b0 = quantile_bins(W, K); one0 = np.zeros(n, int)
    vr = [op_inner_cond(E, one0, E, one0, 1) for E in E0]
    va = [op_inner_cond(E, b0, E, b0, K) for E in E0]
    Dr = np.empty((B, 2)); Da = np.empty((B, 2))
    for i in range(B):
        idx = boot_index(n, DESIGN['block'], rng)
        ub = u[idx] - u[idx].mean(0); zb, Lb = whiten(ub); Wb = W[idx]
        angb = [0., second_angle(Lb)]; Eb = [zb @ rot(t) for t in angb]
        bb = quantile_bins(Wb, K); oneb = np.zeros(n, int)
        Dr[i] = [dev(Eb[j], oneb, E0[j], one0, 1, vr[j]) for j in range(2)]
        Da[i] = [dev(Eb[j], bb, E0[j], b0, K, va[j]) for j in range(2)]
    qr, qa = qtile(Dr.max(1), .95), qtile(Da.max(1), .95)
    br = np.sqrt(np.maximum(np.sqrt(vr) - qr, 0)); ba = np.sqrt(np.maximum(np.sqrt(va) - qa, 0))
    return br.tolist(), ba.tolist(), np.sqrt(vr).tolist(), np.sqrt(va).tolist()


def main(kappa, tag=''):
    rng = np.random.default_rng([2026092706, int(round(kappa * 100))] + ([sum(map(ord, tag))] if tag else []))
    rhoW, rho = true_residual_budget(kappa, rng)
    out = dict(kappa=kappa, rho=rho, rho_residual=rhoW, reps=REPS, B=B, design=DESIGN, reps_out=[])
    t0 = time.time()
    for r in range(REPS):
        out['reps_out'].append(one(rng, kappa))
        if r % 20 == 0:
            print(kappa, r, round(time.time() - t0), flush=True)
    br = np.array([x[0][0] for x in out['reps_out']]); ba = np.array([x[1][0] for x in out['reps_out']])
    brf = np.array([x[0][1] for x in out['reps_out']]); baf = np.array([x[1][1] for x in out['reps_out']])
    out['summary'] = dict(
        raw_true_reject=float((br > 0).mean()), raw_true_exceeds_rho=float((br > rho).mean()),
        adj_true_reject=float((ba > 0).mean()), adj_true_exceeds_rho_residual=float((ba > rhoW).mean()),
        raw_false_reject=float((brf > 0).mean()), adj_false_reject=float((baf > 0).mean()),
        raw_true_mean_norm=float(np.mean([x[2][0] for x in out['reps_out']])),
        adj_true_mean_norm=float(np.mean([x[3][0] for x in out['reps_out']])))
    print(json.dumps(out['summary']), 'rho_residual', rhoW)
    name = f'adjusted_calibration_k{int(round(kappa*100))}' + (f'_{tag}' if tag else '')
    json.dump(out, open(os.path.join(HERE, 'results', name + '.json'), 'w'), indent=1)


EXTRA = {  # added designs (also outside the protocols)
    'lognormal': (dict(vol='lognormal', eta='exp', n=600, k=3, block=10), 0.5),
    'weakproxy': (dict(vol='twostate', eta='exp', n=600, k=2, block=10), 1.5),
    'fomclike': (dict(vol='twostate', eta='t5', n=325, k=3, block=19), 0.5),
}

if __name__ == '__main__':
    if sys.argv[1] in EXTRA:
        d, kap = EXTRA[sys.argv[1]]; DESIGN.update(d); main(kap, sys.argv[1])
    else:
        main(float(sys.argv[1]))
