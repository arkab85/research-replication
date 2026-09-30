"""CONFIRMATORY calibration study (fresh seeds; see PRESPECIFICATION.md).

Part W  whitening-region coverage of the studentised procedure, 10 designs.
Part J  joint rotation/whitening coverage on a fixed 6-degree grid; earlier
        percentile region and studentised region evaluated on the same data.
Part R  size/coverage of the recursive-ordering operator test and the
        co-skewness Wald test at a true recursive ordering (equity procedure).

Usage: python confirm_study.py W|J|R [workers]
Raw replication rows are appended to results/confirm_<part>_<design>.jsonl and
the run resumes from completed seeds.
"""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from designs import simulate, A_ROT, A_REC
from whiten_region import studentized_region, contains
from fastop import op_inner_fast
from svarcore import var_fit, var_regenerate, whiten, rot, coskew, mbb_indices
from certified_revision import CURV
from popc import pop_norm_two_point

RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
ROOT_SEED = 2026092600          # confirmatory seed family (never used in development)
B_WHITE = 1999                  # whitening bootstrap draws
B_OP = 199                      # operator bootstrap draws

W_DESIGNS = {
    'V1_iid_d0_n300':      (dict(A=A_ROT, eta='exp', vol=('iid2', 0.0)), 300, 1),
    'V2_iid_d4_n300':      (dict(A=A_ROT, eta='exp', vol=('iid2', 0.4)), 300, 1),
    'V3_mk95_d4_n300':     (dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, .95)), 300, 10),
    'V3b_mk95_d4_n300_l20': (dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, .95)), 300, 20),
    'V4_iid_d4_n600':      (dict(A=A_ROT, eta='exp', vol=('iid2', 0.4)), 600, 1),
    'V5_mk95_d4_n600':     (dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, .95)), 600, 10),
    'V5b_mk95_d4_n600_l20': (dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, .95)), 600, 20),
    'V6_logsv_n600':       (dict(A=A_ROT, eta='exp', vol=('logsv', .95, .385)), 600, 10),
    'V7_mk98_mix_n600':    (dict(A=A_ROT, eta='mix', vol=('markov2', 0.4, .98)), 600, 10),
    'V8_iid_t5_n600':      (dict(A=A_ROT, eta='t5', vol=('iid2', 0.4)), 600, 1),
}
J_DESIGNS = {
    'J1_iid_d0_n300':  (dict(A=A_ROT, eta='exp', vol=('iid2', 0.0)), 300, 1),
    'J2_iid_d4_n300':  (dict(A=A_ROT, eta='exp', vol=('iid2', 0.4)), 300, 1),
    'J3_mk95_d4_n300': (dict(A=A_ROT, eta='exp', vol=('markov2', 0.4, .95)), 300, 10),
}
R_DESIGNS = {
    'R1_iid_d0_n600':  (dict(A=A_REC, eta='exp', vol=('iid2', 0.0)), 600, 10),
    'R2_iid_d4_n600':  (dict(A=A_REC, eta='exp', vol=('iid2', 0.4)), 600, 10),
    'R3_mk95_d4_n600': (dict(A=A_REC, eta='exp', vol=('markov2', 0.4, .95)), 600, 10),
}
REPS = {'W': 1000, 'J': 500, 'R': 500}
ALL = {'W': W_DESIGNS, 'J': J_DESIGNS, 'R': R_DESIGNS}


def seeds_for(part, name, reps):
    idx = list(ALL[part]).index(name)
    tag = {'W': 1, 'J': 2, 'R': 3}[part]
    return [int(s) for s in np.random.SeedSequence([ROOT_SEED, tag, idx]).generate_state(reps)]


# ----------------------------------------------------------------- helpers
def stats_fast(Z, angles):
    return np.array([op_inner_fast(Z @ rot(t), Z @ rot(t)) for t in angles])


def distances_fast(Zb, Z, ab, ao, orig):
    out = []
    for t, s, v in zip(ab, ao, orig):
        Eb = Zb @ rot(t); E = Z @ rot(s)
        out.append(np.sqrt(max(0.0, op_inner_fast(Eb, Eb) + v - 2 * op_inner_fast(Eb, E))))
    return np.array(out)


def rec(L):
    return np.array([0., np.arctan2(-L[1, 0], L[1, 1]) % (np.pi / 2)])


def resample(U, ell, rng):
    n = len(U)
    return U[rng.integers(0, n, n)] if ell == 1 else U[mbb_indices(n, ell, rng)]


def q_higher(x, level):
    return float(np.quantile(np.asarray(x), level, method='higher'))


def l_of(L):
    return np.array([L[0, 0], L[1, 0], L[1, 1]])


# ------------------------------------------------------------------- parts
def run_W(task):
    name, seed = task
    d, n, ell = W_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(d, n, rng)
    reg = studentized_region(y, 1, [0, 1], ell, B_WHITE, rng)
    return dict(seed=seed, cover=contains(reg, l_of(L0)), c=reg['c'],
                l=reg['l'].tolist(), sd=np.sqrt(np.diag(reg['V']) / n).tolist())


G = np.linspace(0, np.pi / 2, 16)


def true_cell(A):
    L0 = np.linalg.cholesky(A @ A.T); Q0 = np.linalg.solve(L0, A)
    th = np.arctan2(Q0[1, 0], Q0[0, 0]) % (np.pi / 2)
    return min(int(th / (np.pi / 30)), 14), float(th)


def run_J(task):
    name, seed = task
    d, n, ell = J_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(d, n, rng)
    cell, _ = true_cell(d['A'])
    Bh, U = var_fit(y, 1); U = U - U.mean(0)
    Z, L = whiten(U); orig = stats_fast(Z, G)
    D = []; Rold = []
    for b in range(B_OP):
        ys = var_regenerate(Bh, y[:1], resample(U, ell, rng))
        _, ub = var_fit(ys, 1); zb, lb = whiten(ub)
        D.append(np.max(distances_fast(zb, Z, G, G, orig)))
        Rold.append(np.linalg.norm(np.linalg.solve(L, lb - L), 'fro'))
    q = q_higher(D, .975); r_old = q_higher(Rold, .975)
    low = []
    for j in range(15):
        a, b = orig[j:j + 2]; c = op_inner_fast(Z @ rot(G[j]), Z @ rot(G[j + 1])); v = max(a + b - 2 * c, 0)
        t = np.clip((a - c) / v, 0, 1) if v > 1e-16 else 0
        low.append(max(0, np.sqrt(max(a + 2 * t * (c - a) + t * t * v, 0)) - q - CURV * (G[j + 1] - G[j]) ** 2 / 8))
    low = np.array(low); rho2 = rho ** 2
    reg = studentized_region(y, 1, [0, 1], ell, B_WHITE, rng)
    old_cover = bool(np.linalg.norm(np.linalg.solve(L, L0 - L), 'fro') <= r_old)
    new_cover = contains(reg, l_of(L0))
    rot_cover = bool(low[cell] <= rho2)
    return dict(seed=seed, rho=rho, q=q, r_old=r_old, c_new=reg['c'], ind_cover=bool(low[cell] <= 0),
                budget_cover=rot_cover, L_old=old_cover, L_new=new_cover,
                joint_old=bool(rot_cover and old_cover), joint_new=bool(rot_cover and new_cover),
                budget_share=float(np.mean(low <= rho2)), ind_share=float(np.mean(low <= 0)))


POP = {}


def run_R(task):
    name, seed = task
    d, n, ell = R_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = simulate(d, n, rng)
    delta = d['vol'][1]
    cpop = pop_norm_two_point(delta) if delta > 0 else 0.0
    Bh, U = var_fit(y, 1); U = U - U.mean(0)
    Z, L = whiten(U); aa = rec(L); origr = stats_fast(Z, aa)
    DR = []; CS = []
    for b in range(B_OP):
        ys = var_regenerate(Bh, y[:1], resample(U, ell, rng))
        _, ub = var_fit(ys, 1); zb, lb = whiten(ub); a = rec(lb)
        DR.append(distances_fast(zb, Z, a, aa, origr)); CS.append(coskew(zb, a))
    DR = np.array(DR); CS = np.array(CS)
    qr = q_higher(DR.max(1), .95)
    norms = np.sqrt(np.maximum(origr, 0))
    lower = norms - qr
    cs = coskew(Z, aa)
    W = [float(cs[j] @ np.linalg.solve(np.cov(CS[:, j, :].T), cs[j])) for j in range(2)]
    return dict(seed=seed, rho=rho, cpop=cpop, qr=qr, norm_true=float(norms[0]), norm_false=float(norms[1]),
                op_cover=bool(lower[0] <= cpop), op_reject_true=bool(lower[0] > 0),
                op_reject_false=bool(lower[1] > 0), budget_violation=bool(np.sqrt(max(lower[0], 0)) > rho),
                W_true=W[0], W_false=W[1], angle_false=float(np.rad2deg(aa[1])))


RUN = {'W': run_W, 'J': run_J, 'R': run_R}


def main(part, workers):
    reps = REPS[part]
    for name in ALL[part]:
        path = os.path.join(RES, f'confirm_{part}_{name}.jsonl')
        done = set()
        if os.path.exists(path):
            for line in open(path):
                done.add(json.loads(line)['seed'])
        todo = [(name, s) for s in seeds_for(part, name, reps) if s not in done]
        t = time.time()
        if todo:
            with ProcessPoolExecutor(workers) as ex, open(path, 'a') as f:
                for i, row in enumerate(ex.map(RUN[part], todo, chunksize=2)):
                    f.write(json.dumps(row) + '\n'); f.flush()
                    if (i + 1) % 50 == 0:
                        print(part, name, len(done) + i + 1, '/', reps, round(time.time() - t), 's', flush=True)
        print(part, name, 'complete', round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
