"""CONFIRMATORY ADDENDUM (fresh seeds; see PRESPECIFICATION_ADDENDUM.md).

Part RB  operator and co-skewness recursive tests under strongly persistent volatility
         (the designs in which the whitening region undercovered), blocks of ten as in equity.
Part JW  weak identification: Gaussian and weakly non-Gaussian shocks in the joint
         rotation/whitening experiment.
Part WB  long-block rule ell = ceil(sqrt(n)) for the studentized whitening region in the
         eight distinct whitening designs of the main study.

Usage: python confirm_addendum.py RB|JW|WB [workers]
"""
import os, sys, json, time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import designs_addendum as da
from designs import A_ROT, A_REC
from whiten_region import studentized_region, contains
from svarcore import var_fit, var_regenerate, whiten, rot, coskew
from certified_revision import CURV
from confirm_study import (stats_fast, distances_fast, rec, resample, q_higher, l_of, G, true_cell,
                           W_DESIGNS, RES, B_WHITE, B_OP)

ROOT_SEED = 2026092800          # addendum seed family (never used before)

RB_DESIGNS = {
    'R4_logsv_n600': (dict(A=A_REC, eta='exp', vol=('logsv', .95, .385)), 600, 10),
    'R5_mk98_mix_n600': (dict(A=A_REC, eta='mix', vol=('markov2', 0.4, .98)), 600, 10),
}
JW_DESIGNS = {
    'J4_gauss_n300': (dict(A=A_ROT, eta='gauss', vol=('iid2', 0.0)), 300, 1),
    'J5_chi40_n300': (dict(A=A_ROT, eta='chi40', vol=('iid2', 0.0)), 300, 1),
}
WB_NAMES = ['V1_iid_d0_n300', 'V2_iid_d4_n300', 'V3_mk95_d4_n300', 'V4_iid_d4_n600',
            'V5_mk95_d4_n600', 'V6_logsv_n600', 'V7_mk98_mix_n600', 'V8_iid_t5_n600']
WB_DESIGNS = {name.split('_')[0] + 'L_' + '_'.join(name.split('_')[1:]): (W_DESIGNS[name][0], W_DESIGNS[name][1],
              int(np.ceil(np.sqrt(W_DESIGNS[name][1])))) for name in WB_NAMES}
ALL = {'RB': RB_DESIGNS, 'JW': JW_DESIGNS, 'WB': WB_DESIGNS}
REPS = {'RB': 500, 'JW': 500, 'WB': 1000}
TAG = {'RB': 4, 'JW': 5, 'WB': 6}


def seeds_for(part, name, reps):
    idx = list(ALL[part]).index(name)
    return [int(s) for s in np.random.SeedSequence([ROOT_SEED, TAG[part], idx]).generate_state(reps)]


def run_RB(task):
    name, seed = task
    d, n, ell = RB_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = da.simulate(d, n, rng)
    cpop = da.pop_norm(d['eta'], d['vol'])
    Bh, U = var_fit(y, 1); U = U - U.mean(0)
    Z, L = whiten(U); aa = rec(L); origr = stats_fast(Z, aa)
    DR = []; CS = []
    for b in range(B_OP):
        ys = var_regenerate(Bh, y[:1], resample(U, ell, rng))
        _, ub = var_fit(ys, 1); zb, lb = whiten(ub); a = rec(lb)
        DR.append(distances_fast(zb, Z, a, aa, origr)); CS.append(coskew(zb, a))
    DR = np.array(DR); CS = np.array(CS)
    qr = q_higher(DR.max(1), .95)
    norms = np.sqrt(np.maximum(origr, 0)); lower = norms - qr
    cs = coskew(Z, aa)
    W = [float(cs[j] @ np.linalg.solve(np.cov(CS[:, j, :].T), cs[j])) for j in range(2)]
    return dict(seed=seed, rho=float(rho), cpop=cpop, qr=qr, norm_true=float(norms[0]), norm_false=float(norms[1]),
                op_cover=bool(lower[0] <= cpop), op_reject_true=bool(lower[0] > 0),
                op_reject_false=bool(lower[1] > 0), budget_violation=bool(np.sqrt(max(lower[0], 0)) > rho),
                W_true=W[0], W_false=W[1], angle_false=float(np.rad2deg(aa[1])))


def run_JW(task):
    name, seed = task
    d, n, ell = JW_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = da.simulate(d, n, rng)
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
        a, b = orig[j:j + 2]; c = da_inner(Z, j); v = max(a + b - 2 * c, 0)
        t = np.clip((a - c) / v, 0, 1) if v > 1e-16 else 0
        low.append(max(0, np.sqrt(max(a + 2 * t * (c - a) + t * t * v, 0)) - q - CURV * (G[j + 1] - G[j]) ** 2 / 8))
    low = np.array(low)
    reg = studentized_region(y, 1, [0, 1], ell, B_WHITE, rng)
    old_cover = bool(np.linalg.norm(np.linalg.solve(L, L0 - L), 'fro') <= r_old)
    new_cover = contains(reg, l_of(L0))
    rot_cover = bool(low[cell] <= 0)
    return dict(seed=seed, rho=float(rho), q=q, r_old=r_old, c_new=reg['c'], rot_cover=rot_cover,
                L_old=old_cover, L_new=new_cover, joint_old=bool(rot_cover and old_cover),
                joint_new=bool(rot_cover and new_cover), ind_share=float(np.mean(low <= 0)))


def da_inner(Z, j):
    from fastop import op_inner_fast
    return op_inner_fast(Z @ rot(G[j]), Z @ rot(G[j + 1]))


def run_WB(task):
    name, seed = task
    d, n, ell = WB_DESIGNS[name]
    rng = np.random.default_rng(seed)
    y, L0, rho = da.simulate(d, n, rng)
    reg = studentized_region(y, 1, [0, 1], ell, B_WHITE, rng)
    return dict(seed=seed, ell=ell, cover=contains(reg, l_of(L0)), c=reg['c'])


RUN = {'RB': run_RB, 'JW': run_JW, 'WB': run_WB}


def main(part, workers):
    reps = REPS[part]
    for name in ALL[part]:
        path = os.path.join(RES, f'addendum_{part}_{name}.jsonl')
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
