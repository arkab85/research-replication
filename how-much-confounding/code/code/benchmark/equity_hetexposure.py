"""Equity application: exposure at the heteroskedasticity-identified rotation.

If the rotation identified by the lagged-VIX regimes (k = 2) is the true one, Theorem 1 gives
nu_1 nu_2 >= ||C(Q_0)|| and Theorem 2 gives the same for the residual exposures and the
regime-conditioned operator (k = 3).  Lower confidence bounds re-estimate the rotation in every
refitted draw (399 draws with the seeds of the main run, blocks of 10 and 33).
Writes results/equity_hetexposure.json.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bcore import np, rot, whiten, boot_index, het_angle
from acore import op_inner_cond, quantile_bins, dev, qtile
from equity_benchmark import load, DRAW_SEEDS
from svarcore import var_fit, var_regenerate

y, p, Bc, U, W = load(); u = U - U.mean(0); n = len(u)
z, L = whiten(u)
a0 = np.deg2rad(het_angle(u, W, 2)[0]); E0 = z @ rot(a0)
b3 = quantile_bins(W, 3); one = np.zeros(n, int)
vr, vc = op_inner_cond(E0, one, E0, one, 1), op_inner_cond(E0, b3, E0, b3, 3)
out = dict(angle_deg=float(np.rad2deg(a0)), norm=float(np.sqrt(vr)), norm_cond=float(np.sqrt(vc)))
for block in (10, int(np.ceil(np.sqrt(n)))):
    Dr, Dc = [], []
    for s in DRAW_SEEDS:
        idx = boot_index(n, block, np.random.default_rng(s))
        ys = var_regenerate(Bc, y[:p], u[idx]); _, ub = var_fit(ys, p); ub = ub - ub.mean(0)
        zb, _ = whiten(ub); Wb = W[idx]
        ab = np.deg2rad(het_angle(ub, Wb, 2)[0])
        ab = a0 + ((ab - a0 + np.pi / 4) % (np.pi / 2) - np.pi / 4)      # nearest equivalent angle
        Eb = zb @ rot(ab)
        Dr.append(dev(Eb, one, E0, one, 1, vr)); Dc.append(dev(Eb, quantile_bins(Wb, 3), E0, b3, 3, vc))
    qr, qc = qtile(np.array(Dr), .95), qtile(np.array(Dc), .95)
    out[f'block{block}'] = dict(q=qr, exposure_lower=float(np.sqrt(max(np.sqrt(vr) - qr, 0))),
                                q_cond=qc, residual_exposure_lower=float(np.sqrt(max(np.sqrt(vc) - qc, 0))))
json.dump(out, open(os.path.join(HERE, 'results', 'equity_hetexposure.json'), 'w'), indent=1)
print(json.dumps(out))
