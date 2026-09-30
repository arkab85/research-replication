"""Size audit of HOLDOUT_PROTOCOL.md, Section 4: actual hold-out cause series, simulated outcomes independent of them.

    python flows/holdout_size.py <A|B> <cause> <ret|vol> R part nparts      |     python flows/holdout_size.py collect
"""
import sys, os, glob, time
from holdout_lib import *


def run(name, cause, design, R, part, nparts):
    cfg = HOLDOUTS[name]; mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); A = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0)
    F = A[[f'{cause}_{m}' for m in mk]].dropna(); T = len(F); rows = []; t0 = time.time()
    if design == 'vol': phi, sd = rv_calibration(name)
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(7000 + rep); Y = sim_outcomes(T, len(mk), rng) if design == 'ret' else vol_outcomes(T, len(mk), rng, phi=phi, sd=sd)
        P = pd.DataFrame({**{'f_' + m: F[f'{cause}_{m}'].values for m in mk}, **{'r_' + m: Y[:, i] for i, m in enumerate(mk)}}, index=F.index)
        U = make_units(P, mk, hs, cfg['gap'], 'FR', innovate=CAUSES[cause][1]); a = rotate(U, hs, 2, step=3); b = rotate(U, hs, 2, step=3, ip=True)
        for st, w in S.items():
            o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*b, w); rows.append(dict(rep=rep, holdout=name, cause=cause, design=design, statistic=st, value=o, mean_rot=mu, p_plain=p1, p_ip=p2, rotations=a[2]))
        print(f'{name} {cause} {design} rep {rep} {time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'hosize_{name}_{cause}_{design}_part{part}.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'hosize_*_part*.csv'))]); g = D.groupby(['holdout', 'design', 'cause', 'statistic'], sort=False)
        out = g[['p_plain', 'p_ip']].agg(lambda s: round(100 * np.mean(s <= 0.05), 1)); out['reps'] = g.rep.nunique(); out['mean_index'] = (1e3 * g.value.mean()).round(3); out = out.reset_index()
        out.to_csv(os.path.join(RES, 'holdout_size.csv'), index=False); pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300); print(out.to_string(index=False))
    else: run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
