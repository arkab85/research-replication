"""Size of the rotation test for the ALTERNATIVE cause definitions (PROTOCOL.md, Section 13): the actual alternative cause
series, and simulated outcomes independent of them. Two outcome designs, matched to the two real outcomes:
  'ret'  AR(1)-GARCH(1,1), Student-t(6), common factor, as in Step 0a (mimics weekly returns)
  'vol'  persistent Gaussian AR(1) with the autocorrelation and innovation variance of the real log realized variance,
         plus a common factor (mimics log RV, which is persistent and left-skewed but far from heavy-tailed)
No real outcome is used. Plain and impact-preserving rotation, pooled over the twelve markets.

    python flows/alt_size.py <cause> <outcome-design> R part nparts        |        python flows/alt_size.py collect
"""
import sys, os, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from alt_sweep import CAUSES


def vol_outcomes(T, N, rng, rho=0.3, phi=0.62, sd=1.0):
    """N persistent AR(1) series with a common factor; phi and sd are the median autocorrelation and innovation standard
    deviation of the twelve real log realized-variance series (flows/results/alt_size_calibration.csv)."""
    def one():
        e = rng.normal(0, sd, T + 200); x = np.empty(T + 200); v = 0.0
        for t in range(T + 200): v = phi * v + e[t]; x[t] = v
        return x[200:]
    g = one(); return np.column_stack([np.sqrt(rho) * g + np.sqrt(1 - rho) * one() for _ in range(N)])


def run(cause, design, R, part, nparts):
    cfg = SETS['set1']; mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); A = pd.read_csv(os.path.join(DATA, 'weekly_alt_panel.csv'), index_col=0)
    F = A[[f'{cause}_{m}' for m in mk]].dropna(); T = len(F); rows = []
    if design == 'vol':
        RV = A[[f'rv_{m}' for m in mk]].dropna(); phi = float(np.median([np.corrcoef(RV[c].values[1:], RV[c].values[:-1])[0, 1] for c in RV])); sd = float(np.median([RV[c].diff().std() for c in RV])) * np.sqrt((1 - phi ** 2) / 2)
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(7000 + rep); Y = sim_outcomes(T, len(mk), rng) if design == 'ret' else vol_outcomes(T, len(mk), rng, phi=phi, sd=sd)
        P = pd.DataFrame({**{'f_' + m: F[f'{cause}_{m}'].values for m in mk}, **{'r_' + m: Y[:, i] for i, m in enumerate(mk)}}, index=F.index)
        U = make_units(P, mk, hs, cfg['gap'], 'FR', innovate=CAUSES[cause][1]); a = rotate(U, hs, 2, step=3); b = rotate(U, hs, 2, step=3, ip=True)
        for st, w in S.items():
            o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*b, w); rows.append(dict(rep=rep, cause=cause, design=design, statistic=st, value=o, mean_rot=mu, p_plain=p1, p_ip=p2, rotations=a[2]))
        print('rep', rep, flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'altsize_{cause}_{design}_part{part}.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'altsize_*_part*.csv'))]); g = D.groupby(['design', 'cause', 'statistic'], sort=False)
        out = g[['p_plain', 'p_ip']].agg(lambda s: round(100 * np.mean(s <= 0.05), 1)); out['reps'] = g.rep.nunique(); out['mean_index'] = (1e3 * g.value.mean()).round(3); out = out.reset_index()
        out.to_csv(os.path.join(RES, 'alt_size.csv'), index=False); pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300); print(out.to_string(index=False))
    else: run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
