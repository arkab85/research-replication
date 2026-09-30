"""Size audit of OPTIONS_PROTOCOL.md, Section 5 (as amended): actual option signals of the 100 stocks, simulated outcomes
independent of them. python options/options_size.py <cw|skew> R part nparts | collect"""
import sys, glob, time
from options_lib import *


def run(cause, R, part, nparts):
    Q, mk = panel_for(cause); hs, gap, S = design_cfg(cause); F = Q[['f_' + m for m in mk]]; T = len(F); rows = []; t0 = time.time()
    if cause == 'spy':
        from alt_size import vol_outcomes
        rvs = Q['r_SPY'].values; phi = float(np.corrcoef(rvs[1:], rvs[:-1])[0, 1]); sd = float(np.std(np.diff(rvs))) * np.sqrt((1 - phi ** 2) / 2)
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(7000 + rep); Y = sim_outcomes(T, len(mk), rng) if cause != 'spy' else vol_outcomes(T, 1, rng, rho=0.0, phi=phi, sd=sd)
        P = pd.DataFrame({**{'f_' + m: F['f_' + m].values for m in mk}, **{'r_' + m: Y[:, i] for i, m in enumerate(mk)}}, index=F.index)
        U = make_units(P, mk, hs, gap, 'FR', innovate=CAUSES[cause][1]); a = rotate(U, hs, 2, step=5); b = rotate(U, hs, 2, step=5, ip=('sieve' if cause in ('retvol', 'spyvol') else True))
        for st, w in S.items():
            o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*b, w); rows.append(dict(rep=rep, cause=cause, statistic=st, value=o, mean_rot=mu, p_plain=p1, p_ip=p2, rotations=a[2]))
        print(f'{cause} size rep {rep} {time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'optsize_{cause}_part{part}.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'optsize_*_part*.csv'))]); g = D.groupby(['cause', 'statistic'], sort=False)
        out = g[['p_plain', 'p_ip']].agg(lambda s: round(100 * np.mean(s <= 0.05), 1)); out['reps'] = g.rep.nunique(); out['mean_index'] = (1e3 * g.value.mean()).round(3); out = out.reset_index()
        out.to_csv(os.path.join(RES, 'options_size.csv'), index=False); pd.set_option('display.width', 200); print(out.to_string(index=False))
    else: run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
