"""Power of the hold-out test (HOLDOUT_PROTOCOL.md, Section 6): hold-out A, actual open-interest-growth series, simulated
outcomes with the Step 0b channel of run_flows.py, g(e) = e + 0.6(e^2 - 1) of the cause's own innovation, spread over one
and four lags, scaled to a target share of evaluation-block outcome variance. Cause-to-outcome labelling only.

    python flows/holdout_power.py <A|B> <cause> R part nparts     |     python flows/holdout_power.py collect
"""
import sys, os, glob, time
from holdout_lib import *


def run(name, cause, R, part, nparts):
    cfg = HOLDOUTS[name]; mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); A = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0)
    F = A[[f'{cause}_{m}' for m in mk]].dropna(); T = len(F); n, tr = split_n(T - 2, cfg['gap']); ev = slice(2 + tr + cfg['gap'], T); chan = {}; rows = []; t0 = time.time()
    for m in mk:
        f = F[f'{cause}_{m}'].values; Z = np.column_stack([np.ones(T - 2), f[1:-1], f[:-2]]); e = f[2:] - Z @ np.linalg.lstsq(Z[:tr], f[2:][:tr], rcond=None)[0]
        e = e / e[tr + cfg['gap']:].std(); g = np.concatenate([[0, 0], e + 0.6 * (e ** 2 - 1)])
        for k in (1, 4):
            c = np.zeros(T)
            for j in range(1, k + 1): c[j:] += g[:-j]
            chan[m, k] = (c - c[ev].mean()) / c[ev].std()
    designs = [(s, k) for s in (0.05, 0.10) for k in (1, 4)]
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(9000 + rep); base = sim_outcomes(T, len(mk), rng)
        for share, spread in designs:
            Y = base.copy(); b = np.sqrt(share / (1 - share))
            for i, m in enumerate(mk): Y[:, i] += b * chan[m, spread]
            se = float(np.mean([np.var(b * chan[m, spread][ev]) / np.var(Y[ev, i]) for i, m in enumerate(mk)]))
            P = pd.DataFrame({**{'f_' + m: F[f'{cause}_{m}'].values for m in mk}, **{'r_' + m: Y[:, i] for i, m in enumerate(mk)}}, index=F.index)
            U = make_units(P, mk, hs, cfg['gap'], 'FR', innovate=CAUSES[cause][1]); a = rotate(U, hs, 2, step=3); bb = rotate(U, hs, 2, step=3, ip=True)
            for st, w in S.items():
                o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*bb, w); rows.append(dict(rep=rep, holdout=name, cause=cause, share=share, spread=spread, share_eval=se, statistic=st, value=o, p_plain=p1, p_ip=p2))
        print(f'{name} {cause} power rep {rep} {time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'hopower_{name}_{cause}_part{part}.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'hopower_*_part*.csv'))]); g = D.groupby(['holdout', 'cause', 'share', 'spread', 'statistic'], sort=False)
        out = g[['p_plain', 'p_ip']].agg(lambda s: round(100 * np.mean(s <= 0.05), 1)); out['reps'] = g.rep.nunique(); out['share_eval'] = g.share_eval.mean().round(3); out = out.reset_index()
        out.to_csv(os.path.join(RES, 'holdout_power.csv'), index=False); pd.set_option('display.width', 200); pd.set_option('display.max_rows', 300); print(out.to_string(index=False))
    else: run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
