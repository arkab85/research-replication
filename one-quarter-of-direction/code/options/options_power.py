"""Power of OPTIONS_PROTOCOL.md, Section 6 (as amended): cw signals, simulated outcomes with the Step 0b channel at a
10 percent share, lags 1 and 4. python options/options_power.py R part nparts | collect"""
import sys, glob, time
from options_lib import *


def run(R, part, nparts):
    Q, mk = panel_for('cw'); S = stats_opt(); F = Q[['f_' + m for m in mk]]; T = len(F); n, tr = split_n(T - 2, GAP); ev = slice(2 + tr + GAP, T); chan = {}; rows = []; t0 = time.time()
    for m in mk:
        f = F['f_' + m].values; Z = np.column_stack([np.ones(T - 2), f[1:-1], f[:-2]]); e = f[2:] - Z @ np.linalg.lstsq(Z[:tr], f[2:][:tr], rcond=None)[0]; e = e / e[tr + GAP:].std(); g = np.concatenate([[0, 0], e + 0.6 * (e ** 2 - 1)])
        for k in (1, 4):
            c = np.zeros(T)
            for j in range(1, k + 1): c[j:] += g[:-j]
            chan[m, k] = (c - c[ev].mean()) / c[ev].std()
    for rep in range(part, R, nparts):
        rng = np.random.RandomState(9000 + rep); base = sim_outcomes(T, len(mk), rng)
        for share, spread in ((0.10, 1), (0.10, 4)):
            Y = base.copy(); bcoef = np.sqrt(share / (1 - share))
            for i, m in enumerate(mk): Y[:, i] += bcoef * chan[m, spread]
            se = float(np.mean([np.var(bcoef * chan[m, spread][ev]) / np.var(Y[ev, i]) for i, m in enumerate(mk)]))
            P = pd.DataFrame({**{'f_' + m: F['f_' + m].values for m in mk}, **{'r_' + m: Y[:, i] for i, m in enumerate(mk)}}, index=F.index)
            U = make_units(P, mk, HS, GAP, 'FR', innovate=False); a = rotate(U, HS, 2, step=5); bb = rotate(U, HS, 2, step=5, ip=True)
            for st, w in S.items():
                o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*bb, w); rows.append(dict(rep=rep, share=share, spread=spread, share_eval=se, statistic=st, value=o, p_plain=p1, p_ip=p2))
        print(f'power rep {rep} {time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(os.path.join(RES, f'optpower_part{part}.csv'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'optpower_part*.csv'))]); g = D.groupby(['share', 'spread', 'statistic'], sort=False)
        out = g[['p_plain', 'p_ip']].agg(lambda s: round(100 * np.mean(s <= 0.05), 1)); out['reps'] = g.rep.nunique(); out['share_eval'] = g.share_eval.mean().round(3); out = out.reset_index()
        out.to_csv(os.path.join(RES, 'options_power.csv'), index=False); pd.set_option('display.width', 200); print(out.to_string(index=False))
    else: run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
