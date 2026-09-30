"""The real-outcome tests of HOLDOUT_PROTOCOL.md, Sections 3 and 5: both labellings, every admissible shift, plain and
impact-preserving rotation, pooled over the hold-out's markets (and over the groups of Set 1 in hold-out B).

    python flows/holdout_apply.py <A|B> <cause> <r|rv>        |      python flows/holdout_apply.py collect
"""
import sys, os, glob
from holdout_lib import *


def run(name, cause, outcome):
    P, cfg = panel(name, cause, outcome); mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); rows = []
    for d in ('FR', 'RF'):
        U = make_units(P, mk, hs, cfg['gap'], d, innovate=CAUSES[cause][1]); a = rotate(U, hs, 2); b = rotate(U, hs, 2, ip=True); c_ = rotate(U, hs, 2, ip='sieve') if cause == 'r' else None
        for g, cols in [('all', list(range(len(mk))))] + [(g_, [mk.index(m) for m in ms]) for g_, ms in cfg['groups'].items()]:
            for st, w in S.items():
                o, mu, p1 = rot_p(*a, w, cols); _, mu2, p2 = rot_p(*b, w, cols); p3 = rot_p(*c_, w, cols)[2] if c_ is not None else np.nan
                rows.append(dict(holdout=name, cause=cause, outcome=outcome, direction=d, group=g, markets=len(cols), statistic=st, value=o, mean_rot=mu, p_plain=p1, mean_rot_ip=mu2, p_ip=p2, p_ip_sieve=p3,
                                 positive=(int((a[0][int(st[2:-1])][cols] > 0).sum()) if st.startswith('P(') else np.nan), rotations=a[2], T=len(P), n=len(U[hs[0]][0]['Xte']), start=P.index[0], eval_start=P.index[len(P) - len(U[hs[0]][0]['Xte'])]))
        for i, m in enumerate(mk):                       # per-market indices at each horizon, for the record
            for h in hs: rows.append(dict(holdout=name, cause=cause, outcome=outcome, direction=d, group=m, markets=1, statistic=f'P({h})', value=a[0][h][i], mean_rot=a[1][h][:, i].mean(), p_plain=(1 + np.sum(a[1][h][:, i] >= a[0][h][i])) / (a[2] + 1), mean_rot_ip=b[1][h][:, i].mean(), p_ip=(1 + np.sum(b[1][h][:, i] >= b[0][h][i])) / (b[2] + 1), positive=np.nan, rotations=a[2], T=len(P), n=len(U[hs[0]][0]['Xte']), start=P.index[0], eval_start=P.index[len(P) - len(U[hs[0]][0]['Xte'])]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'ho_{name}_{cause}_{outcome}.csv'), index=False, float_format='%.6g'); print(name, cause, outcome, 'done', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, 'ho_*_*_*.csv')))]); D.to_csv(os.path.join(RES, 'holdout_apply.csv'), index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); q = D[(D.group == 'all')]
        print(q.pivot_table(index=['holdout', 'direction', 'outcome', 'cause'], columns='statistic', values='p_ip').round(3).to_string()); print()
        print(q.pivot_table(index=['holdout', 'direction', 'outcome', 'cause'], columns='statistic', values='p_plain').round(3).to_string())
    else: run(sys.argv[1], sys.argv[2], sys.argv[3])
