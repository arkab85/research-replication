"""Real-outcome tests of OPTIONS_PROTOCOL.md: both labellings, every admissible shift, plain and impact-preserving rotation.
python options/options_apply.py <cw|skew> <FR|RF> | collect"""
import sys, glob
from options_lib import *


def run(cause, d):
    Q, mk = panel_for(cause); hs, gap, S = design_cfg(cause); rows = []
    U = make_units(Q, mk, hs, gap, d, innovate=CAUSES[cause][1]); a = rotate(U, hs, 2); b = rotate(U, hs, 2, ip=True); c_ = rotate(U, hs, 2, ip='sieve') if cause in ('retvol', 'spy', 'spyvol') else None
    for st, w in S.items():
        o, mu, p1 = rot_p(*a, w); _, mu2, p2 = rot_p(*b, w); p3 = rot_p(*c_, w)[2] if c_ is not None else np.nan
        rows.append(dict(cause=cause, direction=d, group='all', markets=len(mk), statistic=st, value=o, mean_rot=mu, p_plain=p1, mean_rot_ip=mu2, p_ip=p2, p_ip_sieve=p3,
                         positive=(int((a[0][int(st[2:-1])] > 0).sum()) if st.startswith('P(') else np.nan), rotations=a[2], T=len(Q), n=len(U[hs[0]][0]['Xte']), eval_start=Q.index[len(Q) - len(U[hs[0]][0]['Xte'])]))
    for i, m in enumerate(mk):
        for h in hs: rows.append(dict(cause=cause, direction=d, group=m, markets=1, statistic=f'P({h})', value=a[0][h][i], mean_rot=a[1][h][:, i].mean(), p_plain=(1 + np.sum(a[1][h][:, i] >= a[0][h][i])) / (a[2] + 1), mean_rot_ip=b[1][h][:, i].mean(), p_ip=(1 + np.sum(b[1][h][:, i] >= b[0][h][i])) / (b[2] + 1), positive=np.nan, rotations=a[2], T=len(Q), n=len(U[hs[0]][0]['Xte']), eval_start=Q.index[len(Q) - len(U[hs[0]][0]['Xte'])]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'opt_{cause}_{d}.csv'), index=False, float_format='%.6g'); print(cause, d, 'done', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, 'opt_*_*.csv')))]); D.to_csv(os.path.join(RES, 'options_apply.csv'), index=False)
        pd.set_option('display.width', 250); q = D[D.group == 'all']
        print(q.pivot_table(index=['cause', 'direction'], columns='statistic', values='p_ip').round(3).to_string()); print(); print(q.pivot_table(index=['cause', 'direction'], columns='statistic', values='p_plain').round(3).to_string())
    else: run(sys.argv[1], sys.argv[2])
