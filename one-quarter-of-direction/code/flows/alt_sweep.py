"""Exploratory sweep over alternative cause definitions and outcomes (PROTOCOL.md, Section 13). Same twelve markets, calendar,
split rule, horizons, and tests as the primary analysis; flows to outcome direction only; plain and impact-preserving rotation;
pooled over the twelve markets and over the three groups. Reported in full; not a confirmatory test.

    python flows/alt_sweep.py <cause> <outcome>          cause in CAUSES, outcome in ('r', 'rv')
    python flows/alt_sweep.py collect
"""
import sys, os, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
CAUSES = {'nc_net_chg': ('net non-commercial change / OI (pre-specified)', True), 'lvl_net': ('net non-commercial position / OI (level)', False), 'z52_net': ('52-week z-score of the net position (positioning index)', False),
          'nc_long_chg': ('non-commercial long change / OI', True), 'nc_short_chg': ('non-commercial short change / OI', True), 'comm_net_chg': ('commercial net change / OI', True), 'oi_growth': ('open-interest growth', True),
          'tff_lev_chg': ('leveraged funds net change / OI (TFF, 2006-)', True), 'tff_am_chg': ('asset managers net change / OI (TFF, 2006-)', True), 'tff_dealer_chg': ('dealers net change / OI (TFF, 2006-)', True)}
OUTC = {'r': 'weekly return', 'rv': 'log realized variance'}


def run(cause, outcome):
    cfg = SETS['set1']; mk = cfg['markets']; hs = cfg['hs']; S = stats_for(hs); A = pd.read_csv(os.path.join(DATA, 'weekly_alt_panel.csv'), index_col=0)
    P = pd.DataFrame({**{'f_' + m: A[f'{cause}_{m}'] for m in mk}, **{'r_' + m: A[f'{outcome}_{m}'] for m in mk}}).dropna(); assert np.isfinite(P.values).all()
    U = make_units(P, mk, hs, cfg['gap'], 'FR', innovate=CAUSES[cause][1]); a = rotate(U, hs, 2); b = rotate(U, hs, 2, ip=True); rows = []
    for g, cols in [('all twelve', list(range(len(mk))))] + [(g_, [mk.index(m) for m in ms]) for g_, ms in cfg['groups'].items()]:
        for st, w in S.items():
            o, mu, p1 = rot_p(*a, w, cols); _, mu2, p2 = rot_p(*b, w, cols)
            rows.append(dict(cause=cause, outcome=outcome, group=g, markets=len(cols), statistic=st, value=o, mean_rot=mu, p_plain=p1, mean_rot_ip=mu2, p_ip=p2, positive=(int((a[0][int(st[2:-1])][cols] > 0).sum()) if st.startswith('P(') else np.nan), rotations=a[2], T=len(P), n=len(U[hs[0]][0]['Xte']), start=P.index[0]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f'alt_{cause}_{outcome}.csv'), index=False, float_format='%.6g'); print(cause, outcome, 'done', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'alt_*_r.csv')) + glob.glob(os.path.join(RES, 'alt_*_rv.csv'))]); D.to_csv(os.path.join(RES, 'alt_sweep.csv'), index=False)
        pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); q = D[(D.group == 'all twelve')]
        print(q.pivot_table(index=['outcome', 'cause'], columns='statistic', values='p_ip').round(3).to_string()); print(); print(q.pivot_table(index=['outcome', 'cause'], columns='statistic', values='p_plain').round(3).to_string())
    else: run(sys.argv[1], sys.argv[2])
