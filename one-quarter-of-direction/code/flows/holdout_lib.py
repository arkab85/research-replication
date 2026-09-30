"""Shared configuration for the hold-out test of HOLDOUT_PROTOCOL.md. Everything else is flows_lib.py, unchanged."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flows_lib import *
from alt_size import vol_outcomes

HOLDOUTS = {
    'A': dict(file='holdout_A_panel.csv', hs=(1, 2, 4, 8, 13, 26), gap=26, markets=['WTI', 'VIX', 'AUD', 'NZD'], groups={}),
    'B': dict(file='holdout_B_panel.csv', hs=(1, 2, 4, 8, 13, 26), gap=26, markets=SETS['set1']['markets'], groups=SETS['set1']['groups']),
}
CAUSES = {'oi_growth': ('open-interest growth', True), 'nc_short_chg': ('non-commercial short change / OI', True), 'r': ('weekly return (positive control, HOLDOUT_PROTOCOL_2.md)', True)}
OUTC = {'r': 'weekly return', 'rv': 'log realized variance'}
HYP = [('H1', 'oi_growth', 'r'), ('H2', 'oi_growth', 'rv'), ('H3', 'nc_short_chg', 'rv'), ('H4', 'r', 'rv')]


def panel(name, cause, outcome):
    cfg = HOLDOUTS[name]; mk = cfg['markets']; A = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0)
    P = pd.DataFrame({**{'f_' + m: A[f'{cause}_{m}'] for m in mk}, **{'r_' + m: A[f'{outcome}_{m}'] for m in mk}}).dropna(); assert np.isfinite(P.values).all()
    return P, cfg


def rv_calibration(name):
    cfg = HOLDOUTS[name]; A = pd.read_csv(os.path.join(DATA, cfg['file']), index_col=0); RV = A[[f'rv_{m}' for m in cfg['markets']]].dropna()
    phi = float(np.median([np.corrcoef(RV[c].values[1:], RV[c].values[:-1])[0, 1] for c in RV])); sd = float(np.median([RV[c].diff().std() for c in RV])) * np.sqrt((1 - phi ** 2) / 2)
    return phi, sd
