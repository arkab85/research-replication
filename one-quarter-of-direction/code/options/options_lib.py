"""Configuration for OPTIONS_PROTOCOL.md; everything else is flows/flows_lib.py, unchanged."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'flows')); sys.path.insert(0, os.path.join(ROOT, 'code'))
import numpy as np, pandas as pd
from flows_lib import *
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); RES = os.path.join(HERE, 'results'); os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
HS = (1, 2, 4, 8, 13, 26); GAP = 26; N = 100
CAUSES = {'cw': ('Cremers-Weinbaum call-put IV spread', False), 'skew': ('implied-volatility skew (XZZ)', False), 'big': ('big-firm industry portfolio return (Hou 2007)', False), 'ret': ('market-adjusted stock return (addendum 2)', False), 'retvol': ('market-adjusted stock return, outcome volume (addendum 3)', False), 'spy': ('SPY daily return, outcome log realized variance (addendum 3)', False), 'spyvol': ('SPY daily return, outcome log volume change (addendum 4)', False)}


def stats_opt():
    S = {f'P({h})': {h: 1.0} for h in HS}; S['A'] = {h: 1.0 / len(HS) for h in HS}; S['S'] = {1: 1 / 3, 2: 1 / 3, 4: 1 / 3}
    S['C'] = {**{h: 1.0 / (len(HS) - 1) for h in HS if h <= 13}, 26: -1.0}; return S


def design_cfg(cause):
    """(horizons, gap, statistics) for a design; every design but 'spy' uses the weekly configuration."""
    if cause in ('spy', 'spyvol'):
        hs = (1, 2, 5, 10, 21, 63); S = {f'P({h})': {h: 1.0} for h in hs}; S['A'] = {h: 1.0 / len(hs) for h in hs}; S['S'] = {1: 1 / 3, 2: 1 / 3, 5: 1 / 3}
        S['C'] = {**{h: 1.0 / (len(hs) - 1) for h in hs if h <= 21}, 63: -1.0}; return hs, 63, S
    return HS, GAP, stats_opt()


def load_panel():
    P = pd.read_csv(os.path.join(DATA, 'options_panel.csv'), index_col=0); mk = sorted({c.split('_', 1)[1] for c in P.columns if c.startswith('cw_')}, key=int); return P, mk


def panel_for(cause):
    if cause == 'big':
        P = pd.read_csv(os.path.join(DATA, 'leadlag_panel.csv'), index_col=0); mk = sorted({c.split('_', 1)[1] for c in P.columns if c.startswith('big_')}, key=int)
        Q = pd.DataFrame({**{'f_' + m: P[f'big_{m}'] for m in mk}, **{'r_' + m: P[f'small_{m}'] for m in mk}}); assert np.isfinite(Q.values).all(); return Q, mk
    if cause in ('spy', 'spyvol'):
        Q = pd.read_csv(os.path.join(DATA, 'spy_daily_panel.csv' if cause == 'spy' else 'spyvol_daily_panel.csv'), index_col=0); assert np.isfinite(Q.values).all(); return Q, ['SPY']
    P, mk = load_panel()
    if cause == 'retvol': Q = pd.DataFrame({**{'f_' + m: P[f'y_{m}'] for m in mk}, **{'r_' + m: P[f'dvol_{m}'] for m in mk}}); assert np.isfinite(Q.values).all(); return Q, mk
    if cause == 'ret': Q = pd.DataFrame({**{'f_' + m: P[f'y_{m}'] for m in mk}, **{'r_' + m: P[f'div_{m}'] for m in mk}})
    else: Q = pd.DataFrame({**{'f_' + m: P[f'{cause}_{m}'] for m in mk}, **{'r_' + m: P[f'y_{m}'] for m in mk}})
    assert np.isfinite(Q.values).all(); return Q, mk
