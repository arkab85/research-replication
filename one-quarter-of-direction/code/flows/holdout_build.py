"""Build the two hold-out panels of HOLDOUT_PROTOCOL.md (Section 2).

    python flows/holdout_build.py

A: data/holdout_A_panel.csv   four new markets, 2010-01-05 to 2026-09-15; per market m: oi_growth_m, nc_short_chg_m, r_m, rv_m
B: data/holdout_B_panel.csv   the twelve-market alternative panel truncated at 2019-12-31 (same columns as weekly_alt_panel.csv)
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code')); sys.path.insert(0, HERE)
import build_data as bd
import build_flows_data as B
RAW = B.RAW; OUT = B.OUT; DII = os.path.join(os.path.dirname(ROOT), 'DII_full_package_MS_and_JFQA', 'oilvix', 'raw')
W0, W1 = '2010-01-05', '2026-09-15'
MARKETS = {'WTI': '067651', 'VIX': '1170E1', 'AUD': '232741', 'NZD': '112741'}


def cash_holdout():
    P = {}
    w = pd.read_csv(os.path.join(DII, 'wti.csv'), parse_dates=['Date']).set_index('Date')['Price']; P['WTI'] = pd.to_numeric(w, errors='coerce').dropna()
    v = pd.read_csv(os.path.join(DII, 'vix.csv'), parse_dates=['DATE']).set_index('DATE')['CLOSE']; P['VIX'] = pd.to_numeric(v, errors='coerce').dropna()
    P['AUD'] = B.h10('al'); P['NZD'] = B.h10('nz')                                 # H.10 quotes A$ and NZ$ as US$ per unit
    for k in P: P[k] = P[k][P[k] > 0]; P[k] = P[k][~P[k].index.duplicated()].sort_index()
    return P


def build_A():
    D = pd.read_csv(os.path.join(RAW, 'cot_legacy.csv.gz'), dtype={'code': str}, parse_dates=['date'])
    ref = pd.DatetimeIndex(sorted(D[(D.code == '090741') & (D.date >= W0) & (D.date <= W1)].date.unique()))     # the report calendar
    P = cash_holdout(); A = pd.DataFrame(index=ref); info = []
    for k, code in MARKETS.items():
        d = D[D.code == code].set_index('date')[['oi', 'ncl', 'ncs', 'cl', 'cs']].astype(float); miss = int((~ref.isin(d.index)).sum())
        d = d.reindex(ref).interpolate(limit_area='inside'); oi1 = d.oi.shift(1)
        A['oi_growth_' + k] = 100 * np.log(d.oi).diff(); A['nc_short_chg_' + k] = 100 * d.ncs.diff() / oi1; A['nc_net_chg_' + k] = 100 * (d.ncl - d.ncs).diff() / oi1
        s = P[k]; px = s.reindex(s.index.union(ref)).ffill().reindex(ref); A['r_' + k] = 100 * np.log(px).diff()
        dd = (100 * np.log(s).diff()).dropna(); wk = pd.Series(np.searchsorted(ref.values, dd.index.values, side='left'), index=dd.index)
        rv = (dd ** 2).groupby(wk).sum(); rv = rv[(rv.index > 0) & (rv.index < len(ref))]; A['rv_' + k] = np.log(rv.reindex(range(len(ref))).values + 1e-8)
        info.append(dict(market=k, code=code, missing_weeks=miss, first_cash=s.index.min().date(), last_cash=s.index.max().date(), zero_oi_growth_share=float((A['oi_growth_' + k] == 0).mean())))
    A = A.iloc[1:]; A.index.name = 'date'; A.to_csv(os.path.join(OUT, 'holdout_A_panel.csv')); pd.DataFrame(info).to_csv(os.path.join(OUT, 'holdout_A_info.csv'), index=False)
    print('A', A.shape, A.index[0].date(), A.index[-1].date(), 'NaN cells:', int(A.isna().sum().sum())); print(pd.DataFrame(info).to_string(index=False)); return A


def build_B():
    W = pd.read_csv(os.path.join(OUT, 'weekly_alt_panel.csv'), index_col=0, parse_dates=True); Bp = W[W.index <= '2019-12-31']
    Bp.to_csv(os.path.join(OUT, 'holdout_B_panel.csv')); print('B', Bp.shape, Bp.index[0].date(), Bp.index[-1].date()); return Bp


if __name__ == '__main__':
    build_A(); build_B()
