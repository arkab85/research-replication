"""Build the two panels of flows/PROTOCOL.md from raw public files.

    python flows/get_cot.py ; python flows/get_tic.py ; python flows/build_flows_data.py

Set 1: data/weekly_panel.csv   f_<market> = 100 x change in net non-commercial position / lagged open interest
                               r_<market> = change in the cash price between consecutive report dates
Set 2: data/monthly_panel.csv  f_<currency> = scaled net portfolio flow toward the country, r_<currency> = month-end log change
"""
import os, sys, re, io, json, shutil, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
import build_data as bd
RAW = os.path.join(HERE, 'data_raw'); OUT = os.path.join(HERE, 'data'); H10 = os.path.join(RAW, 'h10')
for p in (OUT, H10): os.makedirs(p, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'}
W0, W1 = '1999-06-22', '2026-09-15'
FX = {'CAD': ('090741', 'ca'), 'EUR': ('099741', 'eu'), 'JPY': ('097741', 'ja'), 'CHF': ('092741', 'sz'), 'GBP': ('096742', 'uk'), 'MXN': ('095741', 'mx')}
EQ = {'SPX': ('13874A', '^GSPC'), 'NDX': ('209742', '^NDX')}
UST = {'UST2': ('042601', 'RIFLGFCY02_N.B'), 'UST5': ('044601', 'RIFLGFCY05_N.B'), 'UST10': ('043602', 'RIFLGFCY10_N.B'), 'USTB': ('020601', 'RIFLGFCY20_N.B')}
GROUPS = {'currencies': list(FX), 'equity': list(EQ), 'treasuries': list(UST)}
USD_PER_UNIT = {'EUR', 'GBP', 'AUD', 'NZD'}
TIC = {'AUD': ('Australia', 'al'), 'CAD': ('Canada', 'ca'), 'DKK': ('Denmark', 'dn'), 'JPY': ('Japan', 'ja'), 'NZD': ('New Zealand', 'nz'),
       'NOK': ('Norway', 'no'), 'SEK': ('Sweden', 'sd'), 'CHF': ('Switzerland', 'sz'), 'GBP': ('United Kingdom', 'uk')}


def get(url, path, fresh=False):
    if fresh or not os.path.exists(path):
        r = requests.get(url, headers=UA, timeout=300); r.raise_for_status(); open(path, 'wb').write(r.content)
    return path


def download():
    codes = sorted({c for _, c in FX.values()} | {c for _, c in TIC.values()})
    for c in codes:
        for dec in ('dat89', 'dat96', 'dat00'):
            if c == 'eu' and dec == 'dat89': continue
            if c == 'mx' and dec == 'dat89': continue
            name = f'{dec}_{c}.' + ('htm' if dec == 'dat00' else 'txt'); dst = os.path.join(H10, name)
            if dec != 'dat00':                                        # closed files: reuse a copy already in the package
                for src in (os.path.join(ROOT, 'data', 'raw', 'h10', name), os.path.join(ROOT, 'holdout', 'data_raw', 'h10', name)):
                    if os.path.exists(src) and not os.path.exists(dst): shutil.copy(src, dst)
            get('https://www.federalreserve.gov/releases/h10/hist/' + name, dst)
    for k, (_, tick) in EQ.items():
        get(f'https://query1.finance.yahoo.com/v8/finance/chart/{tick}?period1=0&period2=1790000000&interval=1d', os.path.join(RAW, f'yahoo_{k}.json'))
    h15 = os.path.join(RAW, 'H15_treasury_cmt.csv')
    if not os.path.exists(h15): shutil.copy(os.path.join(ROOT, 'extension', 'data_raw', 'H15_treasury_cmt.csv'), h15)


def h10(code):
    old = bd.H10; bd.H10 = H10
    try: return bd.h10_daily(code)
    finally: bd.H10 = old


def cash():
    P = {}
    for k, (_, c) in FX.items():
        s = h10(c); P[k] = s if k in USD_PER_UNIT else 1.0 / s                                   # US$ per unit of foreign currency
    for k in EQ:
        j = json.load(open(os.path.join(RAW, f'yahoo_{k}.json')))['chart']['result'][0]
        s = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna(); P[k] = s[~s.index.duplicated()]
    H = pd.read_csv(os.path.join(RAW, 'H15_treasury_cmt.csv'), skiprows=5, index_col=0, parse_dates=True, na_values=['ND', ''])
    for k, (_, col) in UST.items(): P[k] = pd.to_numeric(H[col], errors='coerce').dropna()
    return P


def weekly():
    D = pd.read_csv(os.path.join(RAW, 'cot_legacy.csv.gz'), dtype={'code': str}, parse_dates=['date']); D = D[(D.date >= W0) & (D.date <= W1)]
    M = {**FX, **EQ, **UST}; cal = pd.DatetimeIndex(sorted(D[D.code.isin([v[0] for v in M.values()])].date.unique())); P = cash(); W = pd.DataFrame(index=cal); info = []
    for k, (code, _) in M.items():
        d = D[D.code == code].set_index('date'); N = (d.ncl - d.ncs).reindex(cal); OI = d.oi.reindex(cal); miss = int(N.isna().sum())
        N = N.interpolate(limit_area='inside'); OI = OI.interpolate(limit_area='inside'); W['f_' + k] = 100 * N.diff() / OI.shift(1)
        px = P[k].reindex(P[k].index.union(cal)).ffill().reindex(cal)
        W['r_' + k] = px.diff() if k in UST else 100 * np.log(px).diff()
        if k in FX:       # H.10 is a noon rate and positions are as of the close: the first noon rate AFTER the report date, for the timing check of PROTOCOL 9.6
            s = P[k]; j = s.index.searchsorted(cal, side='right'); nxt = pd.Series(np.where(j < len(s), s.values[np.minimum(j, len(s) - 1)], np.nan), index=cal); W['rw_' + k] = 100 * np.log(nxt).diff()
        info.append(dict(market=k, code=code, missing_weeks=miss, last_cash=P[k].index.max().date(), zero_flow_share=float((W['f_' + k] == 0).mean())))
    last = min(P[k].index.max() for k in M); W = W.iloc[1:]; W = W[W.index <= last]      # a report date enters only if every cash price for it is published
    W.index.name = 'date'; W.to_csv(os.path.join(OUT, 'weekly_panel.csv'))
    pd.DataFrame(info).to_csv(os.path.join(OUT, 'weekly_panel_info.csv'), index=False)
    print('weekly', W.shape, W.index[0].date(), W.index[-1].date(), 'NaN cells:', int(W.isna().sum().sum())); print(pd.DataFrame(info).to_string(index=False)); return W


def monthly():
    L = open(os.path.join(RAW, 's1_globl.txt'), encoding='latin-1').read().splitlines(); rows = []
    for l in L:
        c = [x.strip().strip('"') for x in l.split('\t')]
        if len(c) >= 15 and re.fullmatch(r'\d{4}-\d{2}', c[2] or ''):
            rows.append([c[0], c[2]] + [pd.to_numeric(x.replace(',', ''), errors='coerce') for x in c[3:15]])
    T = pd.DataFrame(rows, columns=['country', 'month'] + [str(i) for i in range(1, 13)]); Mo = pd.DataFrame(index=sorted(T.month.unique())); info = []
    for k, (name, c) in TIC.items():
        d = T[T.country == name].drop_duplicates('month').set_index('month').sort_index().reindex(Mo.index); v = d[[str(i) for i in range(1, 13)]].astype(float)
        net = (v['11'] + v['12'] - v['5'] - v['6']) - (v['1'] + v['2'] + v['3'] + v['4'] - v['7'] - v['8'] - v['9'] - v['10'])
        gross = v.sum(axis=1, min_count=12); Mo['f_' + k] = 100 * net / gross.rolling(12).mean().shift(1)      # no divisor until twelve reported months exist
        s = h10(c); s = s if k in USD_PER_UNIT else 1.0 / s; me = s.groupby(s.index.to_period('M')).last(); me.index = me.index.astype(str)
        Mo['r_' + k] = (100 * np.log(me).diff()).reindex(Mo.index); info.append(dict(currency=k, country=name, tic_months=int(v['1'].notna().sum()), zero_flow_share=float((net == 0).mean())))
    # New Zealand is reported separately in TIC only from 2001: the long panel is the other eight countries; NZD columns are empty before 2001
    Mo.index.name = 'month'; eight = Mo.drop(columns=['f_NZD', 'r_NZD']).dropna(); Mo = Mo.loc[eight.index[0]:eight.index[-1]]; Mo.to_csv(os.path.join(OUT, 'monthly_panel.csv'))
    print('monthly', Mo.shape, Mo.index[0], Mo.index[-1], 'NaN cells outside NZD:', int(Mo.drop(columns=['f_NZD', 'r_NZD']).isna().sum().sum()), '; nine-country months:', len(Mo.dropna())); print(pd.DataFrame(info).to_string(index=False)); return Mo


if __name__ == '__main__':
    download(); weekly(); monthly()
