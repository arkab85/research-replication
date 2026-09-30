"""Alternative cause definitions for the exploratory sweep (PROTOCOL.md, Section 13). Same twelve markets and weekly calendar
as data/weekly_panel.csv. Writes data/weekly_alt_panel.csv with, per market m:
  nc_net_chg   net non-commercial change / OI (the pre-specified flow)          lvl_net   net non-commercial position / OI (level)
  z52_net      52-week z-score of lvl_net (positioning index)                     nc_long_chg, nc_short_chg   long and short changes / OI
  comm_net_chg commercial (hedger/dealer) net change / OI                         oi_growth   log change in open interest
  tff_lev_chg, tff_am_chg, tff_dealer_chg   Traders in Financial Futures net changes / OI (from 2006-06)
  r_m  weekly return (as before)     rv_m  log realized variance of the week (as in positive_control.py)
    python flows/build_alt_flows.py"""
import os, io, sys, zipfile, requests, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_flows_data as B
RAW = B.RAW; OUT = B.OUT; UA = B.UA; W = pd.read_csv(os.path.join(OUT, 'weekly_panel.csv'), index_col=0, parse_dates=True); cal = W.index
M = {**B.FX, **B.EQ, **B.UST}; codes = {k: v[0] for k, v in M.items()}
# --- legacy report: gross positions ---------------------------------------------------------------------------------------
D = pd.read_csv(os.path.join(RAW, 'cot_legacy.csv.gz'), dtype={'code': str}, parse_dates=['date']); D = D[D.code.isin(codes.values())]
# --- Traders in Financial Futures -------------------------------------------------------------------------------------------
TF = os.path.join(RAW, 'tff'); os.makedirs(TF, exist_ok=True); files = ['fin_fut_txt_2006_2016.zip'] + [f'fut_fin_txt_{y}.zip' for y in range(2017, 2027)]; parts = []
for f in files:
    p = os.path.join(TF, f)
    if not os.path.exists(p):
        r = requests.get('https://www.cftc.gov/files/dea/history/' + f, headers=UA, timeout=300)
        if r.status_code != 200: print('missing', f); continue
        open(p, 'wb').write(r.content)
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            t = pd.read_csv(io.BytesIO(z.read(n)), low_memory=False, dtype=str); t.columns = [c.strip() for c in t.columns]; parts.append(t)
T = pd.concat(parts); cc = [c for c in T.columns if 'Contract_Market_Code' in c][0]; dc = [c for c in T.columns if 'Report_Date' in c and 'YYYY' in c][0]
T['code'] = T[cc].str.strip(); T['date'] = pd.to_datetime(T[dc].str.strip(), errors='coerce'); T = T[T.code.isin(codes.values())].dropna(subset=['date'])
def num(s): return pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')
col = lambda pat: [c for c in T.columns if c.startswith(pat)][0]
T['oi'] = num(T[col('Open_Interest_All')]); T['lev'] = num(T[col('Lev_Money_Positions_Long_All')]) - num(T[col('Lev_Money_Positions_Short_All')])
T['am'] = num(T[col('Asset_Mgr_Positions_Long_All')]) - num(T[col('Asset_Mgr_Positions_Short_All')]); T['dealer'] = num(T[col('Dealer_Positions_Long_All')]) - num(T[col('Dealer_Positions_Short_All')])
T = T.drop_duplicates(['code', 'date']).sort_values(['code', 'date']); print('TFF rows', len(T), T.date.min().date(), T.date.max().date())
# --- realized variance -------------------------------------------------------------------------------------------------------
P = B.cash(); A = pd.DataFrame(index=cal)
for k, code in codes.items():
    d = D[D.code == code].set_index('date')[['oi', 'ncl', 'ncs', 'cl', 'cs']].astype(float).reindex(cal).interpolate(limit_area='inside'); oi1 = d.oi.shift(1)
    A['nc_net_chg_' + k] = 100 * (d.ncl - d.ncs).diff() / oi1; A['lvl_net_' + k] = 100 * (d.ncl - d.ncs) / d.oi
    A['z52_net_' + k] = (A['lvl_net_' + k] - A['lvl_net_' + k].rolling(52).mean()) / A['lvl_net_' + k].rolling(52).std()
    A['nc_long_chg_' + k] = 100 * d.ncl.diff() / oi1; A['nc_short_chg_' + k] = 100 * d.ncs.diff() / oi1; A['comm_net_chg_' + k] = 100 * (d.cl - d.cs).diff() / oi1; A['oi_growth_' + k] = 100 * np.log(d.oi).diff()
    t = T[T.code == code].set_index('date')[['oi', 'lev', 'am', 'dealer']].astype(float).reindex(cal).interpolate(limit_area='inside'); toi = t.oi.shift(1)
    for c in ('lev', 'am', 'dealer'): A[f'tff_{c}_chg_' + k] = 100 * t[c].diff() / toi
    A['r_' + k] = W['r_' + k]; s = P[k]; dd = (s.diff() if k in B.UST else 100 * np.log(s).diff()).dropna(); wk = pd.Series(np.searchsorted(cal.values, dd.index.values, side='left'), index=dd.index)
    rv = (dd ** 2).groupby(wk).sum(); rv = rv[(rv.index > 0) & (rv.index < len(cal))]; A['rv_' + k] = np.log(rv.reindex(range(len(cal))).values + 1e-8)
A.index.name = 'date'; A.to_csv(os.path.join(OUT, 'weekly_alt_panel.csv')); print(A.shape, 'first full TFF week:', A[[c for c in A.columns if c.startswith('tff_lev')]].dropna().index[0].date())
