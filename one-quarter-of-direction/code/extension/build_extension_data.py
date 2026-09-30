"""Extension data: (A/B) monthly panel to 2024-01 with updated shocks; (C) daily panel 1988-2019. See PROTOCOL.md."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
import build_data as bd
RAW = os.path.join(HERE, 'data_raw'); os.makedirs(RAW, exist_ok=True); UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'}
SRC = {'shocks_fed_jk_t.csv': 'https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed_202401/main/shocks_fed_jk_t.csv',
       'H15_treasury_cmt.csv': 'https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=bf17364827e38702b42a58cf8eaa3f78&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn',
       'GSPC_yahoo.json': 'https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1=0&period2=1790000000&interval=1d'}


def download():
    import requests
    for name, url in SRC.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000: continue
        r = requests.get(url, headers=UA, timeout=180); r.raise_for_status(); open(path, 'wb').write(r.content); print(f'downloaded {name} {len(r.content):,} bytes')


def monthly():
    P = pd.read_csv(os.path.join(ROOT, 'data', 'analysis_panel.csv'), index_col=0); idx = pd.period_range('1971-01', '2024-01', freq='M').astype(str); P = P.reindex(idx); P.index.name = 'ym'
    jk = pd.read_csv(os.path.join(ROOT, 'data', 'local_inputs', 'jk_m.csv')); jk.index = bd.ym_index(jk['year'].values, jk['month'].values).astype(str)
    P['jk_ext'] = jk['MP_median']; P.loc['1990-01':'2024-01', 'jk_ext'] = P.loc['1990-01':'2024-01', 'jk_ext'].fillna(0.0)
    bs = pd.read_excel(os.path.join(ROOT, 'data', 'local_inputs', 'monetary-policy-surprises-data.xlsx'), sheet_name='Monthly (update 2023)'); bs.index = bd.ym_index(bs['Year'].values, bs['Month'].values).astype(str)
    P['bs_ext'] = pd.to_numeric(bs['MPS_ORTH'], errors='coerce')
    eur = bd.h10_daily('eu'); eur_m = eur.groupby(eur.index.to_period('M')).mean()
    for cur, code in bd.FLOAT9.items():
        s = 100 * np.log(bd.fx_per_usd_monthly(cur, code, eur_m)).diff(); s.index = s.index.astype(str); P['fx_' + cur] = s.reindex(idx)
    cur_md = os.path.join(ROOT, 'data', 'raw', 'fred_md_current.csv'); md_path = cur_md if os.path.exists(cur_md) else os.path.join(ROOT, 'data', 'raw', 'fred_md_2019-08.csv')
    md = pd.read_csv(md_path).iloc[1:]; md = md[md['sasdate'].notna()]; md.index = pd.PeriodIndex(pd.to_datetime(md['sasdate']), freq='M').astype(str)
    P['spread_ext'] = (md['BAA'].astype(float) - md['AAA'].astype(float)).diff().reindex(idx)
    dcol = 'TWEXMMTH' if 'TWEXMMTH' in md.columns else 'TWEXAFEGSMTHx'; P['dollar_ext'] = (100 * np.log(md[dcol].astype(float)).diff()).reindex(idx)
    P.loc[:'1973-03', ['spread_ext', 'dollar_ext']] = np.nan; P.loc['2020-01':, ['spread_ext', 'dollar_ext']] = np.nan
    P.to_csv(os.path.join(HERE, 'monthly_panel_ext.csv'), float_format='%.10g'); print('monthly:', os.path.basename(md_path), P[['jk_ext', 'bs_ext', 'spread_ext', 'dollar_ext', 'fx_AUD']].notna().sum().to_dict(), 'spread ends', P['spread_ext'].last_valid_index())


def daily():
    j = json.load(open(os.path.join(RAW, 'GSPC_yahoo.json')))['chart']['result'][0]
    sp = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s').normalize()).dropna(); sp = sp[~sp.index.duplicated()]
    h = pd.read_csv(os.path.join(RAW, 'H15_treasury_cmt.csv'), skiprows=5); c10 = [c for c in h.columns if c.endswith('Y10_N.B')][0]
    y10 = pd.to_numeric(h[c10], errors='coerce'); y10.index = pd.to_datetime(h.iloc[:, 0]); y10 = y10.dropna()
    v = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'VIX_History.csv')); vix = pd.Series(v['CLOSE'].values, index=pd.to_datetime(v['DATE'], format='%m/%d/%Y'))
    D = pd.DataFrame({'sp500': 100 * np.log(sp).diff(), 'dy10': y10.diff(), 'vix': 100 * np.log(vix).diff()})
    for cur, code in bd.FLOAT9.items():
        s = bd.h10_daily(code); s = 1.0 / s if cur in bd.USD_PER_FX else s; D = D.join((100 * np.log(s).diff()).rename('fx_' + cur), how='outer')
    D = D.loc['1988-01-01':'2019-12-31']
    jk = pd.read_csv(os.path.join(RAW, 'shocks_fed_jk_t.csv')); jk['d'] = pd.to_datetime(jk.iloc[:, 0]).dt.normalize(); jkd = jk.groupby('d')['MP_median'].sum()
    bs = pd.read_excel(os.path.join(ROOT, 'data', 'local_inputs', 'monetary-policy-surprises-data.xlsx'), sheet_name='FOMC (original)'); bs['d'] = pd.to_datetime(bs['Date']).dt.normalize()
    bsd = pd.to_numeric(bs['MPS_ORTH'], errors='coerce').groupby(bs['d']).sum(min_count=1).dropna()
    D['jk'] = jkd.reindex(D.index).fillna(0.0); D['bs'] = bsd.reindex(D.index).fillna(0.0); D.loc[:'1989-12-31', 'jk'] = np.nan; D.loc[:'1988-02-03', 'bs'] = np.nan
    lost = {'jk': int((~jkd.loc['1990':'2019'].index.isin(D.index)).sum()), 'bs': int((~bsd.loc['1988-02-04':'2019'].index.isin(D.index)).sum())}
    D.index.name = 'date'; D.to_csv(os.path.join(HERE, 'daily_panel.csv'), float_format='%.10g')
    print('daily:', D.shape, 'announcement days kept:', {k: int((D[k].fillna(0) != 0).sum()) for k in ('jk', 'bs')}, 'announcement dates not in calendar:', lost)
    print(D.notna().sum().to_string())


if __name__ == '__main__':
    if '--offline' not in sys.argv: download()
    monthly(); daily()
