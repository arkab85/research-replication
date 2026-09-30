"""Builds data/analysis_panel.csv from raw public sources.

Step 1 (download): fetches every public input into data/raw/. Skipped for files already present, so the
package runs offline once data/raw/ is populated (it ships populated).
Step 2 (construct): monthly panel 1971-01..2019-12 with four shocks, six domestic outcomes, nineteen currencies.

Units follow Section 4.2 of the paper: log returns and log changes in percent, yield/spread/EBP changes in
percentage points, currencies as foreign currency per dollar so that dollar appreciation is positive."""
import os, re, io, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw'); LOC = os.path.join(ROOT, 'data', 'local_inputs'); H10 = os.path.join(RAW, 'h10')
UA = {'User-Agent': 'Mozilla/5.0'}

FLOAT9 = {'AUD': 'al', 'CAD': 'ca', 'DKK': 'dn', 'JPY': 'ja', 'NZD': 'nz', 'NOK': 'no', 'SEK': 'sd', 'CHF': 'sz', 'GBP': 'uk'}
EURO10 = {'DEM': 'ge', 'FRF': 'fr', 'ITL': 'it', 'NLG': 'ne', 'BEF': 'be', 'ATS': 'au', 'FIM': 'fn', 'IEP': 'ir', 'ESP': 'sp', 'PTE': 'po'}
USD_PER_FX = {'AUD', 'NZD', 'GBP', 'IEP'}      # H.10 quotes these as US$ per unit; all others are units per US$
EURO_CONV = {'DEM': 1.95583, 'FRF': 6.55957, 'ITL': 1936.27, 'NLG': 2.20371, 'BEF': 40.3399, 'ATS': 13.7603,
             'FIM': 5.94573, 'IEP': 0.787564, 'ESP': 166.386, 'PTE': 200.482}

SOURCES = {
    'ie_data.xls': 'http://www.econ.yale.edu/~shiller/data/ie_data.xls',
    'VIX_History.csv': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
    'oilSupplyNewsShocks_2025M12.xlsx': 'https://raw.githubusercontent.com/dkaenzig/oilsupplynews/master/oilSupplyNewsShocks_2025M12.xlsx',
    'acosta_rrshocks.csv': 'https://raw.githubusercontent.com/miguel-acosta/RomerRomer2004/master/output/rrshocks.csv',
    'fred_md_2019-08.csv': 'https://raw.githubusercontent.com/bashtage/python-introduction/main/course/introduction/data/fred-md.csv',
}
for code in list(FLOAT9.values()) + list(EURO10.values()) + ['eu']:
    for dec in ('dat89', 'dat96', 'dat00'):
        if dec == 'dat00' and code in EURO10.values(): continue      # legacy currencies end with 1998
        if dec == 'dat89' and code == 'eu': continue                 # euro starts 1999 (in the dat96 file)
        ext = 'htm' if dec == 'dat00' else 'txt'                     # the Board publishes 2000+ as an HTML table only
        SOURCES[f'h10/{dec}_{code}.{ext}'] = f'https://www.federalreserve.gov/releases/h10/hist/{dec}_{code}.{ext}'


def download():
    import requests
    os.makedirs(H10, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000: continue
        r = requests.get(url, headers=UA, timeout=120); r.raise_for_status()
        if not name.endswith('.htm') and r.content[:200].lstrip(b'\xef\xbb\xbf \r\n').lower().startswith((b'<!doctype', b'<html')):
            raise RuntimeError(f'{url} returned an HTML page, not data')
        open(path, 'wb').write(r.content); print(f'downloaded {name:40s} {len(r.content):>9,d} bytes')


def ym_index(y, m): return pd.PeriodIndex(pd.to_datetime(dict(year=y, month=m, day=1)), freq='M')


def shiller():
    x = pd.read_excel(os.path.join(RAW, 'ie_data.xls'), sheet_name='Data', header=None)
    hdr = x.index[x[0].astype(str).str.strip() == 'Date'][0]
    d = x.iloc[hdr + 1:, [0, 1, 6]].copy(); d.columns = ['date', 'P', 'GS10']
    d = d[pd.to_numeric(d['date'], errors='coerce').notna()].astype(float)
    y = np.floor(d['date']).astype(int); m = np.round((d['date'] - y) * 100).astype(int)
    d.index = ym_index(y.values, m.values); return d[['P', 'GS10']]


def vix():
    v = pd.read_csv(os.path.join(RAW, 'VIX_History.csv')); v['DATE'] = pd.to_datetime(v['DATE'], format='%m/%d/%Y')
    return v.groupby(v['DATE'].dt.to_period('M'))['CLOSE'].mean()


def h10_daily(code):
    out = []
    for dec in ('dat89', 'dat96', 'dat00'):
        path = os.path.join(H10, f'{dec}_{code}.' + ('htm' if dec == 'dat00' else 'txt'))
        if not os.path.exists(path): continue
        text = open(path, encoding='latin-1').read()
        if dec == 'dat00': hits = re.findall(r'<th[^>]*>\s*(\d{1,2})-([A-Za-z]{3})-(\d{2})\s*</th>\s*<td[^>]*>\s*(\S+)\s*</td>', text)
        else: hits = re.findall(r'(?m)^\s*(\d{1,2})-([A-Za-z]{3})-(\d{2})\s+(\S+)', text)
        for dd, mon, yy, v in hits:
            try: val = float(v)
            except ValueError: continue                             # ND = no data
            year = 1900 + int(yy) if int(yy) >= 50 else 2000 + int(yy)
            out.append((pd.Timestamp(f'{dd}-{mon.title()}-{year}'), val))
    s = pd.Series(dict(out)).sort_index(); return s[~s.index.duplicated()]


def fx_per_usd_monthly(cur, code, eur_m):
    """Monthly average of the daily noon buying rate in its native H.10 quotation, then expressed as foreign
    currency per dollar. Legacy euro-area currencies are continued from 1999-01 through 2001-12 at their
    irrevocable conversion rates, as in the Board's G.5 release."""
    m = h10_daily(code); m = m.groupby(m.index.to_period('M')).mean()
    lvl = 1.0 / m if cur in USD_PER_FX else m
    if cur in EURO_CONV:
        lvl = lvl[:'1998-12']; ext = (EURO_CONV[cur] / eur_m)['1999-01':'2001-12']; lvl = pd.concat([lvl, ext])
    return lvl


def construct():
    idx = pd.period_range('1971-01', '2019-12', freq='M'); P = pd.DataFrame(index=idx); P.index.name = 'ym'
    # ---- shocks ----
    ac = pd.read_csv(os.path.join(RAW, 'acosta_rrshocks.csv')).dropna(subset=['rr_update']); rr = ac.groupby(pd.to_datetime(ac['fomc']).dt.to_period('M'))['rr_update'].sum()
    P['rr'] = rr; P.loc['1974-01':'2019-12', 'rr'] = P.loc['1974-01':'2019-12', 'rr'].fillna(0.0); P.loc[:'1973-12', 'rr'] = np.nan     # meeting-level shocks summed within the month
    chk = pd.read_csv(os.path.join(LOC, 'rr_ip.csv')); chk.index = pd.PeriodIndex(chk['ym'], freq='M'); assert np.allclose(P['rr'].loc['1974-01':].values, chk['shock'].values), 'narrative series differs from the one used in the paper'
    jk = pd.read_csv(os.path.join(LOC, 'jk_m.csv')); jk.index = ym_index(jk['year'].values, jk['month'].values); P['jk'] = jk['MP_median']
    P.loc['1990-01':'2019-12', 'jk'] = P.loc['1990-01':'2019-12', 'jk'].fillna(0.0)          # months without an FOMC announcement
    oil = pd.read_excel(os.path.join(RAW, 'oilSupplyNewsShocks_2025M12.xlsx'), sheet_name='Monthly (pre-Covid)')
    oil.index = pd.PeriodIndex(oil['Date'].str.replace('M', '-'), freq='M'); P['oil'] = oil['Oil supply news shock']
    bs = pd.read_excel(os.path.join(LOC, 'monetary-policy-surprises-data.xlsx'), sheet_name='Monthly (original)')
    bs.index = pd.PeriodIndex(pd.to_datetime(bs.iloc[:, 0]), freq='M')
    P['bs'] = pd.to_numeric(bs['MPS_ORTH'], errors='coerce'); ebp = pd.to_numeric(bs['EBP'], errors='coerce')
    # ---- domestic outcomes ----
    sh = shiller(); P['sp500'] = 100 * np.log(sh['P']).diff(); P['dy10'] = sh['GS10'].diff()
    P['vix'] = 100 * np.log(vix()).diff(); P['ebp'] = ebp.diff()
    md = pd.read_csv(os.path.join(RAW, 'fred_md_2019-08.csv')).iloc[1:]; md.index = pd.PeriodIndex(pd.to_datetime(md['sasdate']), freq='M')
    md = md.astype({'BAA': float, 'AAA': float, 'TWEXMMTH': float})
    P['spread'] = (md['BAA'] - md['AAA']).diff(); P['dollar'] = 100 * np.log(md['TWEXMMTH']).diff()
    P.loc[:'1973-03', ['spread', 'dollar']] = np.nan; P.loc['2007-12':, ['spread', 'dollar']] = np.nan    # paper's window: 1973-04..2007-11
    # ---- currencies ----
    eur = h10_daily('eu'); eur_m = eur.groupby(eur.index.to_period('M')).mean()
    for cur, code in {**FLOAT9, **EURO10}.items(): P['fx_' + cur] = 100 * np.log(fx_per_usd_monthly(cur, code, eur_m)).diff()
    out = os.path.join(ROOT, 'data', 'analysis_panel.csv'); P.to_csv(out, float_format='%.10g')
    print(f'wrote {out}: {P.shape[0]} months x {P.shape[1]} series'); print(P.notna().sum().to_string()); return P


if __name__ == '__main__':
    if '--offline' not in sys.argv: download()
    construct()
