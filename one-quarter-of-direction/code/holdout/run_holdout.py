"""Hold-out tests of holdout/PROTOCOL.md: new outcomes (foreign equity indices), new shock families (Swanson factors,
Bu-Rogers-Wu), new currencies (seven emerging-market currencies). Downloads what is missing, builds
holdout/holdout_panel.csv, runs exact rotation tests.        python holdout/run_holdout.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, os.path.join(ROOT, 'code'))
from rotation import *
import build_data as bd
RAW = os.path.join(HERE, 'data_raw'); os.makedirs(os.path.join(RAW, 'h10'), exist_ok=True); UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36'}
IDX = {'FTSE': '%5EFTSE', 'DAX': '%5EGDAXI', 'CAC': '%5EFCHI', 'NIKKEI': '%5EN225', 'HSI': '%5EHSI', 'TSX': '%5EGSPTSE', 'AORD': '%5EAORD', 'SMI': '%5ESSMI'}
EM = {'MXN': 'mx', 'BRL': 'bz', 'KRW': 'ko', 'ZAR': 'sf', 'INR': 'in', 'THB': 'th', 'SGD': 'si'}
SRC = {'swanson_factors.xlsx': 'https://sites.socsci.uci.edu/~swanson2/papers/pre-and-post-ZLB-factors-extended.xlsx', 'brw_shock.csv': 'https://www.federalreserve.gov/econres/feds/files/brw-shock-series.csv'}
SRC.update({f'yahoo_{k}.json': f'https://query1.finance.yahoo.com/v8/finance/chart/{v}?period1=0&period2=1790000000&interval=1mo' for k, v in IDX.items()})
SRC.update({f'h10/{dec}_{c}.{"htm" if dec == "dat00" else "txt"}': f'https://www.federalreserve.gov/releases/h10/hist/{dec}_{c}.{"htm" if dec == "dat00" else "txt"}' for c in EM.values() for dec in ('dat89', 'dat96', 'dat00')})


def download():
    import requests
    for name, url in SRC.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000: continue
        r = requests.get(url, headers=UA, timeout=120)
        if r.status_code != 200: print('  not available:', name, r.status_code); continue
        open(path, 'wb').write(r.content); print(f'downloaded {name} {len(r.content):,}')


def build():
    D = load_panel(); P = D[['rr', 'jk', 'bs', 'oil', 'sp500', 'dy10']].copy()
    for k in IDX:
        j = json.load(open(os.path.join(RAW, f'yahoo_{k}.json')))['chart']['result'][0]; s = pd.Series(j['indicators']['quote'][0]['close'], index=pd.to_datetime(j['timestamp'], unit='s')).dropna()
        s.index = (s.index + pd.Timedelta(days=3)).to_period('M').astype(str); s = s[~s.index.duplicated(keep='last')]          # Yahoo stamps a month at (or just before) its first day
        P['eq_' + k] = (100 * np.log(s).diff()).reindex(P.index)
    sw = pd.read_excel(os.path.join(RAW, 'swanson_factors.xlsx'), sheet_name='Data', header=None, skiprows=2).iloc[:, 1:5]; sw.columns = ['date', 'sw_ffr', 'sw_fg', 'sw_lsap']
    sw = sw[pd.to_datetime(sw['date'], errors='coerce').notna()]; sw['ym'] = pd.to_datetime(sw['date']).dt.to_period('M').astype(str); m = sw.groupby('ym')[['sw_ffr', 'sw_fg', 'sw_lsap']].sum().astype(float)
    for c in m.columns: P[c] = m[c].reindex(P.index); P.loc[m.index.min():m.index.max(), c] = P.loc[m.index.min():m.index.max(), c].fillna(0.0)
    b = pd.read_csv(os.path.join(RAW, 'brw_shock.csv')).iloc[:, :2]; b.columns = ['month', 'brw']; b = b[b['month'].astype(str).str.match(r'^\d{4}m\d+$')]
    b.index = pd.PeriodIndex(b['month'].str.replace('m', '-'), freq='M').astype(str); P['brw'] = b['brw'].astype(float).reindex(P.index)
    old = bd.H10; bd.H10 = os.path.join(RAW, 'h10')
    for cur, code in EM.items():
        s = bd.h10_daily(code); s = 100 * np.log(s.groupby(s.index.to_period('M')).mean()).diff(); s.index = s.index.astype(str); P['fx_' + cur] = s.reindex(P.index)
    bd.H10 = old; P.to_csv(os.path.join(HERE, 'holdout_panel.csv'), float_format='%.10g'); print(P.notna().sum().to_string()); return P


def smp(P, x, y, start): return pd.concat([P[x].rename('x'), P[y].rename('y')], axis=1).loc[start:'2019-12'].dropna()
if __name__ == '__main__':
    download(); P = build(); rows = []; ST = dict(START, sw_ffr='1991-07', sw_fg='1991-07', sw_lsap='1991-07', brw='1994-01'); audit = []
    def admit(pairs):
        out = []
        for x, y in pairs:
            n = len(smp(P, x, y, ST[x])); audit.append(dict(shock=x, outcome=y, overlap=n, admitted=n >= 300))
            if n >= 300: out.append((x, y))
        return out
    A = admit([(x, 'eq_' + k) for x in ('rr', 'jk', 'bs') for k in IDX]); Ao = admit([('oil', 'eq_' + k) for k in IDX])
    report(rows, 'A. foreign equity indices', A + Ao, end_split([smp(P, x, y, ST[x]) for x, y in A + Ao]), {'monetary shocks x foreign equity (primary)': A, 'narrative': [q for q in A if q[0] == 'rr'], 'Jarocinski-Karadi': [q for q in A if q[0] == 'jk'],
           'Bauer-Swanson': [q for q in A if q[0] == 'bs'], 'oil supply news (secondary)': Ao, 'Jarocinski-Karadi and oil': [q for q in A if q[0] == 'jk'] + Ao})
    Bp = admit([(x, y) for x in ('sw_ffr', 'sw_fg', 'sw_lsap', 'brw') for y in ('sp500', 'dy10')])
    report(rows, 'B. new shock families', Bp, end_split([smp(P, x, y, ST[x]) for x, y in Bp]), {'Swanson factors and BRW x S&P 500, 10y yield (primary)': Bp, 'Swanson federal funds rate factor': [q for q in Bp if q[0] == 'sw_ffr'],
           'Swanson forward guidance factor': [q for q in Bp if q[0] == 'sw_fg'], 'Swanson LSAP factor': [q for q in Bp if q[0] == 'sw_lsap'], 'Bu-Rogers-Wu': [q for q in Bp if q[0] == 'brw']})
    C = admit([(x, 'fx_' + c) for x in ('jk', 'bs') for c in EM])
    report(rows, 'C. emerging-market currencies', C, end_split([smp(P, x, y, ST[x]) for x, y in C]), {'both shocks x EM currencies (primary: P(1))': C, 'Jarocinski-Karadi': [q for q in C if q[0] == 'jk'], 'Bauer-Swanson': [q for q in C if q[0] == 'bs']})
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True); pd.DataFrame(audit).to_csv(os.path.join(HERE, 'results', 'holdout_membership.csv'), index=False)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'results', 'holdout_rotation_tests.csv'), index=False, float_format='%.6g')
    for (bl, s), g in R.groupby(['block', 'set'], sort=False):
        f = lambda st: g[g.statistic == st].iloc[0]; print(f"{bl[:2]} {s[:52]:52s} n={int(g.pairs.iloc[0]):2d} | P1,3,6 = {f('P(1)').value:+.4f} {f('P(3)').value:+.4f} {f('P(6)').value:+.4f} | p1={f('P(1)').p_rotation:.3f} p3={f('P(3)').p_rotation:.3f} p6={f('P(6)').p_rotation:.3f} hump={f('hump').p_rotation:.3f} 3-6={f('P(3)-P(6)').p_rotation:.3f} (min {f('hump').min_p:.3f}) pos={int(f('P(1)').positive)}/{int(f('P(3)').positive)}/{int(f('P(6)').positive)}")
