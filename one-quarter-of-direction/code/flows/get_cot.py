"""Download the CFTC Commitments of Traders legacy futures-only history and list coverage by contract.
Sample construction only: no price is read and no flow-return statistic is computed here.

    python flows/get_cot.py            # download (skips files already present) and write data_raw/cot_legacy.csv.gz
"""
import os, io, sys, zipfile, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, 'data_raw'); os.makedirs(RAW, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'}
FILES = ['deacot1986_2016.zip'] + [f'deacot{y}.zip' for y in range(2017, 2027)]
KEEP = {'Market and Exchange Names': 'market', 'As of Date in Form YYYY-MM-DD': 'date', 'CFTC Contract Market Code': 'code',
        'Open Interest (All)': 'oi', 'Noncommercial Positions-Long (All)': 'ncl', 'Noncommercial Positions-Short (All)': 'ncs',
        'Commercial Positions-Long (All)': 'cl', 'Commercial Positions-Short (All)': 'cs'}


def fetch(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        r = requests.get('https://www.cftc.gov/files/dea/history/' + name, headers=UA, timeout=300); r.raise_for_status()
        open(path, 'wb').write(r.content)
    return path


def read(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            d = pd.read_csv(io.BytesIO(z.read(n)), low_memory=False, dtype=str); d.columns = [c.strip() for c in d.columns]
            out.append(d[list(KEEP)].rename(columns=KEEP))
    return pd.concat(out)


if __name__ == '__main__':
    D = pd.concat([read(fetch(f)) for f in FILES]); D['market'] = D['market'].str.strip(); D['code'] = D['code'].str.strip()
    D['date'] = pd.to_datetime(D['date'].str.strip(), errors='coerce'); D = D.dropna(subset=['date'])
    for c in ('oi', 'ncl', 'ncs', 'cl', 'cs'): D[c] = pd.to_numeric(D[c].str.replace(',', '').str.strip(), errors='coerce')
    D = D.drop_duplicates(['code', 'date']).sort_values(['code', 'date'])
    D.to_csv(os.path.join(RAW, 'cot_legacy.csv.gz'), index=False, compression='gzip')
    g = D.groupby('code').agg(market=('market', 'last'), first=('date', 'min'), last=('date', 'max'), weeks=('date', 'size'), oi=('oi', 'median'))
    g.to_csv(os.path.join(RAW, 'cot_coverage.csv')); print(len(D), 'rows;', len(g), 'contracts')
