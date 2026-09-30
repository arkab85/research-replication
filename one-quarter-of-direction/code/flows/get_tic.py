"""Download the Treasury International Capital S1 file (U.S. transactions with foreigners in long-term securities, by
country, monthly) and print its layout. Sample construction only."""
import os, requests
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, 'data_raw'); os.makedirs(RAW, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'}
URL = 'https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/s1_globl.txt'
if __name__ == '__main__':
    path = os.path.join(RAW, 's1_globl.txt')
    if not os.path.exists(path):
        r = requests.get(URL, headers=UA, timeout=600); r.raise_for_status(); open(path, 'wb').write(r.content)
    L = open(path, encoding='latin-1').read().splitlines(); print(len(L), 'lines')
    for i, l in enumerate(L[:22]): print(i, repr(l[:400]))
    print('...'); [print(repr(l[:300])) for l in L[-3:]]
