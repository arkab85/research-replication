"""Summary statistics for the two applications (journal suggestion for empirical papers).

Monetary application: the four monthly outcomes are rebuilt from the FRED files in
finance/data exactly as in finance_study.py, and the transformed 1990-2019 panel is
checked against finance/analysis_panel_sha256.txt before use. If the FRED files are
absent (run finance_download.py first), only the shock rows are computed and the
outcomes are marked pending. Oil and VIX application: from oilvix/raw with the
protocol's definitions. Output: summary_statistics.json.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
P = S.parent
F = P / 'finance'; D = F / 'data'
TRAIN = pd.period_range('1991-01', '2007-12', freq='M'); TEST = pd.period_range('2009-01', '2018-12', freq='M')


def stats(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()),
                median=float(np.median(v)), max=float(v.max()))


out = {'monetary': {}, 'oilvix': {}}

f = pd.read_csv(D / 'jk_monthly.csv')
mp = pd.Series(f.MP_median.to_numpy(), index=pd.PeriodIndex.from_fields(year=f.year, month=f.month, freq='M'))
have_fred = all((D / f'{k}.csv').exists() for k in ['DEXJPUS', 'GS10', 'BAA', 'AAA', 'VIXCLS'])
if have_fred:
    def fred(name):
        g = pd.read_csv(D / (name + '.csv')); g.iloc[:, 1] = pd.to_numeric(g.iloc[:, 1], errors='coerce')
        s = pd.Series(g.iloc[:, 1].to_numpy(dtype=float), index=pd.to_datetime(g.iloc[:, 0])).dropna()
        return s.groupby(s.index.to_period('M')).last()
    base = pd.DataFrame({'MP': mp, 'FX': 100 * np.log(fred('DEXJPUS')).diff(), 'Treasury': fred('GS10').diff(),
                         'Credit': (fred('BAA') - fred('AAA')).diff(), 'VIX': 100 * np.log(fred('VIXCLS')).diff()}).loc['1990':'2019']
    # lineterminator='\n': to_csv defaults to os.linesep, so on Windows this
    # hashed CRLF and never matched the frozen (LF) digest even with identical
    # data. Pinning LF makes the check platform-independent.
    digest = hashlib.sha256(base.to_csv(index_label='month', lineterminator='\n').encode()).hexdigest()
    assert digest == (F / 'analysis_panel_sha256.txt').read_text().strip(), 'FRED vintage differs from the frozen panel'
    cols = ['MP', 'FX', 'Treasury', 'Credit', 'VIX']
else:
    base = pd.DataFrame({'MP': mp}).loc['1990':'2019']; cols = ['MP']
for c in cols:
    out['monetary'][c] = {'training': stats(base[c].reindex(TRAIN)), 'evaluation': stats(base[c].reindex(TEST))}
out['monetary_outcomes_pending'] = not have_fred

w = pd.read_csv(P / 'oilvix/raw/wti.csv'); w.columns = ['date', 'wti']
v = pd.read_csv(P / 'oilvix/raw/vix.csv')[['DATE', 'CLOSE']]; v.columns = ['date', 'vix']
d = w.merge(v, on='date').assign(date=lambda z: pd.to_datetime(z.date)).sort_values('date')
d = d[(d.date >= '1990-01-02') & (d.date <= '2019-12-31')].dropna().reset_index(drop=True)
x = 100 * np.log(d.wti).diff().to_numpy(); dv = 100 * np.log(d.vix).diff().to_numpy()
r = json.load(open(P / 'oilvix/results.json'))['sample']
dates = d.date.dt.strftime('%Y-%m-%d').to_numpy()
tr = (dates >= r['first']) & (dates <= r['training_end']); ev = (dates >= r['evaluation_start']) & (dates <= r['evaluation_end'])
for name, arr in [('Oil', x), ('VIXd', dv)]:
    out['oilvix'][name] = {'training': stats(arr[tr]), 'evaluation': stats(arr[ev])}
(P / 'summary_statistics.json').write_text(json.dumps(out, indent=1))
print('summary statistics written; monetary outcomes pending' if not have_fred else 'summary statistics written')
