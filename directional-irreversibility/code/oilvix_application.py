"""Prespecified second application (see oilvix/protocol.md). Run once; report all output."""
from pathlib import Path
import hashlib, json, sys, urllib.request
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
P = S.parent
O = P / 'oilvix'
sys.path.insert(0, str(S))
from dii_joint_nuisance import prepare_direction, joint_nuisance_dii  # noqa: E402

def poly2(z):
    z = np.atleast_2d(z)
    cols = [np.ones(len(z))] + [z[:, i] for i in range(z.shape[1])]
    for i in range(z.shape[1]):
        for j in range(i, z.shape[1]):
            cols.append(z[:, i] * z[:, j])
    return np.column_stack(cols)


URLS = {'wti': 'https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv',
        'vix': 'https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv'}
RAW = O / 'raw'; RAW.mkdir(exist_ok=True)
for k, u in URLS.items():
    f = RAW / f'{k}.csv'
    if not f.exists():
        urllib.request.urlretrieve(u, f)
(O / 'data_sha256.txt').write_text(''.join(
    f"{hashlib.sha256((RAW / f'{k}.csv').read_bytes()).hexdigest()}  {k}.csv  {URLS[k]}\n" for k in URLS))

w = pd.read_csv(RAW / 'wti.csv'); w.columns = ['date', 'wti']
v = pd.read_csv(RAW / 'vix.csv')[['DATE', 'CLOSE']]; v.columns = ['date', 'vix']
d = w.merge(v, on='date').assign(date=lambda z: pd.to_datetime(z.date)).sort_values('date')
d = d[(d.date >= '1990-01-02') & (d.date <= '2019-12-31')].dropna().reset_index(drop=True)
assert (d.wti > 0).all() and (d.vix > 0).all()
x = 100 * np.log(d.wti).diff().to_numpy()
dv = 100 * np.log(d.vix).diff().to_numpy()

H = [0, 1, 5]
T = len(d)
orig = np.arange(2, T - max(H))                 # needs X_{t-1} and dV_{t-1}
M = N = 2048
tr, ev = orig[:M], orig[-N:]
use = np.r_[tr, ev]
te = np.arange(M, M + N)
Cm = np.column_stack([x[use - 1], dv[use - 1]])
comps = {}
for h in H:
    X = x[use]; Y = dv[use + h]
    fwd = prepare_direction(Y, np.column_stack([X, Cm]), M, te, poly2)
    rev = prepare_direction(X, np.column_stack([Y, Cm]), M, te, poly2)
    comps[f'h{h}'] = [fwd, rev]


res = joint_nuisance_dii(comps, bootstrap_draws=999, seed=20260920, return_bounds=True)
p = sorted((v['p_intersection'], k) for k, v in res['comparisons'].items())
holm, run = {}, 0.0
for i, (pv, k) in enumerate(p):
    run = max(run, min(1.0, (len(p) - i) * pv)); holm[k] = run
for k, v in res['comparisons'].items():
    v['p_holm'] = holm[k]
res['sample'] = dict(first=str(d.date[tr[0]].date()), training_end=str(d.date[tr[-1]].date()),
                     evaluation_start=str(d.date[ev[0]].date()), evaluation_end=str(d.date[ev[-1]].date()),
                     common_days=T, training=M, evaluation=N)
json.dump(res, open(O / 'results.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
