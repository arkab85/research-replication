"""Design layer for 'One Quarter of Direction': honest-split units, the rule-based pair list, and pooled tests
with one calendar-indexed weight path and one set of block starts shared across units (and across horizons
for contrasts such as the hump). The estimator itself (sieve first stage, HSIC, both bootstraps) is engine.py."""
import os
from engine import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
HS = (1, 3, 6); P_LAGS = 2; GAP = 6
START = {'rr': '1974-02', 'jk': '1990-01', 'oil': '1975-01', 'bs': '1988-02'}
SHOCK_NAME = {'rr': 'Narrative MP', 'jk': 'High-freq. MP', 'oil': 'Oil news', 'bs': 'Bauer-Swanson MP'}
PV = [(x, y) for x in ('rr', 'jk', 'oil', 'bs') for y in ('sp500', 'dy10', 'vix')]              # twelve price and volatility pairs
PAIRS18 = PV + [('rr', 'ebp'), ('jk', 'ebp'), ('bs', 'ebp'), ('rr', 'spread'), ('rr', 'dollar'), ('oil', 'spread')]
FLOAT9 = ['AUD', 'CAD', 'DKK', 'JPY', 'NZD', 'NOK', 'SEK', 'CHF', 'GBP']
EURO10 = ['DEM', 'FRF', 'ITL', 'NLG', 'BEF', 'ATS', 'FIM', 'IEP', 'ESP', 'PTE']


def load_panel():
    D = pd.read_csv(os.path.join(ROOT, 'data', 'analysis_panel.csv'), index_col=0); D.index = D.index.astype(str)
    D['ebp'] = 100 * D['ebp']                       # EBP change enters in basis points (see README, 'Units')
    return D


def long_shock(D, x):
    """The shock on its full available calendar, zero elsewhere: the narrative series from 1969 (Acosta's file) and the
    Jarocinski-Karadi file through 2024. Placebo calendars shift THIS series, so a shift never shortens the sample and
    months shifted in from outside the estimation window carry their actual values (recovered by matching Table 5)."""
    idx = pd.period_range('1965-01', '2026-12', freq='M').astype(str)
    if x == 'rr':
        a = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'acosta_rrshocks.csv')); a['ym'] = pd.to_datetime(a['fomc']).dt.to_period('M').astype(str); s = a.dropna(subset=['rr_update']).groupby('ym')['rr_update'].sum()
    elif x == 'jk':
        j = pd.read_csv(os.path.join(ROOT, 'data', 'local_inputs', 'jk_m.csv')); s = pd.Series(j['MP_median'].values, index=pd.PeriodIndex(pd.to_datetime(dict(year=j['year'], month=j['month'], day=1)), freq='M').astype(str))
    else: s = D[x].dropna()
    return s.reindex(idx).fillna(0.0)


def sample(D, x, y, start=None, end='2019-12', xshift=0):
    xs = long_shock(D, x).shift(xshift).fillna(0.0).reindex(D.index) if xshift else D[x]
    d = pd.concat([xs.rename('x'), D[y].rename('y')], axis=1).loc[start or START[x]:end].dropna(); return d


def unit(d, h, p=P_LAGS, gap=GAP):
    """Honest split: first stage on the first T-n-gap months, gap discarded, index and cores on the last n=floor(T/log T)."""
    X = d['x'].values; Y = d['y'].values; dates = d.index.values; T = len(d); n = int(np.floor(T / np.log(T))); tr = T - n - gap
    s1 = Stage1().fit(build(X[:tr], Y[:tr], h, p)); dte = build(X[tr + gap:], Y[tr + gap:], h, p); ef, eb = s1.resid(dte)
    return dict(s1=s1, dte=dte, dates=np.array(dates[tr + gap + p: tr + gap + p + len(dte)]), Q=core_q(ef, eb, dte), obs=dii(ef, eb, dte),
                diag=len(dte) * hsic2(ef, dte[dte.attrs['f']].values), n=len(dte), T=T, nT=n)


def cell_tests(u, B_w=499, B_p=199, seed=0):
    rng = np.random.RandomState(seed); _, pw, _ = wild_p(u['Q'], u['n'], B_w, rng)
    rng = np.random.RandomState(seed + 1); _, pp, _ = paired_p(u['s1'], u['dte'], B_p, rng); return pw, pp


def pooled_contrast(U, coefs, B_w=499, B_p=149, seed=0):
    """U: {h: [unit, ...]}, coefs: {h: c_h}. Statistic = sum_h c_h * mean_u DII_u(h).
    Wild: one AR(1) weight path on the union calendar, read off at each unit's test dates, applied to every core.
    Paired: one vector of relative block starts shared by all units and horizons; first stages held fixed.
    With a single horizon and c_h=1 this is panel.pooled_tests of the companion package, draw for draw."""
    hs = [h for h in coefs if coefs[h] != 0]; units = [u for h in hs for u in U[h]]
    allD = sorted(set(np.concatenate([u['dates'] for u in units]))); idx = {d: i for i, d in enumerate(allD)}; L = len(allD)
    nbar = int(np.mean([len(u['dte']) for u in units])); ell = max(2, int(round(nbar ** (1 / 3))))
    P = sum(coefs[h] * np.mean([u['obs'] for u in U[h]]) for h in hs)
    pos = {id(u): np.array([idx[d] for d in u['dates']]) for u in units}
    rng = np.random.RandomState(seed); bw = np.empty(B_w)
    for b in range(B_w):
        W = wild_weights(L, ell, rng)
        bw[b] = sum(coefs[h] * np.mean([(lambda w: w @ u['Q'] @ w / len(w) ** 2)(W[pos[id(u)]]) for u in U[h]]) for h in hs)
    pw = (np.sum(bw >= P) + 1) / (B_w + 1)
    pp = np.nan
    if B_p:
        rng = np.random.RandomState(seed + 1); bp = np.empty(B_p)
        for b in range(B_p):
            uu = rng.rand(int(np.ceil(max(len(u['dte']) for u in units) / ell)) + 1); tot = 0.0
            for h in hs:
                vals = []
                for u in U[h]:
                    n = len(u['dte']); nb = int(np.ceil(n / ell)); starts = (uu[:nb] * (n - ell + 1)).astype(int)
                    ix = np.concatenate([np.arange(s, s + ell) for s in starts])[:n]; e = sub(u['dte'], ix); e1, e2 = u['s1'].resid(e); vals.append(dii(e1, e2, e))
                tot += coefs[h] * np.mean(vals)
            bp[b] = tot
        pp = (np.sum((bp - P) >= P) + 1) / (B_p + 1)
    return P, pw, pp


def pooled(units, B_w=499, B_p=149, seed=0): return pooled_contrast({0: units}, {0: 1.0}, B_w, B_p, seed)
HUMP = {1: -0.5, 3: 1.0, 6: -0.5}; D31 = {3: 1.0, 1: -1.0}; D36 = {3: 1.0, 6: -1.0}


def save(df, name):
    path = os.path.join(RES, name); df.to_csv(path, index=False, float_format='%.6g'); print(f'\n== {name}'); print(df.to_string(index=False)); return df
