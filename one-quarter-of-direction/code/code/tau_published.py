"""Resolution in the designs of the published literature, on the public data those literatures use.
Design fixed in advance in code/PUBLISHED_DESIGNS.md; run once.

tau = share of evaluation pairs with |X_i - X_j| <= |g_i - g_j|, the same formula as code/tau_collapse.py, here
applied to the regressand and fitted values of an autoregression or a VAR equation. Descriptive only.

    python code/tau_published.py
"""
import sys, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from engine import sieve, hsic2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw'); RES = os.path.join(ROOT, 'results')
MIN_OBS = 120; GAP = 6


# ---------------------------------------------------------------- data
def fred_md():
    """FRED-MD with the published transformation codes applied as published."""
    F = pd.read_csv(os.path.join(RAW, 'fred_md_2019-08.csv'))
    codes = F.iloc[0, 1:].astype(float)
    D = F.iloc[1:].copy(); D = D[D['sasdate'].notna()]
    idx = pd.to_datetime(D['sasdate']); D = D.drop(columns=['sasdate']).astype(float); D.index = idx
    out = {}
    for c in D.columns:
        x = D[c].copy(); k = codes[c]
        with np.errstate(divide='ignore', invalid='ignore'):
            if k == 1: z = x
            elif k == 2: z = x.diff()
            elif k == 3: z = x.diff().diff()
            elif k == 4: z = np.log(x)
            elif k == 5: z = np.log(x).diff()
            elif k == 6: z = np.log(x).diff().diff()
            elif k == 7: z = (x / x.shift(1) - 1).diff()
            else: continue
        z = z.replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) >= MIN_OBS: out[c] = z
    return out


def shiller_returns():
    """Monthly S&P Composite returns from Shiller's file: nominal total, real total, and price-only."""
    D = pd.ExcelFile(os.path.join(RAW, 'ie_data.xls')).parse('Data', header=None)
    hdr = None
    for r in range(len(D)):
        v = str(D.iloc[r, 0])
        if v[:4].isdigit() and '.' in v: hdr = r; break
    B = D.iloc[hdr:, :5].copy(); B.columns = ['date', 'P', 'Dv', 'E', 'CPI']
    B = B[pd.to_numeric(B['date'], errors='coerce').notna()]
    for c in ('P', 'Dv', 'CPI'): B[c] = pd.to_numeric(B[c], errors='coerce')
    B = B.dropna(subset=['P', 'CPI']).reset_index(drop=True)
    P = B['P'].values; Dv = B['Dv'].fillna(0).values; CPI = B['CPI'].values
    nom = (P[1:] + Dv[1:] / 12.0) / P[:-1] - 1.0                      # dividend is an annual rate in Shiller's file
    infl = CPI[1:] / CPI[:-1] - 1.0
    pr = P[1:] / P[:-1] - 1.0
    return {'sp_total_nominal': pd.Series(nom), 'sp_total_real': pd.Series((1 + nom) / (1 + infl) - 1.0),
            'sp_price_only': pd.Series(pr)}


# ---------------------------------------------------------------- first stages
def fit_predict(Ztr, ytr, Zev, stage):
    if stage == 'ols':
        return LinearRegression().fit(Ztr, ytr).predict(Zev)
    if stage == 'sieve':
        sc = StandardScaler().fit(Ztr)
        return Ridge(alpha=1.0).fit(sieve(sc.transform(Ztr)), ytr).predict(sieve(sc.transform(Zev)))
    if stage == 'rf':
        return RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1).fit(Ztr, ytr).predict(Zev)
    raise ValueError(stage)


def resolution(y, Z, stage, convention):
    """Fit the conditional mean and return the diagnostics on the evaluation block, or None if excluded."""
    T = len(y)
    if T < MIN_OBS: return None
    if convention == 'insample':
        tr = slice(0, T); ev = slice(0, T)
    else:
        n = int(np.floor(T / np.log(T))); cut = T - n - GAP
        if cut < 30 or n < 20: return None
        tr = slice(0, cut); ev = slice(cut + GAP, T)
    ghat = fit_predict(Z[tr], y[tr], Z[ev], stage)
    Xv = y[ev]
    if np.std(Xv) == 0 or len(Xv) < 20: return None
    e = Xv - ghat
    dx = np.abs(Xv[:, None] - Xv[None, :]); dg = np.abs(ghat[:, None] - ghat[None, :])
    iu = np.triu_indices(len(Xv), 1)
    d2 = (e[:, None] - e[None, :]) ** 2; med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    err = np.std(ghat - np.mean(ghat)) if np.std(ghat) > 0 else 1.0
    return dict(tau=float(np.mean(dx[iu] <= dg[iu])), p2=float(np.mean(dx[iu] == 0)),
                bw_over_err=float(np.sqrt(med) / err), n_hsic=float(len(Xv) * hsic2(e, Z[ev])),
                n=int(len(Xv)), T=int(T))


def lagmat(cols, p):
    """Stack p lags of every column; return (y_rows, Z) aligned so row t uses lags t-1..t-p."""
    M = np.column_stack(cols); T = len(M)
    Z = np.column_stack([M[p - k - 1:T - k - 1] for k in range(p)])
    return Z, slice(p, T)


# ---------------------------------------------------------------- run
if __name__ == '__main__':
    rows, dropped = [], 0

    def add(family, design, p, stage, y, Z):
        global dropped
        for conv in ('split', 'insample'):
            try: r = resolution(y, Z, stage, conv)
            except Exception as e: r = None; print(f'  ! {design} p={p} {stage} {conv}: {type(e).__name__}', flush=True)
            if r is None: dropped += 1; continue
            rows.append(dict(family=family, design=design, p=p, stage=stage, convention=conv, **r))

    print('P1: autoregressive conditional mean on equity returns', flush=True)
    for name, s in shiller_returns().items():
        v = s.values.astype(float)
        for p in (1, 3, 9):
            Z, sl = lagmat([v], p); add('P1-equity-AR', name, p, 'ols', v[sl], Z)

    print('P2: reduced-form autoregressions on every FRED-MD series', flush=True)
    FM = fred_md(); print(f'   {len(FM)} series pass the length filter', flush=True)
    for i, (name, s) in enumerate(sorted(FM.items())):
        v = s.values.astype(float)
        for p in (1, 4, 12):
            Z, sl = lagmat([v], p)
            for stage in ('ols', 'sieve', 'rf'): add('P2-fredmd-AR', name, p, stage, v[sl], Z)
        if (i + 1) % 32 == 0: print(f'   [{i + 1}/{len(FM)}]', flush=True)

    print('P3: named VAR systems', flush=True)
    SYSTEMS = {'blanchard-quah': ['INDPRO', 'UNRATE'],
               'monetary': ['INDPRO', 'CPIAUCSL', 'FEDFUNDS'],
               'oil': ['INDPRO', 'CPIAUCSL', 'OILPRICEx']}
    for sysname, names in SYSTEMS.items():
        if not all(n in FM for n in names): print(f'   ! {sysname}: missing {[n for n in names if n not in FM]}'); continue
        A = pd.concat([FM[n].rename(n) for n in names], axis=1).dropna()
        cols = [A[n].values.astype(float) for n in names]
        for p in (4, 12):
            Z, sl = lagmat(cols, p)
            for j, n in enumerate(names): add('P3-VAR', f'{sysname}:{n}', p, 'ols', cols[j][sl], Z)

    D = pd.DataFrame(rows); os.makedirs(RES, exist_ok=True)
    D.round(6).to_csv(os.path.join(RES, 'tau_published.csv'), index=False)
    print(f'\ndone: {len(D)} measured, {dropped} excluded -> results/tau_published.csv', flush=True)
