"""Admissibility screen for finance's directional questions. Design fixed in code/FINANCE_QUESTIONS.md
and anchored before this was run. Computes tau only: no test, no p-value, no adjudication.

    python code/finance_questions.py
"""
import sys, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PKG, 'flows'))
import numpy as np, pandas as pd
import tau_atlas as A

RAW = os.path.join(PKG, 'data', 'raw'); RES = os.path.join(PKG, 'results')


def shiller():
    D = pd.ExcelFile(os.path.join(RAW, 'ie_data.xls')).parse('Data', header=None)
    h = next(r for r in range(len(D)) if str(D.iloc[r, 0])[:4].isdigit() and '.' in str(D.iloc[r, 0]))
    B = D.iloc[h:, :7].copy(); B.columns = ['date', 'P', 'Dv', 'E', 'CPI', 'frac', 'GS10']
    B = B[pd.to_numeric(B['date'], errors='coerce').notna()]
    for c in ('date', 'P', 'Dv', 'E', 'CPI', 'GS10'): B[c] = pd.to_numeric(B[c], errors='coerce')
    return B.dropna(subset=['P', 'CPI']).reset_index(drop=True)


def fredmd():
    F = pd.read_csv(os.path.join(RAW, 'fred_md_2019-08.csv'))
    codes = F.iloc[0, 1:].astype(float); D = F.iloc[1:].copy(); D = D[D['sasdate'].notna()]
    idx = pd.to_datetime(D['sasdate']); D = D.drop(columns=['sasdate']).astype(float); D.index = idx
    out = {}
    for c in D.columns:
        x = D[c].dropna(); k = codes[c]
        with np.errstate(all='ignore'):
            z = (x if k == 1 else x.diff() if k == 2 else x.diff().diff() if k == 3 else
                 np.log(x) if k == 4 else np.log(x).diff() if k == 5 else np.log(x).diff().diff() if k == 6 else
                 (x / x.shift(1) - 1).diff())
        out[c] = z.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def align(a, b):
    j = pd.concat([pd.Series(a).rename('a'), pd.Series(b).rename('b')], axis=1, join='inner').dropna()
    return j['a'].values.astype(float), j['b'].values.astype(float)


def questions():
    Q = []
    S = shiller()
    P, Dv, E, CPI = S['P'].values, S['Dv'].values, S['E'].values, S['CPI'].values
    idx = pd.PeriodIndex(pd.to_datetime(
        [f'{int(d)}-{min(12, max(1, round((d % 1) * 100))):02d}-01' for d in S['date'].values]), freq='M')
    ret = (P[1:] + Dv[1:] / 12.0) / P[:-1] - 1.0
    infl = CPI[1:] / CPI[:-1] - 1.0
    rreal = pd.Series((1 + ret) / (1 + infl) - 1.0, index=idx[1:])
    dp = pd.Series(np.log(Dv / P), index=idx)
    ep = pd.Series(np.log(np.where(E > 0, E, np.nan) / P), index=idx)
    dg = pd.Series(np.diff(np.log(np.where(Dv > 0, Dv, np.nan))), index=idx[1:])
    Q += [('asset pricing', 'dividend-price ratio', 'real return', dp, rreal),
          ('asset pricing', 'dividend-price ratio', 'dividend growth', dp, dg),
          ('asset pricing', 'earnings-price ratio', 'real return', ep, rreal)]

    F = fredmd()
    ip = F.get('INDPRO')
    SP = F.get('S&P 500')
    for lab, key in (('implied volatility', 'VXOCLSx'), ('federal funds rate', 'FEDFUNDS'),
                     ('unemployment', 'UNRATE'), ('term spread', None), ('credit spread', None)):
        pass
    ts = None
    if 'GS10' in F and 'TB3MS' in F:
        j = pd.concat([F['GS10'].rename('l'), F['TB3MS'].rename('s')], axis=1).dropna()
        ts = (j['l'] - j['s'])
    cs = None
    if 'BAA' in F and 'AAA' in F:
        j = pd.concat([F['BAA'].rename('b'), F['AAA'].rename('a')], axis=1).dropna()
        cs = (j['b'] - j['a'])
    if SP is not None:
        if F.get('VXOCLSx') is not None:
            Q.append(('asset pricing', 'implied volatility', 'market return', F['VXOCLSx'], SP))
        if ts is not None: Q.append(('asset pricing', 'term spread', 'market return', ts, SP))
        if cs is not None: Q.append(('asset pricing', 'credit spread', 'market return', cs, SP))
    if ip is not None:
        if cs is not None: Q.append(('macro-finance', 'credit spread', 'industrial production', cs, ip))
        if F.get('VXOCLSx') is not None:
            Q.append(('macro-finance', 'implied volatility', 'industrial production', F['VXOCLSx'], ip))
        if SP is not None: Q.append(('macro-finance', 'market return', 'industrial production', SP, ip))
        if ts is not None: Q.append(('macro-finance', 'term spread', 'industrial production', ts, ip))
        if F.get('FEDFUNDS') is not None:
            Q.append(('macro-finance', 'federal funds rate', 'industrial production', F['FEDFUNDS'], ip))
        if F.get('UNRATE') is not None:
            Q.append(('macro-finance', 'unemployment', 'industrial production', F['UNRATE'], ip))

    # the excess bond premium, from the paper's own monthly panel
    try:
        AP = pd.read_csv(os.path.join(PKG, 'data', 'analysis_panel.csv'), index_col=0)
        ebp = pd.Series(AP['ebp'].dropna().values,
                        index=pd.PeriodIndex(AP['ebp'].dropna().index.astype(str), freq='M'))
        ipm = pd.Series(ip.values, index=pd.PeriodIndex(ip.index, freq='M')) if ip is not None else None
        if ipm is not None:
            Q.append(('macro-finance', 'excess bond premium', 'industrial production', ebp, ipm))
    except Exception as e:
        print('  (EBP skipped: %s)' % type(e).__name__)
    return Q


VERDICT = lambda t1, t2: ('inadmissible' if max(t1, t2) > .5 else
                          'marginal' if max(t1, t2) > .4 else 'admissible')

if __name__ == '__main__':
    rows = []
    for fam, a_lab, b_lab, a, b in questions():
        x, y = align(a, b)
        if len(x) < 150: print('  (too short, skipped: %s / %s, n=%d)' % (a_lab, b_lab, len(x))); continue
        res = {}
        for d, (u, v) in (('fwd', (x, y)), ('rev', (y, x))):
            r = A.measure(pd.DataFrame({'x': u, 'y': v}), 3, 'sieve')
            res[d] = r
        if res['fwd'] is None or res['rev'] is None: continue
        rows.append(dict(family=fam, cause=a_lab, outcome=b_lab, T=len(x), n=res['fwd']['n'],
                         tau_fwd=res['fwd']['tau'], tau_rev=res['rev']['tau'],
                         bw_fwd=res['fwd']['bw_over_err'], bw_rev=res['rev']['bw_over_err'],
                         verdict=VERDICT(res['fwd']['tau'], res['rev']['tau'])))
    D = pd.DataFrame(rows).sort_values(['family', 'verdict', 'cause'])
    D.round(4).to_csv(os.path.join(RES, 'finance_questions.csv'), index=False)
    print()
    print('%-14s %-24s %-24s %5s %6s %6s  %s' % ('family', 'cause', 'outcome', 'n', 'tau->', 'tau<-', 'verdict'))
    print('-' * 104)
    for _, r in D.iterrows():
        print('%-14s %-24s %-24s %5d %6.3f %6.3f  %s' % (
            r.family, r.cause, r.outcome, r.n, r.tau_fwd, r.tau_rev, r.verdict))
    print()
    print(D.verdict.value_counts().to_string())
