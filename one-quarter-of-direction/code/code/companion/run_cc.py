"""Companion-paper cross-country CPI panel (seven economies x {oil news, Fed} x h in {6,12,24}), rebuilt from
cc.py and panel.py and checked against the stored companion_stored_results/cc7_results.pkl.

    python code/companion/run_cc.py            # B = 999 wild, 299 paired, as stored
"""
import sys, os, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from engine import *; from panel import pooled_tests; import cc
ROOT = os.path.dirname(os.path.dirname(HERE)); ST = os.path.join(ROOT, 'companion_stored_results')


def load():
    """cc.load() with the package's paths (the original read data/kaenzig_*.xlsx relative to a sandbox directory)."""
    v = pd.read_excel(os.path.join(ROOT, 'data', 'raw', 'kaenzig_VARdata.xlsx'), sheet_name='Monthly'); v['ym'] = pd.PeriodIndex(v['Date'].str.replace('M', '-'), freq='M')
    out = v[['ym']].copy(); out['cpi_us'] = v['CPI'].values
    for c in ['uk', 'de', 'jp', 'fr', 'it', 'ca']:
        d = pd.read_csv(os.path.join(ST, 'data', f'cpi_{c}.csv')); d['ym'] = pd.PeriodIndex(d['ym'], freq='M'); out = out.merge(d, on='ym', how='left')
    sh = pd.read_excel(os.path.join(ROOT, 'data', 'raw', 'oilSupplyNewsShocks_2025M12.xlsx'), sheet_name='Monthly (pre-Covid)'); sh['ym'] = pd.PeriodIndex(sh['Date'].str.replace('M', '-'), freq='M')
    out = out.merge(sh[['ym', 'Oil supply news shock']].rename(columns={'Oil supply news shock': 'oil'}), on='ym', how='left')
    jk = pd.read_csv(os.path.join(ROOT, 'data', 'local_inputs', 'jk_m.csv')); jk['ym'] = pd.PeriodIndex(pd.to_datetime(dict(year=jk['year'], month=jk['month'], day=1)), freq='M')
    out = out.merge(jk[['ym', 'MP_median']].rename(columns={'MP_median': 'fed'}), on='ym', how='left')
    for c in ['us', 'uk', 'de', 'jp', 'fr', 'it', 'ca']: out[f'pi_{c}'] = np.log(out[f'cpi_{c}']).diff() * 100
    return out


# Month-of-year means (estimated on the training block only) are removed from the CPI series that are published
# not seasonally adjusted; the US and German series are already adjusted. These two settings, and the 1990-01
# start of the Fed sample (months without an announcement set to zero), were recovered by matching the stored
# point estimates exactly; they were not recorded in the original drivers.
SEASONAL = dict(us=False, uk=True, de=False, jp=True, fr=True, it=True, ca=True)


def run(B_w=999, B_p=299):
    d = load(); d = d[d['ym'] <= '2019-12']; res = {}
    for shock, lo in (('oil', '1975-01'), ('fed', '1990-01')):
        g = d[d['ym'] >= lo].copy()
        if shock == 'fed': g['fed'] = g['fed'].fillna(0.0)
        g = g.dropna(subset=[shock] + [f'pi_{c}' for c in SEASONAL])
        for h in (6, 12, 24):
            us = [cc.unit_cc(g[shock].values, g[f'pi_{c}'].values, g['ym'].astype(str).values, h, seasonal=s) for c, s in SEASONAL.items()]
            P, pw, pp = pooled_tests(us, B_w, B_p); res[(shock, h)] = (P, pw, pp, [u['obs'] for u in us], [u['diag'] for u in us], us[0]['n'])
    return res


if __name__ == '__main__':
    stored = pickle.load(open(os.path.join(ST, 'cc7_results.pkl'), 'rb'))
    res = run(*((99, 29) if '--quick' in sys.argv else (999, 299)))
    pickle.dump(res, open(os.path.join(ROOT, 'results', 'companion_cc7_results.pkl'), 'wb'))
    for k, v in res.items():
        s = stored[k]; print(f'{k}: P={v[0]:+.6f} (stored {s[0]:+.6f})  wild {v[1]:.3f} ({s[1]:.3f})  paired {v[2]:.3f} ({s[2]:.3f})  n={v[5]} ({s[5]})  max|DII diff|={np.max(np.abs(np.array(v[3]) - np.array(s[3]))):.1e}')
