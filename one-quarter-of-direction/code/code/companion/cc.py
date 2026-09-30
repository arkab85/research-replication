import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from engine import *; from panel import pooled_tests
import pandas as pd
def load():
    v=pd.read_excel('data/kaenzig_var.xlsx',sheet_name='Monthly'); v['ym']=pd.PeriodIndex(v['Date'].str.replace('M','-'),freq='M')
    out=v[['ym']].copy(); out['cpi_us']=v['CPI'].values
    for c in ['uk','de','jp','fr','it','ca']:
        d=pd.read_csv(f'data/cc/cpi_{c}.csv'); d['ym']=pd.PeriodIndex(d['ym'],freq='M'); out=out.merge(d,on='ym',how='left')
    sh=pd.read_excel('data/kaenzig_shocks.xlsx',sheet_name='Monthly (pre-Covid)'); sh['ym']=pd.PeriodIndex(sh['Date'].str.replace('M','-'),freq='M'); out=out.merge(sh[['ym','Oil supply news shock']].rename(columns={'Oil supply news shock':'oil'}),on='ym',how='left')
    jk=pd.read_csv('data/jk_m.csv'); jk['ym']=pd.PeriodIndex(pd.to_datetime(dict(year=jk['year'],month=jk['month'],day=1)),freq='M'); out=out.merge(jk[['ym','MP_median']].rename(columns={'MP_median':'fed'}),on='ym',how='left')
    out['fed']=out['fed'].fillna(0.0)
    for c in ['us','uk','de','jp','fr','it','ca']: out[f'pi_{c}']=np.log(out[f'cpi_{c}']).diff()*100
    return out
def deseason(y,tr):
    """month-of-year means estimated on the training block only, removed from the whole series"""
    m=np.arange(len(y))%12; adj=y.copy(); mu=np.array([np.nanmean(y[:tr][m[:tr]==k]) for k in range(12)]); return adj-mu[m]+np.nanmean(mu)
def unit_cc(X,Y,dates,h,p=2,gap=12,seasonal=False):
    T=len(X); n=int(np.floor(T/np.log(T))); tr=T-n-gap
    if seasonal: Y=deseason(Y,tr)
    dtr=build(X[:tr],Y[:tr],h,p); dte=build(X[tr+gap:],Y[tr+gap:],h,p); s1=Stage1().fit(dtr); ef,eb=s1.resid(dte)
    return dict(s1=s1,dte=dte,dates=np.array(dates[tr+gap+p: tr+gap+p+len(dte)]),Q=core_q(ef,eb,dte),obs=dii(ef,eb,dte),diag=len(dte)*hsic2(ef,dte[dte.attrs['f']].values),n=len(dte))
