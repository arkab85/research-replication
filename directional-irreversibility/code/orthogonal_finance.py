"""Secondary, post-results correction audit on the frozen financial design.
Same data, means, units, and 12-cell family; matched nonoverlapping comparators.
Neither result replaces the original primary circular/wild tests.
"""
from pathlib import Path
import hashlib,json,itertools,numpy as np,pandas as pd
from full_study import kernel
from dii_orthogonal import corrected_gram,nbb_dii_from_grams
P=Path(__file__).resolve().parent.parent/'finance'
f=P/'analysis_panel_local.csv';assert hashlib.sha256(f.read_bytes()).hexdigest()==(P/'analysis_panel_sha256.txt').read_text().strip()
base=pd.read_csv(f,index_col='month');base.index=pd.PeriodIndex(base.index,freq='M')
tridx=pd.period_range('1991-01','2007-12',freq='M');teidx=pd.period_range('2009-01','2018-12',freq='M');grams={'raw':{},'orthogonal':{}}
def poly(z):return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])
for asset in ('FX','Treasury','Credit','VIX'):
 for h in (1,3,6):
  f=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)}).reindex(tridx.append(teidx))
  a=f.to_numpy();assert np.isfinite(a).all();mu=a[:204].mean(0);sd=a[:204].std(0);z=(a-mu)/sd
  xf=np.column_stack([z[:,0],z[:,2:]]);xb=np.column_stack([z[:,1],z[:,2:]])
  Xf=poly(xf);Xb=poly(xb)
  rf=z[:,1]-Xf@np.linalg.lstsq(Xf[:204],z[:204,1],rcond=None)[0];rb=z[:,0]-Xb@np.linalg.lstsq(Xb[:204],z[:204,0],rcond=None)[0]
  lf=kernel(xf[204:]);lb=kernel(xb[204:]);name=f'{asset}_h{h}'
  grams['raw'][name]=(kernel(rf[204:]),lf,kernel(rb[204:]),lb)
  grams['orthogonal'][name]=(corrected_gram(rf[204:],xf[204:],rf[:204],xf[:204]),lf,corrected_gram(rb[204:],xb[204:],rb[:204],xb[:204]),lb)
res={};rows=[]
for method,gs in grams.items():
 r=nbb_dii_from_grams(gs,999,20260915);res[method]=r
 order=sorted(r['comparisons'],key=lambda k:r['comparisons'][k]['p_intersection']);prev=0
 for i,k in enumerate(order):
  prev=max(prev,(12-i)*r['comparisons'][k]['p_intersection']);r['comparisons'][k]['p_holm']=min(1.,prev)
 for k,v in r['comparisons'].items():rows.append({'method':method,'comparison':k,**v})
 print(method,'unadjusted',sum(x['p_intersection']<=.05 for x in r['comparisons'].values()),'Holm',sum(x['p_holm']<=.05 for x in r['comparisons'].values()))
res['status']='Post-results secondary audit. Same frozen 12-cell design, corrected and raw NBB comparators. No empirical verification of nuisance risks, mixing or random-scale transfer; no causal certification.'
res['protocol_sha256']=hashlib.sha256((P/'analysis_protocol.md').read_bytes()).hexdigest()
(P/'orthogonal_results.json').write_text(json.dumps(res,indent=2));pd.DataFrame(rows).to_csv(P/'orthogonal_results.csv',index=False)
