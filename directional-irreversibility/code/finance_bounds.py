"""Post-results simultaneous bounds on the frozen 12-cell financial design.
No changes to the original protocol, data, primary results, or fitted model class.
New output is explicitly secondary and uses one path across all 24 operators.
"""
from pathlib import Path
import hashlib,json,itertools,numpy as np,pandas as pd
from dii_confidence import simultaneous_dii_bounds
P=Path(__file__).resolve().parent.parent/'finance'
f=P/'analysis_panel_local.csv'
assert hashlib.sha256(f.read_bytes()).hexdigest()==(P/'analysis_panel_sha256.txt').read_text().strip()
base=pd.read_csv(f,index_col='month');base.index=pd.PeriodIndex(base.index,freq='M')
tridx=pd.period_range('1991-01','2007-12',freq='M');teidx=pd.period_range('2009-01','2018-12',freq='M')
comps={}
def poly(z):
 return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])
for asset in ('FX','Treasury','Credit','VIX'):
 for h in (1,3,6):
  f=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)}).reindex(tridx.append(teidx))
  a=f.to_numpy();assert np.isfinite(a).all();mu=a[:204].mean(0);sd=a[:204].std(0);z=(a-mu)/sd
  xf=np.column_stack([z[:,0],z[:,2:]]);xb=np.column_stack([z[:,1],z[:,2:]])
  Xf=poly(xf);Xb=poly(xb)
  rf=z[204:,1]-Xf[204:]@np.linalg.lstsq(Xf[:204],z[:204,1],rcond=None)[0]
  rb=z[204:,0]-Xb[204:]@np.linalg.lstsq(Xb[:204],z[:204,0],rcond=None)[0]
  comps[f'{asset}_h{h}']=(rf,xf[204:],rb,xb[204:])
r=simultaneous_dii_bounds(comps,1999,20260914)
r['status']='Post-results secondary analysis; new nonoverlapping block algorithm; no empirical verification of mixing, nuisance rates, or random-scaling transfer. Original primary outputs are unchanged.'
r['primary_protocol_sha256']=hashlib.sha256((P/'analysis_protocol.md').read_bytes()).hexdigest()
# Reconstruct the same point statistics; discrepancy reveals implementation error.
old=json.loads((P/'financial_results.json').read_text())['primary_tests'];errors=[]
for x in old:
 y=r['comparisons'][f"{x['asset']}_h{x['h']}"]
 errors.extend(abs(x[k]-y[k]) for k in ['forward_hsic','reverse_hsic','dii'])
assert max(errors)<1e-12
r['max_point_statistic_discrepancy']=max(errors)
(P/'simultaneous_bounds_results.json').write_text(json.dumps(r,indent=2))
pd.DataFrame([{'comparison':k,**v} for k,v in r['comparisons'].items()]).to_csv(P/'simultaneous_bounds.csv',index=False)
print('n',r['n_retained'],'ell',r['block_length'],'max discrepancy',max(errors),'positive lower bounds',sum(x['dii_lower']>0 for x in r['comparisons'].values()))
