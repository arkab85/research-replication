from pathlib import Path
import json,hashlib,numpy as np,pandas as pd,itertools
from dii_joint_nuisance import prepare_direction,joint_nuisance_dii,certify_original_dii
P=Path(__file__).resolve().parent.parent/'finance';f=P/'analysis_panel_local.csv';assert hashlib.sha256(f.read_bytes()).hexdigest()==(P/'analysis_panel_sha256.txt').read_text().strip()
base=pd.read_csv(f,index_col='month');base.index=pd.PeriodIndex(base.index,freq='M');tr=pd.period_range('1991-01','2007-12',freq='M');te=pd.period_range('2009-01','2018-12',freq='M');comps={}
def poly(z):return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])
for asset in ('FX','Treasury','Credit','VIX'):
 for h in (1,3,6):
  a=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)}).reindex(tr.append(te)).to_numpy();assert np.isfinite(a).all()
  x=a[:,0];y=a[:,1];zf=np.column_stack([x,a[:,2:]]);zb=np.column_stack([y,a[:,2:]])
  comps[f'{asset}_h{h}']=[prepare_direction(y,zf,204,np.arange(204,324),poly),prepare_direction(x,zb,204,np.arange(204,324),poly)]
r=joint_nuisance_dii(comps,999,20260918,True,True);old=json.loads((P/'training_aware_results.json').read_text());maxerr=0
for k,v in r['comparisons'].items():
 for x in ('forward_hsic','reverse_hsic','dii'):maxerr=max(maxerr,abs(v[x]-old['comparisons'][k][x]))
assert maxerr<1e-12
order=sorted(r['comparisons'],key=lambda k:r['comparisons'][k]['p_intersection']);prev=0
for i,k in enumerate(order):prev=max(prev,(12-i)*r['comparisons'][k]['p_intersection']);r['comparisons'][k]['p_holm']=min(1.,prev)
r['point_estimate_max_difference']=maxerr
r['original_dii_certificates']=certify_original_dii(r)
r['status']='Secondary post-results audit. Joint fitted coefficient, scale, and centering uncertainty included. Under misspecification inference concerns projection residuals, not the original conditional-mean DII. Dependence/stability remain empirical assumptions. No original-DII robust sign claim without a justified approximation-error bound.'
(P/'joint_nuisance_results.json').write_text(json.dumps(r,indent=2));pd.DataFrame([dict(comparison=k,**v) for k,v in r['comparisons'].items()]).to_csv(P/'joint_nuisance_results.csv',index=False)
print('Unadjusted',sum(v['p_intersection']<=.05 for v in r['comparisons'].values()),'Holm',sum(v['p_holm']<=.05 for v in r['comparisons'].values()),'positive simultaneous lower',sum(v['dii_lower95']>0 for v in r['comparisons'].values()),'max_point_difference',maxerr,'radius',r['joint_radius95'])
