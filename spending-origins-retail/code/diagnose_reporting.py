"""Post-estimation missing-status bounds and feasible-composition support check."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from estimate import fit_cluster,infer
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'data/storefront_primary_sample.csv.gz',low_memory=False)
pre=s[s.year==2019].set_index('bbl').sort_index()
x=pre[['residential_share_500','office_share_500','retail_share_500']].copy()
x['density']=np.log(pre.building_500/(np.pi*500**2))
x=pd.concat([x,pd.get_dummies(pre.BoroCode,dtype=float)],axis=1).to_numpy()
w=np.zeros(x.shape[1]);w[0]=10;w[1]=-10
weights=x@np.linalg.solve(x.T@x,w)
weight_map=dict(zip(pre.index,weights))
s['weight']=s.bbl.map(weight_map)*np.where(s.year==2019,-1.,1./5.)
s['lo']=s.vacant/s.registered
s['hi']=(s.vacant+s.registered-s.reported)/s.registered
lower=float(np.where(s.weight>=0,s.weight*s.lo,s.weight*s.hi).sum())
upper=float(np.where(s.weight>=0,s.weight*s.hi,s.weight*s.lo).sum())
missing=int((s.registered-s.reported).sum())
out={'lower_pp':lower,'upper_pp':upper,'missing_status_records':missing,'interpretation':'Finite-sample sharp bounds for the all-registered-storefront outcome in the balanced reporting-parcel sample; all unknown statuses may vary arbitrarily. Not a confidence interval; excludes completely unreported storefronts and absent parcels.'}
(R/'results/reporting_bounds.json').write_text(json.dumps(out,indent=2));print(out)
# Additional check motivated by the exposure-distribution audit, not the original protocol.
ids=pre[(pre.office_share_500>=.1)&(pre.residential_share_500<=.9)].index
d=s[s.bbl.isin(ids)].reset_index(drop=True)
z=pd.DataFrame(index=d.index)
for lab,c in [('R','residential'),('O','office'),('K','retail')]:z[lab+'_post']=d[f'{c}_share_500']*(d.year>2019)
z['density_post']=np.log(d.building_500/(np.pi*500**2))*(d.year>2019)
z=pd.concat([z,pd.get_dummies(d.BoroCode.astype(str)+'_'+d.year.astype(str),dtype=float)],axis=1)
f=fit_cluster(d.vacancy,z,d.bbl,d.CD)
res={'spec':'Feasible ten-point contrast','n':len(d),'parcels':d.bbl.nunique(),'clusters':f['groups'],**infer(f,{'R_post':10,'O_post':-10})}
pd.DataFrame([res]).to_csv(R/'results/storefront_support_check.csv',index=False);print(res)
