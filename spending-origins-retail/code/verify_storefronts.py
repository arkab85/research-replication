"""Independent collapse-based verification and direct spatial/source checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
R=Path(__file__).resolve().parent
s=pd.read_csv(R/'data/storefront_primary_sample.csv.gz',low_memory=False)
assert not s.duplicated(['bbl','year']).any()
assert s.groupby('bbl').year.nunique().eq(6).all()
assert s.vacancy.between(0,1).all() and s.reported.ge(1).all()
assert np.allclose(s.vacancy,s.vacant/s.reported)
pre=s[s.year==2019].set_index('bbl').sort_index()
post=s[s.year>=2020].groupby('bbl').vacancy.mean().sort_index()
y=(post-pre.vacancy).to_numpy()
x=pre[['residential_share_500','office_share_500','retail_share_500']].copy()
x['density']=np.log(pre.building_500/(np.pi*500**2))
x=pd.concat([x,pd.get_dummies(pre.BoroCode,dtype=float)],axis=1).to_numpy()
b=np.linalg.lstsq(x,y,rcond=None)[0]
direct=10*(b[0]-b[1])
est=pd.read_csv(R/'results/storefront_estimates.csv').query("spec=='Primary'").iloc[0]
assert abs(direct-est.estimate)<1e-8
res=y-x@b;g,lev=pd.factorize(pre.CD);scores=np.zeros((len(lev),x.shape[1]));np.add.at(scores,g,x*res[:,None])
bread=np.linalg.inv(x.T@x);vc=bread@scores.T@scores@bread
# Panel CR1 uses all absorbed parcel effects and 29 independent year/borough terms.
# There are five independent calendar changes in each of five boroughs, plus four exposures.
n=len(s);k=len(pre)+5*5+4;ng=len(lev)
w=np.zeros(x.shape[1]);w[0]=10;w[1]=-10
se=float(np.sqrt(w@vc@w*ng/(ng-1)*(n-1)/(n-k)))
assert abs(se-est.se)<1e-8,(se,est.se)
p=pd.read_csv(R/'data/pluto_2016_relevant_fields.csv.gz')
xy=p[['XCoord','YCoord']].to_numpy(float);vals=p[['BldgArea','ResArea','OfficeArea','RetailArea']].fillna(0).clip(lower=0).to_numpy(float)
for _,row in pre.sample(10,random_state=20260913).iterrows():
 dist=np.linalg.norm(xy-np.array([row.XCoord,row.YCoord]),axis=1)*1200/3937
 for rad in [250,500,1000]:
  chosen=(dist<=rad)&(p.BBL.to_numpy()!=row.name)
  brute=vals[chosen].sum(axis=0)
  actual=row[[f'{c}_{rad}' for c in ['building','residential','office','retail']]].to_numpy(float)
  assert np.allclose(brute,actual,rtol=1e-12)
raw=pd.read_csv(R/'raw/storefront_registry.csv',dtype=str).fillna('')
raw['year']=raw.reporting_year.str[:4].astype(int)
raw['bbl']=raw.borough_block_lot.astype('int64')
raw['v']=raw.vacant_on_12_31.map({'YES':1,'Y':1,'NO':0,'N':0})
for _,row in s.sample(20,random_state=193).iterrows():
 rr=raw[(raw.bbl==row.bbl)&(raw.year==row.year)]
 assert len(rr)==row.registered and rr.v.count()==row.reported and rr.v.sum()==row.vacant
out={'balanced_panel':True,'independent_collapsed_coefficient':direct,'independent_collapsed_se_with_panel_correction':se,'regression_matches_to':1e-8,'direct_spatial_checks':30,'registration_aggregation_checks':20,'no_duplicate_parcel_years':True,'all_outcomes_valid':True}
(R/'results/storefront_verification.json').write_text(json.dumps(out,indent=2));print(out)
