"""Create a parcel-year reported-vacancy panel; missing status is never occupied."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'raw/storefront_registry.csv',dtype=str).fillna('')
d['year']=d.reporting_year.str.extract(r'(20\d\d)',expand=False).astype(int)
d['bbl']=pd.to_numeric(d.borough_block_lot,errors='raise').astype('int64')
d['status']=d.vacant_on_12_31.str.strip().str.upper().map({'YES':1.,'Y':1.,'NO':0.,'N':0.})
d['construction']=d.construction_reported.str.strip().str.upper().isin(['YES','Y']).astype(int)
summary={'raw_rows':len(d),'years':d.year.value_counts().sort_index().to_dict(),
 'status_codes':d.vacant_on_12_31.value_counts().to_dict()}
d.groupby('year').agg(registrations=('bbl','size'),explicit=('status','count'),vacant=('status','sum')).assign(missing=lambda x:x.registrations-x.explicit, vacancy_rate=lambda x:x.vacant/x.explicit).to_csv(R/'results/storefront_coverage.csv')
d=d[d.year.between(2019,2024)].copy()
# Rows cannot be deduplicated into units: the public file has no stable storefront ID.
a=d.groupby(['bbl','year']).agg(registered=('status','size'),reported=('status','count'),vacant=('status','sum'),construction=('construction','max'),zip_code=('zip_code','first')).reset_index()
a['vacancy']=a.vacant/a.reported.replace(0,np.nan)
a['completeness']=a.reported/a.registered
a=a.sort_values(['bbl','year'])
a['zip_fixed']=a.groupby('bbl').zip_code.transform(lambda s:next((x for x in s if x), 'missing'))
p=pd.read_csv(R/'data/pluto_2016_relevant_fields.csv.gz')
summary['registered_parcels']=int(a.bbl.nunique())
f=p[p.BBL.isin(a.bbl)].copy()
summary['matched_parcels']=len(f)
xy=p[['XCoord','YCoord']].to_numpy(float);tree=cKDTree(xy)
v=p[['BldgArea','ResArea','OfficeArea','RetailArea']].fillna(0).clip(lower=0).to_numpy(float)
fi=f.index.to_numpy();out=np.zeros((len(f),3,4));radii=[250,500,1000]
for start in range(0,len(f),256):
 ids=fi[start:start+256]
 nbs=tree.query_ball_point(xy[ids],1000*3937/1200,workers=1)
 for j,(idx,nb) in enumerate(zip(ids,nbs)):
  nb=np.asarray(nb,dtype=int);nb=nb[nb!=idx]
  dist=np.linalg.norm(xy[nb]-xy[idx],axis=1)*1200/3937
  for k,rad in enumerate(radii):out[start+j,k]=v[nb[dist<=rad]].sum(axis=0)
 if start%5120==0:print('Storefront neighborhoods',start,len(f),flush=True)
f=f.rename(columns={'BBL':'bbl'}).reset_index(drop=True)
for k,rad in enumerate(radii):
 for j,name in enumerate(['building','residential','office','retail']):f[f'{name}_{rad}']=out[:,k,j]
 for name in ['residential','office','retail']:f[f'{name}_share_{rad}']=f[f'{name}_{rad}']/f[f'building_{rad}'].replace(0,np.nan)
 f[f'coverage_{rad}']=f[[f'{name}_share_{rad}' for name in ['residential','office','retail']]].sum(axis=1)
f.to_csv(R/'data/storefront_exposures.csv.gz',index=False)
a=a.merge(f,on='bbl',how='inner',validate='many_to_one')
a.to_csv(R/'data/storefront_panel_all.csv.gz',index=False)
summary['matched_parcel_years']=len(a)
(R/'results/storefront_construction.json').write_text(json.dumps(summary,indent=2))
print(summary,flush=True)
