"""Implementation checks, separate from the empirical evidence in the paper."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import linalg
from estimate import fit_cluster
R=Path(__file__).resolve().parent
# Independent full dummy-variable regression on a constructed numerical test design.
rng=np.random.default_rng(9132026);n=600
absorb=np.repeat(np.arange(60),10);cluster=np.repeat(np.arange(30),20)
x=pd.DataFrame(rng.normal(size=(n,4)),columns=list('abcd'))
y=x.to_numpy()@np.array([.2,-.5,1.1,.7])+rng.normal(size=60)[absorb]+rng.normal(size=n)
f=fit_cluster(y,x,absorb,cluster)
X=np.column_stack([x.to_numpy(),np.eye(60)[absorb]])
b=linalg.lstsq(X,y)[0];resid=y-X@b;bread=linalg.inv(X.T@X)
score=np.zeros((30,X.shape[1]));np.add.at(score,cluster,X*resid[:,None])
cov=bread@score.T@score@bread*30/29*(n-1)/(n-X.shape[1])
np.testing.assert_allclose(f['beta'].to_numpy(),b[:4],atol=1e-10,rtol=1e-10)
np.testing.assert_allclose(f['cov'].to_numpy(),cov[:4,:4],atol=1e-10,rtol=1e-10)
# Direct distances against the complete retained PLUTO inventory.
p=pd.read_csv(R/'data/pluto_2016_relevant_fields.csv.gz');e=pd.read_csv(R/'data/neighborhood_exposures.csv.gz')
xy=p[['XCoord','YCoord']].to_numpy(float);areas=p[['BldgArea','ResArea','OfficeArea','RetailArea']].fillna(0).clip(lower=0).to_numpy(float)
ids=p.BBL.to_numpy();labels=['building','residential','office','retail']
for _,row in e.sample(10,random_state=9132026).iterrows():
 distances=np.sqrt(((xy-np.array([row.XCoord,row.YCoord]))**2).sum(axis=1))/(3937/1200)
 for radius in [250,500,1000]:
  exact=areas[(distances<=radius)&(ids!=row.bbl)].sum(axis=0)
  saved=np.array([row[f'{v}_{radius}'] for v in labels],float)
  np.testing.assert_allclose(exact,saved,atol=1e-6,rtol=1e-12)
for v in labels+['lot_count']:
 assert (e[f'{v}_250']<=e[f'{v}_500']).all()
 assert (e[f'{v}_500']<=e[f'{v}_1000']).all()
counts=pd.read_csv(R/'data/sales_source_counts.csv')
assert len(counts)==45 and counts.year_mismatch.sum()==0 and counts.dated_sales.sum()==757285
report={'full_dummy_OLS_and_CR1':'pass','direct_distance_focal_lots_checked':10,'nested_radii_focal_lots':len(e),'official_workbooks_checked':45,'dated_sales':757285,'calendar_year_mismatches':0,'note':'The constructed regression is a numerical unit check only. It supplies no empirical findings.'}
(R/'results/implementation_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
