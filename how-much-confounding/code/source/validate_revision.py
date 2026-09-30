"""Substantive numerical identities and stored-output consistency."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from revision_study import population, sharpness_coefficient, cell
P=Path(__file__).resolve().parents[1];R=P/'results';out={}
c=sharpness_coefficient();assert abs(c-.8**4/54)<1e-14
small=population(.8,.005,1.,180)
assert small[2]>0 and abs(small[2]/.005**2/c-1)<1e-4
out['sharpness_coefficient']=c;out['small_loading_scaled_contrast']=small[2]/.005**2
out['quadrature_difference_180_500']=abs(population(1.,1.,.5,180)[2]-population(1.,1.,.5,500)[2])
assert out['quadrature_difference_180_500']<2e-10
rng=np.random.default_rng(66231)
A,B,q,r,threshold=rng.uniform(0,1,(5,20000))
lo=np.maximum(B-q-2*r,0)**2-np.minimum(A+q+2*r,1)**2
crit=(B-A-2*q-threshold/(A+B))/4
assert np.array_equal(lo>threshold,r<crit)
out['frontier_identity_scenarios']=len(A)
raw=pd.read_csv(R/'revision_mc_raw.csv');tr=pd.read_csv(R/'revision_training_raw.csv')
assert len(raw)==4800 and len(tr)==800
assert raw.groupby(['model','phi','n','degree']).size().eq(200).all()
assert tr.groupby(['phi','n_train']).size().eq(200).all()
assert raw.cover_aware.all() and (raw.rv_aware==0).all()
out['new_replications']=len(raw)+len(tr)
fields=['D_true','D_hat','Hf','Hb','radius','r_f','r_b','L_aware','U_aware'];gaps={}
for model in ['confounding','quadratic','linear']:
 old=raw.query('model==@model and phi==0.6 and n==250 and degree==2 and rep==0').iloc[0]
 new=cell((model,.6,250,2,1,399))[0]
 gap=max(abs(float(old[f])-float(new[f])) for f in fields);assert gap<1e-12;gaps[model]=gap
out['representative_reproduction_max_gaps']=gaps
cal=pd.read_csv(R/'revision_calendar.csv');strata=pd.read_csv(R/'revision_strata_components.csv')
base=pd.read_csv(R/'pooled_application.csv')
assert len(cal)==48 and (cal.rv==0).all()
pool_gaps={}
for outcome,group in strata.groupby('outcome'):
 weighted=np.average(group.D,weights=group.n)
 direct=cal.query('outcome==@outcome and pooling=="direct_sum"').D.iloc[0]
 gap=abs(weighted-direct);assert gap<1e-12;pool_gaps[outcome]=gap
 archived=base.query('design==@outcome and ell==4 and r==0').dii.iloc[0]
 avg=cal.query('outcome==@outcome and pooling=="average"').D.iloc[0]
 assert abs(avg-archived)<1e-12
for _,group in cal.groupby(['outcome','pooling','calendar_months']):
 group=group.sort_values('r')
 assert (np.diff(group.lower)<=1e-14).all() and (np.diff(group.upper)>=-1e-14).all()
out['direct_sum_identity_max_gaps']=pool_gaps;out['calendar_scenarios']=len(cal)
assert hashlib.sha256((P/'data/analysis_panel.csv').read_bytes()).hexdigest()==(P/'data/panel_sha256.txt').read_text().strip()
checked=0
for item in json.loads((P/'audit/original_submission/manifest.json').read_text()):
 path=P/item['path']
 if item['path'].startswith('data/') and path.exists():
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];checked+=1
out['frozen_data_files_hash_checked']=checked;out['passed']=True
out['scope']='Computational checks; not independent peer review of every theorem.'
(P/'audit/validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
