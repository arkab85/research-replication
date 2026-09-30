"""Dated-protocol vacancy estimates and reporting diagnostics, including all checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from estimate import fit_cluster,infer
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'data/storefront_panel_all.csv.gz',dtype={'zip_fixed':str})

def valid(s,rad):
 ok=s[f'building_{rad}'].gt(0)&s[f'coverage_{rad}'].gt(0)&s.reported.gt(0)
 for c in ['residential','office','retail']:ok &= s[f'{c}_share_{rad}'].between(0,1)
 return s[ok].copy()

def panel(s,years,balanced=True):
 s=s[s.year.isin(years)].copy()
 g=s.groupby('bbl').year.agg(['min','nunique'])
 ids=g[(g['min']==2019)&(g['nunique']==len(years) if balanced else g['nunique'].ge(2))].index
 return s[s.bbl.isin(ids)].reset_index(drop=True)

def design(s,rad=500,event=False,zip_year=False,evenness=False):
 x=pd.DataFrame(index=s.index)
 for lab,c in [('R','residential'),('O','office'),('K','retail')]:x[lab]=s[f'{c}_share_{rad}']
 x['density']=np.log(s[f'building_{rad}']/(np.pi*rad**2))
 if evenness:
  sh=x[['R','O','K']].to_numpy();sh=sh/sh.sum(axis=1,keepdims=True)
  x['mix']=1.5*(1-(sh**2).sum(axis=1));cs=['mix','density']
 else:cs=['R','O','K','density']
 z=pd.DataFrame(index=s.index)
 for year in (sorted(s.year.unique())[1:] if event else ['post']):
  for c in cs:z[f'{c}_{year}']=x[c]*((s.year>2019) if year=='post' else (s.year==year))
 geog=s.zip_fixed if zip_year else s.BoroCode.astype(str)
 return pd.concat([z,pd.get_dummies(geog+'_'+s.year.astype(str),dtype=float,prefix='time')],axis=1)

rows=[];annual=[];fits={}
specs=['Primary','250m radius','1000m radius','Available years','Constant reported count','No reported construction','2019-2022 panel','ZIP by year','One registration each year','Three-use evenness','C D K O building classes']
for spec in specs:
 rad=250 if spec=='250m radius' else 1000 if spec=='1000m radius' else 500
 years=list(range(2019,2023 if spec=='2019-2022 panel' else 2025))
 s=panel(valid(d,rad),years,spec!='Available years')
 if spec=='Constant reported count':s=s[s.groupby('bbl').reported.transform('nunique')==1]
 if spec=='No reported construction':s=s[s.groupby('bbl').construction.transform('max')==0]
 if spec=='One registration each year':s=s[s.groupby('bbl').registered.transform('max')==1]
 if spec=='C D K O building classes':s=s[s.BldgClass.str[0].isin(['C','D','K','O'])]
 s=s.reset_index(drop=True)
 fit=fit_cluster(s.vacancy,design(s,rad,zip_year=spec=='ZIP by year',evenness=spec=='Three-use evenness'),s.bbl,s.CD)
 w={'mix_post':10} if spec=='Three-use evenness' else {'R_post':10,'O_post':-10}
 result={'spec':spec,'n':len(s),'parcels':s.bbl.nunique(),'clusters':fit['groups'],**infer(fit,w)}
 rows.append(result);print(result,flush=True)
 if spec=='Primary':
  fits['primary']={'beta':fit['beta'].to_dict(),'cov':fit['cov'].to_dict(),'dropped':fit['dropped']}
  for var in ['R','O','K']:rows.append({'spec':'Individual '+var,'n':len(s),'parcels':s.bbl.nunique(),'clusters':fit['groups'],**infer(fit,{var+'_post':10})})
  s.to_csv(R/'data/storefront_primary_sample.csv.gz',index=False)
  desc=s.groupby('year').agg(parcels=('bbl','size'),registrations=('registered','sum'),reported=('reported','sum'),vacant=('vacant','sum'),mean_parcel_vacancy=('vacancy','mean'),mean_completeness=('completeness','mean')).reset_index()
  desc.to_csv(R/'results/storefront_primary_description.csv',index=False)
  eventfit=fit_cluster(s.vacancy,design(s,event=True),s.bbl,s.CD)
  for year in sorted(s.year.unique()):
   res={'estimate':0.,'se':0.,'lower':0.,'upper':0.,'p':None} if year==2019 else infer(eventfit,{f'R_{year}':10,f'O_{year}':-10})
   annual.append({'year':int(year),**res})
  f=fit_cluster(s.completeness,design(s),s.bbl,s.CD)
  rows.append({'spec':'Reporting completeness','n':len(s),'parcels':s.bbl.nunique(),'clusters':f['groups'],**infer(f,{'R_post':10,'O_post':-10})})
  # Basic common-support audit, stated descriptively rather than as identification.
  base=s[s.year==2019]
  (R/'results/storefront_exposure_summary.json').write_text(json.dumps({'R':base.residential_share_500.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),'O':base.office_share_500.quantile([0,.1,.25,.5,.75,.9,1]).to_dict(),'at_least_10pct_office':float((base.office_share_500>=.1).mean())},indent=2))
pd.DataFrame(rows).to_csv(R/'results/storefront_estimates.csv',index=False)
pd.DataFrame(annual).to_csv(R/'results/storefront_annual.csv',index=False)
(R/'results/storefront_model.json').write_text(json.dumps(fits,indent=2))

# Economic equivalence margins are illustrative sensitivity benchmarks, selected after prior price results.
e=pd.read_csv(R/'results/estimates.csv');out=[]
for sector in ['Retail','Office']:
 x=e[(e.sector==sector)&(e.spec=='Primary')].iloc[0];df=x.clusters-1
 for margin in [.05,.10]:
  p=max(stats.t.sf((x.estimate+margin)/x.se,df),stats.t.cdf((x.estimate-margin)/x.se,df))
  out.append({'sector':sector,'margin':margin,'estimate':x.estimate,'ci90_lower':x.estimate-stats.t.ppf(.95,df)*x.se,'ci90_upper':x.estimate+stats.t.ppf(.95,df)*x.se,'tost_p':p})
pd.DataFrame(out).to_csv(R/'results/price_equivalence.csv',index=False)
