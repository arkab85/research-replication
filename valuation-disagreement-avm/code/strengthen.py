"""Reproduce focal regressions and fixed-specification additions from licensed FA_Luxury.csv.
County-blocked validation is archived-state classification, not prospective prediction.
Usage: python strengthen.py input.csv [county_history.csv | public_county_2018.csv] [--regression-only | --validation-only]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,warnings
warnings.filterwarnings('ignore')
import pandas as pd,numpy as np
VALIDATE_ONLY="--validation-only" in sys.argv
if not VALIDATE_ONLY: import pyfixest as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss,roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;src=Path(sys.argv[1]);d=pd.read_csv(src,dtype=str,low_memory=False);d.columns=d.columns.str.strip()
num=lambda s:pd.to_numeric(s.str.replace(r'[$,\s]','',regex=True),errors='coerce')
for c in ['CurrentListingPrice','FACurrentAVM','Low_Value','High_Value','HomeSize','LotSizeSqFt','YearBuilt','ConfidenceScore','DOM']:d[c]=num(d[c])
w=lambda x:x.clip(x.quantile(.01),x.quantile(.99))
P,A=d.CurrentListingPrice,d.FACurrentAVM;lr=np.log(P/A)
d['y']=d.Status.isin(['Pending','Contingent']).astype(int);d['gap']=lr;d['absd']=w(lr.abs());d['prem']=w(lr.clip(lower=0));d['disc']=w((-lr).clip(lower=0));d['conf10']=d.ConfidenceScore/10;d['hiconf']=(d.ConfidenceScore>=80).astype(int)
d['lp']=w(np.log(P))
for v,c in [('lsq','HomeSize'),('llot','LotSizeSqFt')]:
 x=np.log(d[c].where(d[c]>0));d[v+'_m']=x.isna().astype(int);d[v]=w(x).fillna(0)
for year in [2019,2020]:
 age=year-d.YearBuilt;d['age_m']=age.isna().astype(int);d['age'+str(year)]=w(age).fillna(0)
d['age']=d.age2019;d['city']=d['CITY/STATE'].fillna('NA');d['ptype']=d.PropertyType.fillna('NA');d['zip']=d.PropertyZip.fillna('NA');d['county']=d.FIPS.str.zfill(5);d['state']=d.PropertyState
ld=pd.to_datetime(d.ListingDate,errors='coerce');up=pd.to_datetime(d.FA_UpdateTimeStamp,errors='coerce');av=pd.to_datetime(d.ValuationDate,errors='coerce');sold=pd.to_datetime(d.SoldDate,errors='coerce');ref=pd.Timestamp('2020-02-04')
d['month']=ld.dt.to_period('M').astype(str);d['cq']=d.county+'_'+ld.dt.to_period('Q').astype(str);d['recency']=(ref-up).dt.days;d['update_bin']=pd.cut(d.recency,[-1,30,60,90,180,365,np.inf]).astype(str)
d['listing_age']=(ref-ld).dt.days;d['avm_lag']=(ref-av).dt.days;d['logwidth']=np.log(d.High_Value/d.Low_Value);d['plaus']=(A/P).between(.5,2)
R={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n':len(d),'status_counts':d.Status.value_counts().to_dict(),'unique_property_ids':d.FA_PropertyID.nunique(),'unique_tracking_ids':d.ListingTrackingID.nunique(),'nonnull_sold_dates':int(sold.notna().sum()),'dom_identity_rows':int((d.DOM==d.listing_age).sum()),'estimates':{}}
C='lp + lsq + llot + age + lsq_m + llot_m + age_m';FE='city + ptype + month';pl=d[d.plaus & d.ConfidenceScore.notna()].copy()
def est(name,terms,x,fe=FE,controls=C,cluster='city'):
 if VALIDATE_ONLY:return
 m=pf.feols('y ~ '+terms+' + '+controls+' | '+fe,data=x,vcov={'CRV1':cluster},fixef_rm='none',fixef_maxiter=100000);t=m.tidy();R['estimates'][name]={'n':int(m._N),'coef':{k:float(v) for k,v in t.Estimate.items()},'se':{k:float(v) for k,v in t['Std. Error'].items()}};print(name,m._N,flush=True)
est('absolute','absd + conf10',d)
est('premium_discount','prem + disc + conf10',pl)
est('low_confidence','prem + disc + conf10',pl[pl.hiconf==0]);est('high_confidence','prem + disc + conf10',pl[pl.hiconf==1])
terms='prem + disc + prem:hiconf + disc:hiconf + hiconf'
est('interactions',terms,pl)
# Observed atypicality proxies, computed without status outcomes.
for v in ['lsq','llot','age2020']:
 med=d.groupby('county')[v].transform('median');dev=(d[v]-med).abs();mad=dev.groupby(d.county).transform('median');d['atyp_'+v]=(dev/mad.where(mad>0)).clip(upper=10).fillna(0)
cts=d.groupby(['county','ptype']).y.transform('size');tot=d.groupby('county').y.transform('size');d['rarity']=-np.log(cts/tot)
d['lsq2']=d.lsq**2;d['llot2']=d.llot**2;d['age2']=d.age2020**2
pl=d[d.plaus & d.ConfidenceScore.notna()].copy();C2=C.replace('age +','age2020 +')+' + lsq2 + llot2 + age2 + atyp_lsq + atyp_llot + atyp_age2020 + rarity'
est('physical_atypicality',terms,pl,controls=C2)
est('county_cohort',terms,pl,fe='cq + ptype',controls=C2,cluster='county')
est('update_conditioned',terms,pl,fe='city + ptype + month + update_bin',controls=C2)
est('exclude_sold_dates',terms,pl.loc[sold.loc[pl.index].isna()],controls=C2)
# 2018 public county conditions, complete calendar retained; at least 9 valid months.
public=[]
if len(sys.argv)>2 and not sys.argv[2].startswith('--') and 'county_median_listing_price' in pd.read_csv(sys.argv[2],nrows=0).columns:
 # Precomputed 2018 county means (public_county_2018.csv), written by an earlier run on the full vintage.
 out=pd.read_csv(sys.argv[2],dtype={'county':str}).set_index('county');public=list(out.columns)
 d=d.join(out,on='county');R['public_county_match_rows']=int(d[public].notna().all(axis=1).sum());R['public_county_means_sha256']=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
elif len(sys.argv)>2 and not sys.argv[2].startswith('--'):
 c=pd.read_csv(sys.argv[2],dtype={'county_fips':str},low_memory=False);c=c[c.month_date_yyyymm.between(201801,201812)&c.quality_flag.eq(0)].copy();c['county']=c.county_fips.str.zfill(5)
 for k in ['median_listing_price','median_days_on_market','price_reduced_share','new_listing_count','active_listing_count']:c[k]=pd.to_numeric(c[k],errors='coerce')
 c['new_active']=c.new_listing_count/c.active_listing_count.where(c.active_listing_count>0)
 cols=['median_listing_price','median_days_on_market','price_reduced_share','new_active'];agg=c.groupby('county')[cols].agg(['mean','count']);out=pd.DataFrame(index=agg.index)
 for v in cols:
  nm='county_'+v;out[nm]=agg[(v,'mean')].where(agg[(v,'count')]>=9);public.append(nm)
 out.to_csv(ROOT/'public_county_2018.csv');d=d.join(out,on='county');R['public_county_match_rows']=int(d[public].notna().all(axis=1).sum());R['public_county_source_sha256']=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if not VALIDATE_ONLY:
 (ROOT/'regression_results.json').write_text(json.dumps(R,indent=2,default=int))
 if '--regression-only' in sys.argv: sys.exit(0)
else:
 R=json.loads((ROOT/'regression_results.json').read_text())
# Four fixed models; same rows, folds, tuning. No age filtering for best performance.
s=d[d.ConfidenceScore.notna() & np.isfinite(d.gap)&np.isfinite(d.logwidth)].copy();s=s.reset_index(drop=True)
X=pd.DataFrame({'logprice':np.log(s.CurrentListingPrice),'loghome':np.log(s.HomeSize.where(s.HomeSize>0)),'loglot':np.log(s.LotSizeSqFt.where(s.LotSizeSqFt>0)),'age':s.age2020,'listing_age':s.listing_age,'lat':pd.to_numeric(s.SitusLatitude,errors='coerce'),'lon':pd.to_numeric(s.SitusLongitude,errors='coerce')})
# State and broad property type use deterministic encodings; rare types grouped without Y.
for k in ['state','ptype']:
 z=s[k];z=z.where(z.map(z.value_counts())>=100,'OTHER');X[k]=z.astype('category').cat.codes
for v in public:X[v]=s[v]
base=list(X);X['gap']=s.gap;X['confidence']=s.conf10;X['logwidth']=s.logwidth;X['report_age']=s.recency;X['avm_age']=s.avm_lag
sets={'property':base,'gap':base+['gap'],'metadata':base+['confidence','logwidth'],'full':base+['gap','confidence','logwidth'],'gap_recency':base+['gap','report_age','avm_age'],'full_recency':base+['gap','confidence','logwidth','report_age','avm_age']}
Y=s.y.to_numpy();g=s.county.to_numpy();pred={k:np.zeros(len(s)) for k in sets};folds=np.zeros(len(s),int)
pool_limit=threadpool_limits(limits=2)
for fold,(tr,te) in enumerate(GroupKFold(5).split(X,Y,g)):
 assert not set(g[tr])&set(g[te]);folds[te]=fold
 for name,cols in sets.items():
  cat=[cols.index(k) for k in ['state','ptype']]
  m=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=100,learning_rate=.05,l2_regularization=1,categorical_features=cat,early_stopping=False,random_state=71)
  print('fitting',fold,name,flush=True);m.fit(X.iloc[tr][cols],Y[tr]);pred[name][te]=m.predict_proba(X.iloc[te][cols])[:,1]
 print('fold',fold,'done',flush=True)
R['validation']={'n':len(s),'counties':s.county.nunique(),'positive_share':float(Y.mean()),'model_metrics':{}}
for k,p in pred.items():R['validation']['model_metrics'][k]={'logloss':log_loss(Y,p),'brier':brier_score_loss(Y,p),'auc':roc_auc_score(Y,p),'average_precision':average_precision_score(Y,p)}
# County cluster bootstrap of loss differences; weighted by held-out record count.
B=pd.DataFrame({'county':g,'n':1});rng=np.random.default_rng(31217)
for k,p in pred.items():p=np.clip(p,1e-7,1-1e-7);B[k]=-(Y*np.log(p)+(1-Y)*np.log(1-p))
A=B.groupby('county').sum();R['validation']['contrasts']={}
for x,y in [('gap','full'),('metadata','full'),('property','full'),('gap_recency','full_recency')]:
 b=[]
 for _ in range(2000):
  z=A.iloc[rng.integers(0,len(A),len(A))];b.append(100*(1-z[y].sum()/z[x].sum()))
 R['validation']['contrasts'][x+'_to_'+y]={'logloss_reduction_pct':100*(1-A[y].sum()/A[x].sum()),'ci95':np.quantile(b,[.025,.975]).tolist()}
# Restricted-release output has no property addresses/IDs, but county predictions remain private.
# Save aggregate fold losses only in distributable package.
metrics=[]
for f in range(5):
 for name,p in pred.items():metrics.append({'fold':f,'model':name,'n':int((folds==f).sum()),'logloss':log_loss(Y[folds==f],p[folds==f])})
pd.DataFrame(metrics).to_csv(ROOT/'validation_folds.csv',index=False)
(ROOT/'verified_results.json').write_text(json.dumps(R,indent=2,default=int));print(json.dumps(R['validation'],indent=2),flush=True)
