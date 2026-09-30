"""Execute the fixed retrospective financial protocol; never select by significance."""
from pathlib import Path
import numpy as np,pandas as pd,json,itertools,hashlib
from dii_core import dii_profile
P=Path(__file__).resolve().parent.parent/'finance';D=P/'data'
H=(1,3,6);TRAIN=pd.period_range('1991-01','2007-12',freq='M');TEST=pd.period_range('2009-01','2018-12',freq='M')
def fred(name):
 f=pd.read_csv(D/(name+'.csv'));f.iloc[:,1]=pd.to_numeric(f.iloc[:,1],errors='coerce')
 s=pd.Series(f.iloc[:,1].to_numpy(dtype=float),index=pd.to_datetime(f.iloc[:,0])).dropna()
 return s.groupby(s.index.to_period('M')).last()
f=pd.read_csv(D/'jk_monthly.csv');mp=pd.Series(f.MP_median.to_numpy(),index=pd.PeriodIndex.from_fields(year=f.year,month=f.month,freq='M'))
assert not mp.index.duplicated().any()
base=pd.DataFrame({'MP':mp,'FX':100*np.log(fred('DEXJPUS')).diff(),'Treasury':fred('GS10').diff(),'Credit':(fred('BAA')-fred('AAA')).diff(),'VIX':100*np.log(fred('VIXCLS')).diff()}).loc['1990':'2019']
# lineterminator='\n': to_csv defaults to os.linesep. On Windows that made
# both this digest and the analysis_panel_local.csv written below use CRLF,
# breaking this assert and the five downstream scripts that re-hash that file.
panel_bytes=base.to_csv(index_label='month',lineterminator='\n').encode()
assert hashlib.sha256(panel_bytes).hexdigest()==(P/'analysis_panel_sha256.txt').read_text().strip(), 'Historical analysis data differ from frozen vintage; do not combine with fixed results.'
(P/'analysis_panel_local.csv').write_bytes(panel_bytes)
def scale(x,tr):
 mu=x[tr].mean(axis=0);sd=x[tr].std(axis=0,ddof=0);assert np.all(sd>1e-12)
 return (x-mu)/sd,mu,sd

def poly(z,degree):
 return np.column_stack([np.ones(len(z))]+[np.prod(z[:,ix],axis=1) for deg in range(1,degree+1) for ix in itertools.combinations_with_replacement(range(z.shape[1]),deg)])
def fit_predict(X,y,tr,te):
 b=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]
 return X[te]@b,b
results={};predrows=[];rows=[];diagnostics=[]
for asset in ['FX','Treasury','Credit','VIX']:
 comps={2:{},3:{}}
 for h in H:
  f=pd.DataFrame({'x':base.MP,'y':base[asset].shift(-h),'c1':base.MP.shift(1),'c2':base[asset].shift(1)})
  index=TRAIN.append(TEST);f=f.reindex(index)
  assert f.notna().all().all(),f'{asset} h={h} contains missing rows'
  tr=np.arange(len(TRAIN));te=np.arange(len(TRAIN),len(f));arr=f.to_numpy()
  z,mu,sd=scale(arr,tr);x,y=z[:,0],z[:,1];C=z[:,2:]
  for degree in (2,3):
   zf=np.column_stack([x,C]);zb=np.column_stack([y,C])
   pf,bf=fit_predict(poly(zf,degree),y,tr,te);pb,bb=fit_predict(poly(zb,degree),x,tr,te)
   comps[degree][f'h{h}']=(y[te]-pf,zf[te],x[te]-pb,zb[te])
   diagnostics.append({'asset':asset,'h':h,'degree':degree,'forward_train_condition':float(np.linalg.cond(poly(zf,degree)[tr])),'reverse_train_condition':float(np.linalg.cond(poly(zb,degree)[tr])),'train_n':len(tr),'eval_n':len(te)})
  X0=np.column_stack([np.ones(len(f)),C]);X1=np.column_stack([X0,x]);X2=np.column_stack([X1,x*x])
  pred0,_=fit_predict(X0,arr[:,1],tr,te);pred1,b1=fit_predict(X1,arr[:,1],tr,te);pred2,b2=fit_predict(X2,arr[:,1],tr,te)
  actual=arr[te,1];err0=(actual-pred0)**2;err1=(actual-pred1)**2;err2=(actual-pred2)**2
  rng=np.random.default_rng(20260913);ell=12;B=1999;starts=rng.integers(0,len(te),size=(B,len(te)//ell));idx=(starts[:,:,None]+np.arange(ell)[None,None,:])%len(te);idx=idx.reshape(B,-1)
  gains=100*(1-err2[idx].mean(axis=1)/err1[idx].mean(axis=1));lo,hi=np.quantile(gains,[.025,.975])
  predrows.append({'asset':asset,'h':h,'mse_history':float(err0.mean()),'mse_linear':float(err1.mean()),'mse_quadratic':float(err2.mean()),'quadratic_gain_percent':float(100*(1-err2.mean()/err1.mean())),'gain_ci_low':float(lo),'gain_ci_high':float(hi),'linear_train_slope_per_shock_sd':float(b1[-1]),'quadratic_train_term':float(b2[-1]),'shock_train_sd':float(sd[0])})
 for label,degree,ell in [('primary',2,12),('block6',2,6),('cubic',3,12)]:
  r=dii_profile(comps[degree],ell,999,20260913,contrasts={'average':[1/3]*3,'hump':[-.5,1,-.5]})
  results[asset+'_'+label]=r
  for h in H:rows.append({'asset':asset,'h':h,'specification':label,**r['comparisons'][f'h{h}']})
  print(asset,label,{h:round(r['comparisons'][f'h{h}']['p_intersection'],3) for h in H},flush=True)
primary=[r for r in rows if r['specification']=='primary'];order=sorted(range(len(primary)),key=lambda i:primary[i]['p_intersection']);prev=0
for rank,i in enumerate(order):
 prev=max(prev,(len(primary)-rank)*primary[i]['p_intersection']);primary[i]['p_holm']=min(1.,prev)
# Descriptive serial-dependence diagnostics cannot prove finite dependence.
from scipy.stats import chi2
acf=[]
for name in base.columns:
 a=base[name].reindex(TEST).to_numpy();a=a-a.mean();rs=[float(a[k:]@a[:-k]/(a@a)) for k in range(1,13)];Q=len(a)*(len(a)+2)*sum(rs[k-1]**2/(len(a)-k) for k in range(1,13))
 acf.append({'series':name,'acf1':rs[0],'acf12':rs[-1],'ljung_box_12_p':float(chi2.sf(Q,12))})
out={'protocol_sha256':hashlib.sha256((P/'analysis_protocol.md').read_bytes()).hexdigest(),'training_origin_start':str(TRAIN[0]),'training_origin_end':str(TRAIN[-1]),'evaluation_origin_start':str(TEST[0]),'evaluation_origin_end':str(TEST[-1]),'training_n':len(TRAIN),'evaluation_n':len(TEST),'primary_tests':primary,'all_specifications':rows,'predictive_comparisons':predrows,'joint_results':results,'fit_diagnostics':diagnostics,'dependence_diagnostics':acf,'interpretation':'Retrospective observed-proxy application. Bootstrap significance is conditional on unverified working assumptions; not automatically certified by finite-memory theory. No real-time or causal claim.'}
(P/'financial_results.json').write_text(json.dumps(out,indent=2));pd.DataFrame(rows).to_csv(P/'dii_all_results.csv',index=False);pd.DataFrame(predrows).to_csv(P/'prediction_results.csv',index=False)
print('Primary Holm rejections:',sum(r['p_holm']<=.05 for r in primary));print(pd.DataFrame(predrows)[['asset','h','quadratic_gain_percent','gain_ci_low','gain_ci_high']].to_string(index=False))
