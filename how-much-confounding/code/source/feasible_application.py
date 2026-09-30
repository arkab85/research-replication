"""Fixed-calendar preprocessing/training/evaluation with joint calendar multipliers.
All tuning choices below fixed before this analysis; this is a new sensitivity
analysis of the supplied frozen data, not an independent empirical replication.
"""
from pathlib import Path
import numpy as np,pandas as pd,json
from threadpoolctl import threadpool_limits
from pooled_application import load,ref_K,poly
from core import component,center,interval
from feasible_inference import derivative_geometry
P=Path(__file__).resolve().parents[1];R=P/'results'
def build_rows(S,idx,label,basis):
 design=lambda z:poly(z) if basis=="quadratic" else poly(np.tanh(z/2))
 archived=pd.read_csv(R/'pooled_strata.csv');rows=[];counts=[]
 for name,(d,cand,longest,eq,bond) in S.items():
  ycol=eq if idx==0 else bond
  old=archived[(archived.design==label)&(archived.stratum==name)].iloc[0]
  cols=old.instruments.split();dd=d.dropna(subset=cols+[ycol]).sort_values('t').reset_index(drop=True)
  pre=dd.t<'2016-01-01';train=(dd.t>='2016-01-01')&(dd.t<'2021-01-01');test=dd.t>='2021-01-01'
  nr=[int(m.sum()) for m in [pre,train,test]]
  counts.append(dict(outcome=label,stratum=name,n_pre=nr[0],n_train=nr[1],n_eval=nr[2],included=min(nr)>=20))
  if min(nr)<20:continue
  M=dd[cols].to_numpy(float);mu=M[pre].mean(0);sd=M[pre].std(0);keep=sd>0
  Z=(M[:,keep]-mu[keep])/sd[keep];used=[c for c,k in zip(cols,keep) if k]
  w=np.linalg.svd(Z[pre],full_matrices=False)[2][0];ref=next(c for c in longest if c in used)
  if w[used.index(ref)]<0:w=-w
  X=np.column_stack([Z@w,dd[ycol].to_numpy(float)]);X=(X-X[pre].mean(0))/X[pre].std(0)
  rho=float(np.corrcoef(X[pre].T)[0,1]);K=ref_K(rho)
  tr=X[train];te=X[test];comps=[];geos=[];scores=[]
  for k in [0,1]:
   pt=design(tr[:,k]);inv=np.linalg.inv(pt.T@pt);b=inv@pt.T@tr[:,1-k];err=tr[:,1-k]-pt@b
   h=np.einsum('ij,jk,ik->i',pt,inv,pt)
   scores.append((pt*(err/np.maximum(1-h,1e-8))[:,None])@inv[:,1:])
   er=te[:,1-k]-design(te[:,k])@b
   comps.append(component(er,te[:,k]));geos.append(derivative_geometry(er,te[:,k],features=design(te[:,k])[:,1:]))
  rows.append(dict(name=name,n=len(te),nt=len(tr),K=K,comps=comps,geos=geos,scores=scores,
                   train_dates=pd.DatetimeIndex(dd.t[train]),eval_dates=pd.DatetimeIndex(dd.t[test])))
 return rows,counts

def run(rows,label,months,basis):
 all_dates=np.concatenate([np.concatenate([r['train_dates'].values,r['eval_dates'].values]) for r in rows])
 ad=pd.DatetimeIndex(all_dates);labels=(ad.year*12+ad.month-1)//months;unique=np.unique(labels)
 B=1999;w=np.random.default_rng(20261103+months).normal(size=(B,len(unique)))
 def indices(d):return np.searchsorted(unique,(d.year*12+d.month-1)//months)
 n=sum(r['n'] for r in rows);pi=np.array([r['n']/n for r in rows]);K=sum(p*r['K'] for p,r in zip(pi,rows))
 hs=np.zeros(2);norms=np.zeros((2,B));norms0=np.zeros((2,B))
 for weight,r in zip(pi,rows):
  wi=w[:,indices(r['eval_dates'])];wt=w[:,indices(r['train_dates'])]
  # Shared weights preserve same-calendar dependence across banks and regressions.
  for k in [0,1]:
   hs[k]+=weight*r['comps'][k][0];G,C=r['geos'][k];v=wt@r['scores'][k]
   A=center(r['comps'][k][1]);samp=np.einsum('bi,ij,bj->b',wi,A,wi,optimize=True)/r['n']**2
   par=np.einsum('bi,ij,bj->b',v,G,v,optimize=True)
   cross=2*np.einsum('bi,ij,bj->b',wi,C,v,optimize=True)/r['n']
   norms[k]+=weight*(samp+par+cross);norms0[k]+=weight*samp
 q=float(np.quantile(np.sqrt(np.maximum(norms,0)).max(0),.95,method='higher'))
 q0=float(np.quantile(np.sqrt(np.maximum(norms0,0)).max(0),.95,method='higher'))
 out=[]
 for eta in [0.,.01,.025,.05]:
  lo,hi=interval(*hs,q,eta,eta);l0,u0=interval(*hs,q0,eta,eta)
  out.append(dict(outcome=label,basis=basis,calendar_months=months,strata=len(rows),n=n,n_train=sum(r['nt'] for r in rows),eta=eta,
                  Hf=hs[0],Hb=hs[1],D=hs[1]-hs[0],q=q,q_ignore=q0,K=K,lower=lo,upper=hi,rv=np.sqrt(max(lo,0)/K),
                  upper_certificate=np.sqrt(max(hi,0)/K),lower_ignore=l0,upper_ignore=u0))
 return out
if __name__=='__main__':
 threadpool_limits(limits=1);S=load();out=[];counts=[]
 for idx,label in [(0,'Equity index'),(1,'10-year bond yield')]:
  for basis in ['quadratic','bounded']:
   rows,c=build_rows(S,idx,label,basis)
   if basis=='quadratic':counts+=c
   for months in [1,3,6]:out+=run(rows,label,months,basis)
   print(label,basis,'strata',len(rows),'eval',sum(r['n'] for r in rows),flush=True)
 pd.DataFrame(counts).to_csv(R/'feasible_application_counts.csv',index=False)
 d=pd.DataFrame(out);d.to_csv(R/'feasible_application.csv',index=False)
 (R/'feasible_application_metadata.json').write_text(json.dumps({'preprocess_through':'2015-12-31','regression_training':'2016-2020','evaluation_from':'2021-01-01','instrument_sets':'frozen original analysis sets','minimum_each_period':20,'bootstrap_draws':1999,'calendar_months':[1,3,6],'design_status':'quadratic analysis followed by disclosed bounded-basis stress check; not preregistered','qualification':'model approximation and calendar sampling assumptions remain maintained'},indent=2))
 print(d.query('calendar_months==3').to_string(index=False),flush=True)
