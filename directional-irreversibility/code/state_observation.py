"""DII state-observation population calculations and paired estimation experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import roots_hermitenorm
P=Path(__file__).resolve().parent.parent

def kernel(x):
 x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
 d=((x[:,None,:]-x[None,:,:])**2).sum(2)
 return np.exp(-d/2)
def hsic(u,z,w=None):
 K=kernel(u);L=kernel(z);n=len(u)
 w=np.full(n,1/n) if w is None else w
 kw=K@w;lw=L@w
 return float(w@((K*L)@w)+(w@kw)*(w@lw)-2*w@(kw*lw))
def statistic(x,y,r,alpha,v,w=None):
 hf=hsic(y-alpha*r*x,np.column_stack([x,r]),w)
 hb=hsic(x-alpha*r*np.tanh(y/v),np.column_stack([y,r]),w)
 return hf,hb,hb-hf

def population(sigma,q,order):
 z,w=roots_hermitenorm(order);w=w/np.sqrt(2*np.pi)
 blocks=[];weights=[]
 for x in [-1.,1.]:
  for s in [-1.,1.]:
   for b,pb in [(1.,1-q),(-1.,q)]:
    if pb==0:continue
    blocks.append(np.column_stack([np.full(order,x),s*x+sigma*z,np.full(order,s*b)]));weights.append(w*pb/4)
 a=np.concatenate(blocks);ww=np.concatenate(weights)
 return statistic(*a.T,1-2*q,sigma**2,ww)

def main():
 import sys
 pop=[]
 for sigma in [1.,2.]:
  for q in [0.,.1,.25,.5]:
   vals=[population(sigma,q,k) for k in [80,160,240]]
   err=max(abs(np.array(vals[-1])-np.array(vals[-2])))
   assert err<2e-6,(sigma,q,vals)
   hf,hb,d=vals[-1];delta=8*q*(1-q)+2*q*np.sqrt(2*(1-np.exp(-2)))
   pop.append(dict(sigma=sigma,q=q,forward_hsic=hf,reverse_hsic=hb,dii=d,quadrature_change=err,operator_perturbation_bound=delta))
   if q==0:assert abs(hf)<2e-10 and d>0
   if q==.5:assert max(abs(hf),abs(hb))<2e-6
   print('Population',sigma,q,hf,hb,d,flush=True)
 for row in pop:
  h0=next(x['reverse_hsic'] for x in pop if x['sigma']==row['sigma'] and x['q']==0)
  delta=row['operator_perturbation_bound']
  row['dii_lower_bound']=max(0,np.sqrt(h0)-delta)**2-delta**2
  assert row['dii']>=row['dii_lower_bound']-2e-6
 (P/'state_observation_population.json').write_text(json.dumps(pop,indent=2)+'\n')
 if '--population-only' in sys.argv:return
 rng=np.random.default_rng(20260923);rows=[];reps=250
 for sigma in [1.,2.]:
  for n in [120,480]:
   m=2*n
   for rep in range(reps):
    x=rng.choice([-1.,1.],m+n);s=rng.choice([-1.,1.],m+n);y=s*x+rng.normal(scale=sigma,size=m+n);v=rng.random(m+n)
    for q in [0.,.1,.25,.5]:
     r=s*np.where(v<q,-1.,1.)
     alpha=float(np.clip(np.mean(r[:m]*x[:m]*y[:m]),0,1));var=float(np.clip(np.mean(y[:m]**2)-1,.1,10))
     oracle=statistic(x[m:],y[m:],r[m:],1-2*q,sigma**2)
     fitted=statistic(x[m:],y[m:],r[m:],alpha,var)
     rows.append(dict(sigma=sigma,n=n,m=m,q=q,rep=rep,alpha_hat=alpha,variance_hat=var,oracle_hf=oracle[0],oracle_hb=oracle[1],oracle_dii=oracle[2],fitted_hf=fitted[0],fitted_hb=fitted[1],fitted_dii=fitted[2]))
   print('Simulation',sigma,n,'complete',flush=True)
   pd.DataFrame(rows).to_csv(P/'state_observation_raw.csv',index=False)
 df=pd.DataFrame(rows);summary=[]
 for (sigma,n,q),g in df.groupby(['sigma','n','q']):
  target=next(z['dii'] for z in pop if z['sigma']==sigma and z['q']==q)
  r=dict(sigma=sigma,n=int(n),m=int(2*n),q=q,replications=reps,population_dii=target)
  for method in ['oracle','fitted']:
   a=g[method+'_dii'].to_numpy();sq=(a-target)**2;rmse=np.sqrt(sq.mean())
   r[method]=dict(mean=float(a.mean()),mean_mcse=float(a.std(ddof=1)/np.sqrt(reps)),rmse=float(rmse),rmse_mcse=float(sq.std(ddof=1)/(2*rmse*np.sqrt(reps))))
  r['fitted_oracle_rms']=float(np.sqrt(np.mean((g.fitted_dii-g.oracle_dii)**2)))
  summary.append(r)
 (P/'state_observation_results.json').write_text(json.dumps(dict(seed=20260923,independent_datasets=1000,paired_design_evaluations=4000,summary=summary),indent=2)+'\n')
 print('Saved all cells',flush=True)
if __name__=='__main__':main()
