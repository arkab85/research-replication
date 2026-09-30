from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
from core import *
P=Path(__file__).resolve().parents[1];R=P/'results';R.mkdir(exist_ok=True)
seed=20260913;rng=np.random.default_rng(seed)
rows=[]
for phi in [.2,.8]:
 for rx,ry in [(.2,.8),(.5,.5),(.8,.2)]:
  q=benchmark(phi,rx,ry)
  for delta in [0.,.1,.3,.6]:
   estimates=[]
   for j in range(10):
    x,y,c=sample(q,delta,2000,rng)
    f,b=pair(oracle(q,delta,x,y,c,140))
    estimates.append(unbiased(b[2],b[3])-unbiased(f[2],f[3]))
   rows.append({'phi':phi,'rx':rx,'ry':ry,'delta':delta,'dii':float(np.mean(estimates)),'mcse':float(np.std(estimates,ddof=1)/np.sqrt(10)),'K_b':envelope_constant(q),'outcome_bound':4*amplitude(q,0,delta)**2,'budget_bound':envelope_constant(q)*delta**2})
   print('population',phi,rx,ry,delta,rows[-1]['dii'],flush=True)
pd.DataFrame(rows).to_csv(R/'population.csv',index=False)
# Oracle confidence coverage for actual no-channel models, including double independence.
rng=np.random.default_rng(seed)
rows=[]
for n in [400,1200]:
 for rx,ry,delta in [(.3,.7,0.),(.2,.8,.02),(.8,.2,.02)]:
  q=benchmark(.8,rx,ry);K=envelope_constant(q)
  for rep in range(120):
   x,y,c=sample(q,delta,n,rng,serial=True);f,b=pair(oracle(q,delta,x,y,c,140))
   rad=radius([f[1],b[1]],ell=8 if n==400 else 12,B=399,seed=seed+rep)
   lo,hi=interval(f[0],b[0],rad)
   rows.append({'n':n,'rx':rx,'ry':ry,'delta':delta,'rep':rep,'dii':b[0]-f[0],'lower':lo,'upper':hi,'gamma_lower':np.sqrt(max(0,lo)/K),'false_exclusion':lo>K*delta**2,'radius':rad})
  print('inference',n,rx,ry,delta,flush=True)
pd.DataFrame(rows).to_csv(R/'inference_raw.csv',index=False)
pd.DataFrame(rows).groupby(['n','rx','ry','delta']).agg(false_exclusion=('false_exclusion','mean'),positive_lower=('lower',lambda s:float((s>0).mean())),mean_gamma=('gamma_lower','mean'),mean_radius=('radius','mean')).reset_index().to_csv(R/'inference.csv',index=False)
# Quadrature convergence, using the same observations at increasing integration orders.
audit=[]
for rx,ry in [(.2,.8),(.8,.2)]:
 q=benchmark(.8,rx,ry);x,y,c=sample(q,.6,2000,rng)
 prev=None
 for nodes in [80,140,240]:
  comp=oracle(q,.6,x,y,c,nodes);f,b=pair(comp)
  audit.append({'rx':rx,'ry':ry,'nodes':nodes,'dii':b[0]-f[0],'max_reverse_residual_change':None if prev is None else float(np.max(np.abs(comp[2]-prev)))})
  prev=comp[2]
(R/'quadrature.json').write_text(json.dumps(audit,indent=2))
print('complete',flush=True)
