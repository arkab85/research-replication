from pathlib import Path
import numpy as np,pandas as pd
from core import *
P=Path(__file__).resolve().parents[1];rng=np.random.default_rng(20260914);rows=[]
K=envelope_constant(benchmark(.8,.3,.7))
for n in [400,1200]:
 for rep in range(120):
  xt=rng.normal(size=n+1);x=xt[1:];c=xt[:-1];e=.3*rng.normal(size=n);y=(x*x-1)/np.sqrt(2)+e
  f,b=pair((e,np.column_stack([x,c]),x,np.column_stack([y,c])))
  rad=radius([f[1],b[1]],ell=8 if n==400 else 12,B=399,seed=20260914+rep)
  lo,hi=interval(f[0],b[0],rad)
  for gamma in [0.,.002,.005,.01]:
   rows.append({'n':n,'rep':rep,'gamma':gamma,'K':K,'dii':b[0]-f[0],'lower':lo,'upper':hi,'rejected':lo>K*gamma**2})
 print('power',n,flush=True)
pd.DataFrame(rows).to_csv(P/'results/power_raw.csv',index=False)
pd.DataFrame(rows).groupby(['n','gamma']).agg(rejection_rate=('rejected','mean'),mean_dii=('dii','mean'),mean_lower=('lower','mean')).reset_index().to_csv(P/'results/power.csv',index=False)
