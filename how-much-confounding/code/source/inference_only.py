from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from core import *
P=Path(__file__).resolve().parents[1];R=P/'results';seed=20260913;rng=np.random.default_rng(seed)
# Oracle confidence coverage for actual no-channel models, including double independence.
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
