"""Rate check for the quadratic envelope: population components as the loading shrinks."""
from pathlib import Path
import numpy as np,pandas as pd
from core import *
P=Path(__file__).resolve().parents[1];rng=np.random.default_rng(20260921);rows=[]
for phi,rx,ry in [(.2,.8,.2),(.8,.2,.8)]:
    q=benchmark(phi,rx,ry)
    for delta in [.1,.2,.4,.8]:
        hf,hb=[],[]
        for j in range(20):
            x,y,c=sample(q,delta,2000,rng);f,b=pair(oracle(q,delta,x,y,c,140))
            hf.append(unbiased(f[2],f[3]));hb.append(unbiased(b[2],b[3]))
        rows.append({'phi':phi,'rx':rx,'ry':ry,'delta':delta,'Hf':np.mean(hf),'Hf_se':np.std(hf,ddof=1)/np.sqrt(20),'Hb':np.mean(hb),'Hb_se':np.std(hb,ddof=1)/np.sqrt(20),'K_b':envelope_constant(q)})
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(P/'results/rate_study.csv',index=False)
