"""Population check of Proposition (transmission): forward component is zero,
backward component is positive, under additive-noise transmission.
Designs: cause persistence phi in {0, 0.8}; loading quadratic or cubic.
Y_{t+1}=f(X_t)+e_{t+1}, e~N(0,0.3^2), C_t=X_{t-1}. Unit bandwidths.
Unbiased HSIC U-statistics over 10 independent batches of 2,000 tuples."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import roots_hermitenorm,logsumexp
from core import gram,unbiased
P=Path(__file__).resolve().parents[1];rng=np.random.default_rng(20260919)
F={'quadratic':lambda x:(x*x-1)/np.sqrt(2),'cubic':lambda x:(x+x**3)/np.sqrt(22)}
se=.3;z,w=roots_hermitenorm(200);w=w/np.sqrt(2*np.pi);rows=[]
for phi in [0.,.8]:
 for name,f in F.items():
  hf,hb=[],[]
  for batch in range(10):
   n=2000;c=rng.normal(size=n);x=phi*c+np.sqrt(1-phi**2)*rng.normal(size=n)
   y=f(x)+se*rng.normal(size=n)
   ef=y-f(x)  # forward conditional mean is f(X_t) exactly
   mu=phi*c;sd=np.sqrt(1-phi**2);xb=mu[:,None]+sd*z[None,:]
   lw=np.log(w)[None,:]-.5*((y[:,None]-f(xb))/se)**2
   eb=x-np.sum(np.exp(lw-logsumexp(lw,axis=1)[:,None])*xb,axis=1)
   hf.append(unbiased(gram(ef),gram(np.column_stack([x,c]))))
   hb.append(unbiased(gram(eb),gram(np.column_stack([y,c]))))
  hf,hb=np.array(hf),np.array(hb);d=hb-hf
  rows.append({'phi':phi,'loading':name,'Hf':hf.mean(),'Hf_se':hf.std(ddof=1)/np.sqrt(10),
   'Hb':hb.mean(),'Hb_se':hb.std(ddof=1)/np.sqrt(10),'dii':d.mean(),'dii_se':d.std(ddof=1)/np.sqrt(10)})
  print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(P/'results/transmission.csv',index=False)
