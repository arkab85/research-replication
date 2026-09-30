"""Exact finite-support checks of the Gaussian-kernel predictability bound."""
from pathlib import Path
import numpy as np,json
A=np.array([-3.,-2.,-1.,1.,2.,3.]);A/=np.sqrt(np.mean(A*A));n=len(A);center=np.eye(n)-np.ones((n,n))/n
rows=[]
for phi in [0,.5,.9,.97,.999]:
 v=1-phi*phi;x=np.sqrt(v)*A;y=x*x
 for ell in [.5,1.,2.]:
  K=np.exp(-.5*((x[:,None]-x[None,:])/ell)**2);L=np.exp(-.5*(y[:,None]-y[None,:])**2)
  H=float(np.sum((center@K@center)*(center@L@center))/n**2)
  assert H<=v/(ell*ell)+1e-12
  rows.append(dict(innovation_variance=v,bandwidth=ell,reverse_hsic=H,upper_bound=v/(ell*ell)))
P=Path(__file__).resolve().parent.parent
(P/'innovation_bound_verification.json').write_text(json.dumps({'design':'Exact finite-support symmetric laws; numerical identity checks, not a power study','rows':rows},indent=2)+'\n')
print('Verified',len(rows),'finite-support/kernel configurations')
