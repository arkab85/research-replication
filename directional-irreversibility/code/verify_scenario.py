"""Check the finite empirical embedding identity and analytic normal-weight constants."""
from pathlib import Path
import numpy as np,json
from scipy.integrate import quad
from math import exp,sqrt,pi
rng=np.random.default_rng(20260913);n=9;u=rng.normal(size=n);z=.6*u+rng.normal(size=n)
k=lambda x,y:np.exp(-.5*(np.asarray(x)[:,None]-np.asarray(y)[None,:])**2)
K=k(u,u);L=k(z,z);C=np.eye(n)-np.ones((n,n))/n
hs=float(np.sum((C@K@C)*(C@L@C))/n**2)
# Signed empirical joint minus full product law, using an explicit support Gram matrix.
U=np.r_[u,np.repeat(u,n)];Z=np.r_[z,np.tile(z,n)];a=np.r_[np.ones(n)/n,-np.ones(n*n)/(n*n)]
gram=k(U,U)*k(Z,Z);mmd=float(a@gram@a)
assert abs(hs-mmd)<1e-12
norm=sqrt(mmd);witness=float(a@gram@(a/norm));assert abs(witness-norm)<1e-12
rows=[]
for s in [0.,2.]:
 c=exp(-s*s/4)/sqrt(2);numeric=quad(lambda x:exp(-.5*(x-s)**2)*exp(-x*x/2)/sqrt(2*pi),-np.inf,np.inf)[0]
 assert abs(c-numeric)<1e-11
 rows.append(dict(s=s,weight_normalizer=c,weight_norm=1/c,epsilon=.05,hsic_threshold=(.05*c)**2))
out=dict(empirical_hsic=hs,empirical_mmd_squared=mmd,identity_error=abs(hs-mmd),witness_error=abs(witness-norm),analytic_scenarios=rows)
P=Path(__file__).resolve().parent.parent;(P/'scenario_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
