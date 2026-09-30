"""Independent posterior and operator-coupling checks for the state theorem."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm
P=Path(__file__).resolve().parent.parent
posterior_checks=0;max_error=0
for sig in [1.,2.]:
 for q in [0,.001,.1,.25,.5]:
  for r in [-1,1]:
   for y in np.linspace(-5,5,31):
    weights=[];xs=[]
    for x in [-1,1]:
     for s in [-1,1]:
      weights.append(.25*((1-q) if r==s else q)*norm.pdf(y,loc=s*x,scale=sig));xs.append(x)
    exact=np.dot(weights,xs)/sum(weights);formula=(1-2*q)*r*np.tanh(y/sig**2)
    max_error=max(max_error,abs(exact-formula));posterior_checks+=1
assert max_error<1e-12

def cross(a,b):
 a=np.asarray(a);b=np.asarray(b)
 if a.ndim==1:a=a[:,None];b=b[:,None]
 return np.exp(-((a[:,None,:]-b[None,:,:])**2).sum(2)/2)
def cent(A):return A-A.mean(0)-A.mean(1)[:,None]+A.mean()
def inner(u,z,v,w):return np.mean(cent(cross(u,v))*cent(cross(z,w)))
rng=np.random.default_rng(894521);checks=0;max_ratio=0
for sig in [1.,2.]:
 for q in [.001,.05,.1,.25,.5]:
  x=rng.choice([-1,1],256);s=rng.choice([-1,1],256);e=rng.normal(scale=sig,size=256);y=s*x+e;r=s*np.where(rng.random(256)<q,-1,1);a=1-2*q
  for u,z,v,w in [(y-a*r*x,np.column_stack([x,r]),e,np.column_stack([x,s])),(x-a*r*np.tanh(y/sig**2),np.column_stack([y,r]),x-s*np.tanh(y/sig**2),np.column_stack([y,s]))]:
   distance=np.sqrt(max(0,inner(u,z,u,z)+inner(v,w,v,w)-2*inner(u,z,v,w)))
   # The expectation is empirical here; its exact coupling bound is checked.
   du=np.sqrt(2*(1-np.exp(-(u-v)**2/2))).mean();dz=np.sqrt(2*(1-np.exp(-((z-w)**2).sum(1)/2))).mean();bound=2*(du+dz)
   assert distance<=bound+1e-12
   if bound>0:max_ratio=max(max_ratio,distance/bound)
   checks+=1
out=dict(posterior_checks=posterior_checks,maximum_posterior_error=max_error,operator_coupling_checks=checks,maximum_empirical_bound_ratio=max_ratio)
(P/'state_observation_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
