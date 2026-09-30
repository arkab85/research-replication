"""Exact finite-state protection example and checks of RKHS decision bounds."""
from pathlib import Path
import numpy as np,json
P=Path(__file__).resolve().parent.parent
x=np.repeat([-2.,-1.,1.,2.],2);y=np.sqrt(1+x*x)*np.tile([-1.,1.],4);n=len(x);ell=.5;price=.22
k=lambda a,b:np.exp(-.5*((np.asarray(a)[:,None]-np.asarray(b)[None,:])/ell)**2)
K=k(y,y);L=k(x,x);J=np.eye(n)-np.ones((n,n))/n
H=float(np.sum((J@K@J)*(J@L@J))/n**2)
g=k(y,[-np.sqrt(5)]).ravel();w=k(x,[-2,2]).sum(1);q=g*w
vp=float(q.mean());vq=float(g.mean()*w.mean());B=float(np.sqrt(k(np.array([-2,2]),np.array([-2,2])).sum()))
assert vp>price>vq
assert abs(vp-vq)<=B*np.sqrt(H)+1e-12
# Conditional means zero exactly under the equally weighted symmetric support.
for a in np.unique(x):assert abs(y[x==a].mean())<1e-12
for a in np.unique(y):assert abs(x[y==a].mean())<1e-12
Hb=float(np.sum((J@L@J)*(J@K@J))/n**2);assert abs(H-Hb)<1e-15
# Independently check the normalized embedding witness achieves the discrepancy.
U=np.r_[y,np.repeat(y,n)];Z=np.r_[x,np.tile(x,n)];weights=np.r_[np.ones(n)/n,-np.ones(n*n)/n**2]
G=k(U,U)*k(Z,Z);mmd2=float(weights@G@weights);assert abs(mmd2-H)<1e-12
# Finite action library: signed combinations of Gaussian product sections plus costs.
rng=np.random.default_rng(20260913);cy=np.array([-np.sqrt(5),0,np.sqrt(5)]);cx=np.array([-2.,0,2.]);ay=np.repeat(cy,3);ax=np.tile(cx,3)
F=k(y,ay)*k(x,ax);fq=k(y,ay).mean(0)*k(x,ax).mean(0);gram=k(ay,ay)*k(ax,ax)
max_violation=0.;checks=0
for _ in range(100):
 coef=rng.normal(size=(7,9));cost=rng.uniform(-.1,.1,size=7);norms=np.sqrt(np.maximum(0,np.einsum('ai,ij,aj->a',coef,gram,coef)))
 valsP=coef@F.mean(0)-cost;valsQ=coef@fq-cost;chosen=int(np.argmax(valsQ));regret=float(valsP.max()-valsP[chosen]);bound=float(2*norms.max()*np.sqrt(H));assert regret<=bound+1e-10
 for a in range(7):
  for b in range(7):
   delta=coef[a]-coef[b];pairnorm=np.sqrt(max(0,float(delta@gram@delta)));margin=valsQ[a]-valsQ[b];true=valsP[a]-valsP[b]
   assert abs(true-margin)<=pairnorm*np.sqrt(H)+1e-10
   checks+=1
out=dict(type='Exact illustrative distribution; no estimated financial performance or Monte Carlo power claim',kernel_bandwidth=ell,price=price,forward_hsic=H,reverse_hsic=Hb,dii=Hb-H,expected_payoff_joint=vp,expected_payoff_independent=vq,true_net_value=vp-price,independent_net_value=vq-price,independent_choice='do not buy',true_choice='buy',foregone_expected_value=vp-price,payoff_rkhs_norm=B,score_error_bound=B*np.sqrt(H),generic_regret_bound=2*B*np.sqrt(H),finite_action_checks=checks)
(P/'decision_regret_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
