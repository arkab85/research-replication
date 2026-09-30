"""Check kernel implementation against explicit finite feature calculations."""
import numpy as np,json
from pathlib import Path
from dii_confidence import simultaneous_dii_bounds,dyadic_block_length
rng=np.random.default_rng(719);n=24;ell=4;J=n//ell
f=rng.normal(size=(n,3));g=rng.normal(size=(n,2));f-=f.mean(0);g-=g.mean(0)
A=np.einsum('ni,nj->nij',f,g);Q=(f@f.T)*(g@g.T)
errs=[]
for _ in range(40):
 c=rng.multinomial(J,np.ones(J)/J);w=np.repeat(c-1,ell)
 explicit=np.sum((np.einsum('n,nij->ij',w,A)/np.sqrt(n))**2)
 errs.append(abs(explicit-w@Q@w/n))
 # Each component-norm interval contains the true squared norm whenever its
 # operator error is within the radius; test signed propagation independently.
 C=rng.normal(size=(3,2));E=rng.normal(size=(3,2));r=np.linalg.norm(E)
 z=np.linalg.norm(C+E);L=max(z-r,0)**2;U=(z+r)**2
 assert L-1e-10<=np.sum(C*C)<=U+1e-10
x=rng.normal(size=120);y=rng.normal(size=120)
r=simultaneous_dii_bounds({'same':(y,x,x,y)},199,91)
a=r['comparisons']['same'];assert abs(a['dii'])<1e-14 and a['dii_lower']<=0<=a['dii_upper']
assert dyadic_block_length(65)==dyadic_block_length(128)==4
assert dyadic_block_length(129)==dyadic_block_length(256)==4
assert dyadic_block_length(257)==8
assert max(errs)<1e-10
out={'max_gram_vs_explicit_error':max(errs),'exact_cancellation_interval':a,'dyadic_schedule_checked':True}
(Path(__file__).resolve().parent.parent/'confidence_verification.json').write_text(json.dumps(out,indent=2));print(out)
