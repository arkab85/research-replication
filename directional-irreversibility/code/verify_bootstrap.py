"""Check optimized bootstrap expressions against direct resampling and multiplication."""
import numpy as np
from full_study import kernel,center,component
rng=np.random.default_rng(92);n=12;B=9
K=kernel(rng.normal(size=n));L=kernel(rng.normal(size=(n,2)))
idx=rng.integers(0,n,size=(B,n));counts=np.zeros((B,n))
np.add.at(counts,(np.repeat(np.arange(B),n),idx.ravel()),1)
W=rng.normal(size=(B,n));h,w,p=component(K,L,W,counts)
errors=[]
for b in range(B):
 kb=K[np.ix_(idx[b],idx[b])];lb=L[np.ix_(idx[b],idx[b])]
 direct=np.sum(center(kb)*center(lb))/n**2
 errors.extend([abs(direct-p[b]),abs(w[b]-W[b]@(center(K)*center(L))@W[b]/n**2)])
assert max(errors)<1e-12
print('Maximum discrepancy from direct computations:',max(errors))
