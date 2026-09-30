"""Joint operator confidence bounds using paired nonoverlapping blocks.

An additional algorithm: it does not silently replace dii_core's circular/wild
procedures. Kernels are fixed unit Gaussian; all comparisons share a calendar.
Validity requires the weak-dependence, nuisance-rate and quantile conditions in
Theorem 'Simultaneous component and directional bounds'. No uniform claim.
"""
import numpy as np
from full_study import kernel

def dyadic_block_length(N):
    if int(N)!=N or N<2: raise ValueError('Need integer N >= 2')
    # Constant for 2**(k-1) < N <= 2**k, nondecreasing, O(N**(1/3)).
    k=(int(N)-1).bit_length()
    return 2**(k//3)

def operator_gram(residual,regressors):
    k=kernel(np.asarray(residual));l=kernel(np.asarray(regressors))
    k=k-k.mean(0)[None,:]-k.mean(1)[:,None]+k.mean()
    l=l-l.mean(0)[None,:]-l.mean(1)[:,None]+l.mean()
    return k*l

def simultaneous_dii_bounds(comparisons,bootstrap_draws=999,seed=20260914,alpha=.05,block_length=None):
    if not comparisons: raise ValueError('Empty comparisons')
    names=list(comparisons);N=len(comparisons[names[0]][0])
    ell=dyadic_block_length(N) if block_length is None else int(block_length)
    if ell<1 or ell>N or (block_length is not None and ell!=block_length):raise ValueError('Invalid block length')
    B=int(bootstrap_draws)
    if B<1 or B!=bootstrap_draws or not 0<alpha<1:raise ValueError('Invalid draws/alpha')
    n=N//ell*ell;J=n//ell
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(J,np.ones(J)/J,size=B)
    W=np.repeat(counts-1,ell,axis=1).astype(float)
    observed=[];boot_norm=[]
    for name in names:
        arrays=list(map(np.asarray,comparisons[name]))
        if len(arrays)!=4 or any(len(a)!=N or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Require finite aligned arrays')
        rf,zf,rb,zb=[a[:n] for a in arrays]
        for r,z in ((rf,zf),(rb,zb)):
            A=operator_gram(r,z)
            observed.append(float(max(A.mean(),0)))
            boot_norm.append(np.sqrt(np.maximum(np.sum((W@A)*W,axis=1)/n,0)))
    T=np.max(boot_norm,axis=0)
    # Upper empirical order statistic; B must grow for the stated quantile theorem.
    rank=int(np.ceil((B+1)*(1-alpha)))
    q=float(np.sort(T)[rank-1]) if rank<=B else float('inf')
    t=q/np.sqrt(n);obs=np.asarray(observed);s=np.sqrt(obs)
    lo=np.maximum(s-t,0)**2;hi=(s+t)**2
    result={}
    for j,name in enumerate(names):
        f,b=2*j,2*j+1
        result[name]={'forward_hsic':float(obs[f]),'reverse_hsic':float(obs[b]),
          'dii':float(obs[b]-obs[f]),'forward_lower':float(lo[f]),'forward_upper':float(hi[f]),
          'reverse_lower':float(lo[b]),'reverse_upper':float(hi[b]),
          'dii_lower':float(lo[b]-hi[f]),'dii_upper':float(hi[b]-lo[f])}
    return {'n_available':N,'n_retained':n,'block_length':ell,'bootstrap_draws':B,'seed':seed,
      'alpha':alpha,'operator_critical_value':q,'operator_radius':float(t),'comparisons':result,
      'interpretation':'Simultaneous pointwise asymptotic bounds, conditional on stated assumptions; not a causal certificate.'}
