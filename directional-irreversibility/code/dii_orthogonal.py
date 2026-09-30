"""Orthogonal DII in the original Gaussian RKHS, with training-only nuisances.

No random Fourier approximation is used in the estimator. The derivative
conditional mean is a Nadaraya-Watson average of Gaussian derivative features.
The target kernels are fixed at one. The nuisance smoother uses bandwidth
    m**(-1/(p+4)), shrinking with its training sample size and dimension p. The caller must justify
mean/derivative nuisance rates, moments, splitting and joint dependence.
"""
import numpy as np
from full_study import kernel,component
from dii_confidence import dyadic_block_length

def corrected_gram(u,z,v,z_train):
    u=np.asarray(u).reshape(-1);v=np.asarray(v).reshape(-1)
    z=np.asarray(z);zt=np.asarray(z_train)
    if z.ndim==1:z=z[:,None]
    if zt.ndim==1:zt=zt[:,None]
    if len(z)!=len(u) or len(zt)!=len(v) or z.shape[1]!=zt.shape[1]:raise ValueError('Incompatible arrays')
    if not all(np.isfinite(a).all() for a in [u,v,z,zt]):raise ValueError('Nonfinite input')
    d=((z[:,None,:]-zt[None,:,:])**2).sum(2)
    bandwidth=len(v)**(-1/(z.shape[1]+4))
    logits=-d/(2*bandwidth**2);logits-=logits.max(1)[:,None]
    w=np.exp(logits);w/=w.sum(1)[:,None]
    uv=u[:,None]-v[None,:];ev=np.exp(-uv*uv/2)*uv
    vv=v[:,None]-v[None,:];dd=(1-vv*vv)*np.exp(-vv*vv/2)
    A=u[:,None]*w
    cross=ev@A.T
    K=kernel(u)-cross-cross.T+A@dd@A.T
    return (K+K.T)/2

def nbb_dii_from_grams(grams,bootstrap_draws=199,seed=20260915):
    """name -> (residual-forward Gram, regressor-forward Gram, reverse pair).
    Uses one nonoverlapping path across the full fixed family. Squared norms of
    the corrected covariance operators estimate the original population HSICs.
    """
    names=list(grams);N=len(grams[names[0]][0]);ell=dyadic_block_length(N);n=N//ell*ell;J=n//ell;B=bootstrap_draws
    rng=np.random.default_rng(seed);mult=rng.multinomial(J,np.ones(J)/J,size=B)
    counts=np.repeat(mult,ell,axis=1).astype(float);W=counts-1
    result={}
    for name in names:
        Kf,Lf,Kb,Lb=[np.asarray(a)[:n,:n] for a in grams[name]]
        hf,qf,pf=component(Kf,Lf,W,counts);hb,qb,pb=component(Kb,Lb,W,counts)
        D=hb-hf;quad=qb-qf;paired=pb-pf-D
        pq=(1+np.count_nonzero(quad>=D-1e-12))/(B+1);pp=(1+np.count_nonzero(paired>=D-1e-12))/(B+1)
        result[name]={'forward_hsic':float(hf),'reverse_hsic':float(hb),'dii':float(D),'p_quadratic':float(pq),'p_paired':float(pp),'p_intersection':float(max(pq,pp))}
    return {'n':n,'block_length':ell,'bootstrap_draws':B,'seed':seed,'comparisons':result}
