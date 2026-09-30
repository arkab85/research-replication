"""Reproducible fixed-kernel DII sensitivity calculations. No adaptive tuning."""
import numpy as np
from scipy.special import roots_hermitenorm, logsumexp
from scipy.spatial.distance import cdist

def shape_function(u):
    return (np.cos(u)-np.exp(-.5))/np.sqrt((1+np.exp(-2))/2-np.exp(-1))

def center(K):
    return K-K.mean(0)[None,:]-K.mean(1)[:,None]+K.mean()

def gram(x):
    x=np.asarray(x); x=x[:,None] if x.ndim==1 else x
    return np.exp(-cdist(x,x,'sqeuclidean')/2)

def component(e,z):
    K=gram(e); L=gram(z); A=center(K)*center(L)
    return max(0.,float(A.mean())), A, K, L

def unbiased(K,L):
    K=K.copy();L=L.copy();np.fill_diagonal(K,0);np.fill_diagonal(L,0);n=len(K)
    return ((K*L).sum()+K.sum()*L.sum()/((n-1)*(n-2))-2*(K.sum(1)*L.sum(1)).sum()/(n-2))/(n*(n-3))

def radius(As,ell=8,B=999,alpha=.05,seed=1):
    """Joint block Gaussian multiplier radius for covariance operators.
    Blocks sum centered tensor features. Remainder is a final shorter block.
    Returns norm radius (already divided by n), not a scalar DII bootstrap.
    """
    n=len(As[0]);starts=np.arange(0,n,ell);m=len(starts)
    w=np.random.default_rng(seed).normal(size=(B,m));mx=np.zeros(B)
    for A in As:
        Ac=center(A)
        S=np.add.reduceat(np.add.reduceat(Ac,starts,axis=0),starts,axis=1)
        norms=np.sqrt(np.maximum(0,np.einsum('bi,ij,bj->b',w,S,w,optimize=True)))/n
        mx=np.maximum(mx,norms)
    return float(np.quantile(mx,1-alpha,method='higher'))

def interval(hf,hb,rad,rf=0.,rb=0.):
    # Residual kernel bandwidths are one in all reported computations.
    lo=max(0,np.sqrt(hb)-rad-2*rb)**2-min(1,np.sqrt(hf)+rad+2*rf)**2
    hi=min(1,np.sqrt(hb)+rad+2*rb)**2-max(0,np.sqrt(hf)-rad-2*rf)**2
    return float(lo),float(hi)

def benchmark(phi=.8,rx=.3,ry=.7,p=1,h=1,k=1,both=False):
    # V=(X_t,Y_{t+h},C_t); U variance one; labels index latent state times.
    labels=[('x',0),('y',h-k)]+[('x',-j) for j in range(1,p+1)]
    if both: labels += [('y',-j-k) for j in range(1,p+1)]
    a,b=np.sqrt(rx),np.sqrt(ry);sx,se=np.sqrt(1-rx),np.sqrt(1-ry)
    load=np.array([a if v=='x' else b for v,t in labels]);times=np.array([t for v,t in labels])
    noise=np.array([sx if v=='x' else se for v,t in labels])
    cov=np.outer(load,load)*phi**np.abs(times[:,None]-times[None,:])+np.diag(noise**2)
    pars={'labels':labels,'noise':noise,'cov':cov,'phi':phi,'a':a,'bload':b,'sx':sx,'se':se,'p':p,'both':both}
    for side,j in [('b',0),('f',1)]:
        z=np.array([i for i in range(len(labels)) if i!=j]);beta=np.linalg.solve(cov[np.ix_(z,z)],cov[z,j]);v=cov[j,j]-cov[j,z]@beta
        pars[side]={'j':j,'z':z,'beta':beta,'v':float(v)}
    return pars

def amplitude(q,dx,dy,side='b'):
    d=np.array([dx if t=='x' else dy for t,_ in q['labels']]);r=q[side];z=r['z']
    S=float(np.sum((d/q['noise'])**2))
    residual=d[r['j']]+np.abs(r['beta'])@d[z]+np.sqrt(r['v']*S)
    scales=q.get('kernel_scales',np.ones(len(d)))
    return float(residual+np.linalg.norm(d[z]/scales[z]))

def envelope_constant(q,side='b'):
    return 4*max(amplitude(q,1.,0.,side),amplitude(q,0.,1.,side))**2

def sample(q,delta,n,rng,serial=False):
    """Outcome-only perturbation, h=k=1, cause-only p=1 history."""
    assert q['p']==1 and not q['both']
    phi=q['phi'];a,b,sx,se=[q[k] for k in ('a','bload','sx','se')]
    if serial:
        u=np.empty(n+1);u[0]=rng.normal()
        eta=rng.normal(size=n)*np.sqrt(1-phi**2)
        for i in range(n):u[i+1]=phi*u[i]+eta[i]
        x=a*u+sx*rng.normal(size=n+1);hist=x[:-1];x=x[1:];u=u[1:]
    else:
        old=rng.normal(size=n);u=phi*old+np.sqrt(1-phi**2)*rng.normal(size=n)
        hist=a*old+sx*rng.normal(size=n);x=a*u+sx*rng.normal(size=n)
    g=lambda t:b*t+delta*shape_function(t)
    y=g(u)+se*rng.normal(size=n)
    return x,y,hist

def oracle(q,delta,x,y,c,nodes=100):
    # U_t | X_{t-1} is Gaussian; conditional means are integrated in 1D.
    a,b,sx,se=[q[k] for k in ('a','bload','sx','se')];phi=q['phi']
    mu=phi*a*c;v=1-(phi*a)**2
    gain=v*a/(a*a*v+sx*sx)
    mf=mu+gain*(x-a*mu);vf=v*sx*sx/(a*a*v+sx*sx)
    z,w=roots_hermitenorm(nodes);w=w/np.sqrt(2*np.pi)
    uf=mf[:,None]+np.sqrt(vf)*z;g=lambda u:b*u+delta*shape_function(u)
    predy=g(uf)@w
    ub=mu[:,None]+np.sqrt(v)*z
    logw=np.log(w)[None,:]-.5*((y[:,None]-g(ub))/se)**2
    weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);predx=a*np.sum(weights*ub,axis=1)
    return y-predy,np.column_stack([x,c]),x-predx,np.column_stack([y,c])

def pair(comp):
    ef,zf,eb,zb=comp;f=component(ef,zf);b=component(eb,zb)
    return f,b

def strat_center(K,strata):
    """Center a Gram matrix within strata: H_s K H_s with H_s = I - block averaging."""
    strata=np.asarray(strata);M=np.zeros_like(K)
    for s in np.unique(strata):
        idx=np.where(strata==s)[0];M[np.ix_(idx,idx)]=1/len(idx)
    H=np.eye(len(K))-M
    return H@K@H

def strat_component(e,z,strata):
    """Stratified covariance operator (1/n) sum_i (Phi_i - Phi_bar_s)(x)(Psi_i - Psi_bar_s):
    the n_s/n-weighted average of within-stratum empirical covariance operators.
    Returns (squared norm, Gram matrix of the stratum-centered tensors)."""
    A=strat_center(gram(e),strata)*strat_center(gram(z),strata)
    return max(0.,float(A.mean())),A
