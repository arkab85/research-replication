import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from engine import *
def kfold_dii(X,Y,h,p=1,K=2):
    """Cross-fitted (K contiguous folds) sieve-ridge estimate of DII(h) with the full-vector statistic."""
    d=build(X,Y,h,p); n=len(d); edges=np.linspace(0,n,K+1).astype(int); ef=np.empty(n); eb=np.empty(n)
    for k in range(K):
        te=np.arange(edges[k],edges[k+1]); tr=np.setdiff1d(np.arange(n),te)
        s1=Stage1().fit(sub(d,tr)); e1,e2=s1.resid(sub(d,te)); ef[te]=e1; eb[te]=e2
    return dii(ef,eb,d)
def ar1s(T,rng,phi=0.5,noise=None):
    x=np.zeros(T)
    for t in range(1,T): x[t]=phi*x[t-1]+np.sqrt(1-phi**2)*(rng.normal() if noise is None else noise())
    return x
def cond(name,T,rng):
    if name=='A': return ar1s(T,rng),ar1s(T,rng)
    if name=='B': X=ar1s(T,rng); Y=np.zeros(T); Y[1:]=0.8*X[:-1]+rng.normal(0,1,T-1); return X,Y
    if name in ('C','I'): X=ar1s(T,rng); Y=np.zeros(T); Y[1:]=0.8*X[:-1]**2+rng.normal(0,1,T-1); return X,Y
    if name=='D': Y=ar1s(T,rng); X=np.zeros(T); X[1:]=0.8*Y[:-1]**2+rng.normal(0,1,T-1); return X,Y
    if name=='E':
        X=np.zeros(T); Y=np.zeros(T)
        for t in range(1,T):
            X[t]=0.4*X[t-1]+0.5*np.tanh(Y[t-1])+rng.normal(0,0.8); Y[t]=0.4*Y[t-1]+0.5*np.tanh(X[t-1])+rng.normal(0,0.8)
        return X,Y
    if name=='F': X=ar1s(T,rng,noise=lambda: rng.standard_t(3)/np.sqrt(3)); Y=np.zeros(T); Y[1:]=0.8*X[:-1]**2+rng.standard_t(3,T-1)/np.sqrt(3); return X,Y
    if name=='G': U=ar1s(T,rng,0.9); X=U+rng.normal(0,0.5,T); Y=np.zeros(T); Y[1:]=np.tanh(U[:-1])+rng.normal(0,0.3,T-1); return X,Y
    if name=='H': X=ar1s(T,rng); Y=np.zeros(T); Y[1:]=0.3*X[:-1]**2+rng.normal(0,1,T-1); return X,Y
    if name in ('J1','J2'): X=ar1s(T,rng); Y=np.zeros(T); Y[2:]=0.8*X[:-2]**2+rng.normal(0,1,T-2); return X,Y
