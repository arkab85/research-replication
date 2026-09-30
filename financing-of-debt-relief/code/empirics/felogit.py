import numpy as np, pandas as pd
def sig(z): return 1/(1+np.exp(-np.clip(z,-35,35)))
def fe_logit(y, Z, cell, cluster=None, tol=1e-9, maxit=200):
    """Logit with cell fixed effects via block coordinate ascent. Drops cells with no variation in y.
    Returns beta, cluster-robust SE (partialled sandwich), n, ncell."""
    y=np.asarray(y,float); Z=np.asarray(Z,float); cell=pd.factorize(np.asarray(cell))[0]
    ym=np.bincount(cell,y)/np.bincount(cell)
    keep=(ym[cell]>0)&(ym[cell]<1)
    y,Z,cell=y[keep],Z[keep],pd.factorize(cell[keep])[0]
    if cluster is not None: cluster=pd.factorize(np.asarray(cluster)[keep])[0]
    C=cell.max()+1; k=Z.shape[1]
    # drop regressors with no within-cell variation
    zm=np.vstack([np.bincount(cell,Z[:,j],C) for j in range(k)]).T/np.bincount(cell)[:,None]
    ok=[j for j in range(k) if np.sum((Z[:,j]-zm[cell,j])**2)>1e-9]
    if len(ok)<k:
        Z=Z[:,ok]; k=len(ok)
    m=np.bincount(cell,y)/np.bincount(cell); a=np.log(m/(1-m)); b=np.zeros(k)
    for it in range(maxit):
        for _ in range(3):
            p=sig(a[cell]+Z@b); g=np.bincount(cell,y-p,C); h=np.bincount(cell,p*(1-p),C)
            a+=g/np.maximum(h,1e-12)
        p=sig(a[cell]+Z@b); w=p*(1-p)
        # partial out cell means (weighted) for beta Newton step
        zb=np.vstack([np.bincount(cell,w*Z[:,j],C) for j in range(k)]).T/np.maximum(np.bincount(cell,w,C),1e-12)[:,None]
        Zt=Z-zb[cell]
        A=(Zt*w[:,None]).T@Zt+1e-8*np.eye(k); gr=Zt.T@(y-p)
        step=np.linalg.solve(A,gr)
        step=np.clip(step,-1,1); b+=step
        if np.max(np.abs(step))<tol: break
    p=sig(a[cell]+Z@b); w=p*(1-p)
    zb=np.vstack([np.bincount(cell,w*Z[:,j],C) for j in range(k)]).T/np.maximum(np.bincount(cell,w,C),1e-12)[:,None]
    Zt=Z-zb[cell]; A=(Zt*w[:,None]).T@Zt; s=Zt*(y-p)[:,None]
    if cluster is None: B=s.T@s
    else:
        G=cluster.max()+1; S=np.vstack([np.bincount(cluster,s[:,j],G) for j in range(k)]).T; B=S.T@S*G/(G-1)
    Ai=np.linalg.inv(A+1e-8*np.eye(k)); V=Ai@B@Ai
    return b, np.sqrt(np.diag(V)), len(y), C, a, keep
