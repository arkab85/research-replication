import numpy as np, pandas as pd

def within(df, cols, group):
    """demean cols within group"""
    out = df[cols].to_numpy(float).copy()
    g = df[group].to_numpy()
    idx = pd.factorize(g)[0]
    for j in range(out.shape[1]):
        s = np.bincount(idx, weights=out[:,j]); n = np.bincount(idx)
        out[:,j] -= (s/n)[idx]
    return out

def twfe(df, y, X, unit, time, cluster, w=None):
    """two-way FE via alternating projections; cluster-robust SEs."""
    cols=[y]+X
    M=df[cols].to_numpy(float).copy()
    ui=pd.factorize(df[unit])[0]; ti=pd.factorize(df[time])[0]
    ww = np.ones(len(df)) if w is None else df[w].to_numpy(float)
    for _ in range(60):
        for idx in (ui,ti):
            sw=np.bincount(idx,weights=ww)
            for j in range(M.shape[1]):
                s=np.bincount(idx,weights=ww*M[:,j])
                M[:,j]-=(s/sw)[idx]
    yv=M[:,0]; Xv=M[:,1:]
    W=np.sqrt(ww)[:,None]
    XtX=(Xv*ww[:,None]).T@Xv
    beta=np.linalg.solve(XtX, (Xv*ww[:,None]).T@yv)
    r=yv-Xv@beta
    ci=pd.factorize(df[cluster])[0]; G=ci.max()+1
    meat=np.zeros((Xv.shape[1],Xv.shape[1]))
    for g in range(G):
        m=ci==g
        xg=(Xv[m]*ww[m,None]).T@r[m]
        meat+=np.outer(xg,xg)
    n,k=Xv.shape
    adj=(G/(G-1))*((n-1)/(n-k))
    XtXi=np.linalg.inv(XtX)
    V=adj*XtXi@meat@XtXi
    return beta, np.sqrt(np.diag(V)), G, n
