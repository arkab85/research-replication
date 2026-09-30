import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from engine import *
def joint_tests(X,Y,hs,p,tr,gap,Bw,Bp,seed):
    """Persistence statistic P_H = mean_h DII(h); wild bootstrap with a common weight path, paired MBB with common blocks."""
    S=[];D=[]
    for h in hs:
        s1=Stage1().fit(build(X[:tr],Y[:tr],h,p)); d=build(X[tr+gap:],Y[tr+gap:],h,p); S.append(s1); D.append(d)
    ns=[len(d) for d in D]; nmax=max(ns); ell=max(2,int(round(nmax**(1/3))))
    Qs=[]; obs=[]
    for s1,d in zip(S,D):
        ef,eb=s1.resid(d); Qs.append(core_q(ef,eb,d)); obs.append(dii(ef,eb,d))
    P=float(np.mean(obs)); rng=np.random.RandomState(seed)
    bw=np.array([np.mean([W[:n]@Q@W[:n]/n**2 for Q,n in zip(Qs,ns)]) for W in (wild_weights(nmax,ell,rng) for _ in range(Bw))])
    pw=(np.sum(bw>=P)+1)/(Bw+1)
    rng=np.random.RandomState(seed+1); bp=np.empty(Bp)
    for b in range(Bp):
        u=rng.rand(int(np.ceil(nmax/ell))+1); vals=[]
        for s1,d,n in zip(S,D,ns):
            nb=int(np.ceil(n/ell)); starts=(u[:nb]*(n-ell+1)).astype(int); idx=np.concatenate([np.arange(s,s+ell) for s in starts])[:n]
            e=sub(d,idx); e1,e2=s1.resid(e); vals.append(dii(e1,e2,e))
        bp[b]=np.mean(vals)
    pp=(np.sum((bp-P)>=P)+1)/(Bp+1)
    return obs,P,pw,pp
