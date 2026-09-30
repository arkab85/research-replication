import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); from engine import *
import pandas as pd
def unit(X,Y,dates,h,p=2,gap=12):
    """Honest split for one unit; returns stage-1, test design, calendar index of test rows, core Q."""
    T=len(X); n=int(np.floor(T/np.log(T))); tr=T-n-gap
    dtr=build(X[:tr],Y[:tr],h,p); dte=build(X[tr+gap:],Y[tr+gap:],h,p); s1=Stage1().fit(dtr); ef,eb=s1.resid(dte)
    lo=max(p,p); rows_dates=dates[tr+gap+lo: tr+gap+lo+len(dte)]   # row t corresponds to date of X_t
    return s1,dte,np.array(rows_dates),core_q(ef,eb,dte),dii(ef,eb,dte),len(dte)*hsic2(ef,dte[dte.attrs['f']].values)
def pooled_tests(units,B_w=999,B_p=299,seed=0):
    """units: list of dicts with keys s1,dte,dates,Q,obs. Pooled statistic = mean over units of DII.
    Wild: one calendar-indexed weight path shared by all units. Paired: shared relative block starts."""
    allD=sorted(set(np.concatenate([u['dates'] for u in units]))); idx={d:i for i,d in enumerate(allD)}; L=len(allD)
    nbar=int(np.mean([len(u['dte']) for u in units])); ell=max(2,int(round(nbar**(1/3))))
    P=np.mean([u['obs'] for u in units]); rng=np.random.RandomState(seed)
    bw=np.empty(B_w)
    for b in range(B_w):
        W=wild_weights(L,ell,rng); bw[b]=np.mean([(lambda w: w@u['Q']@w/len(w)**2)(W[[idx[d] for d in u['dates']]]) for u in units])
    pw=(np.sum(bw>=P)+1)/(B_w+1)
    rng=np.random.RandomState(seed+1); bp=np.empty(B_p)
    for b in range(B_p):
        uu=rng.rand(int(np.ceil(max(len(u['dte']) for u in units)/ell))+1); vals=[]
        for u in units:
            n=len(u['dte']); nb=int(np.ceil(n/ell)); starts=(uu[:nb]*(n-ell+1)).astype(int); ix=np.concatenate([np.arange(s,s+ell) for s in starts])[:n]
            e=sub(u['dte'],ix); e1,e2=u['s1'].resid(e); vals.append(dii(e1,e2,e))
        bp[b]=np.mean(vals)
    pp=(np.sum((bp-P)>=P)+1)/(B_p+1)
    return P,pw,pp
