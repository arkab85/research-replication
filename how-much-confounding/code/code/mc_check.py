import numpy as np, time
from jointproc import analyse
Kb=256/9; gam=0.01; Dtrue=0.0164277150
def path(n,phi,rng):
    x=np.empty(n); x[0]=rng.standard_normal()
    for t in range(1,n): x[t]=phi*x[t-1]+np.sqrt(1-phi**2)*rng.standard_normal()
    return x
for phi,reps in ((0.0,100),(0.6,60)):
    t0=time.time(); rng=np.random.default_rng(12345+int(phi*10)); exc=cov=exf=0; rv=[]
    for r in range(reps):
        n=600
        xtr=path(n,phi,rng); ytr=(xtr**2-1)/np.sqrt(2)+0.5*rng.standard_normal(n)
        xev=path(n,phi,rng); yev=(xev**2-1)/np.sqrt(2)+0.5*rng.standard_normal(n)
        lags=int(np.floor(4*(n/100)**(2/9))) if phi>0 else None
        o=analyse(xtr,ytr,xev,yev,draws=399,block=round(n**(1/3)),hc="HC3",hac_lags=lags,seed=r)
        L,U=o[("joint",0.0)]; Lf,_=o[("fixed",0.0)]
        exc+=L>Kb*gam**2; exf+=Lf>Kb*gam**2; cov+=(L<=Dtrue<=U); rv.append(np.sqrt(max(L,0)/Kb))
    print(f"phi={phi}: joint exclusion {100*exc/reps:.1f}% (paper {64.0 if phi==0 else 34.5}), ignore-uncertainty {100*exf/reps:.1f}% (paper {100.0 if phi==0 else 99.0}), coverage {100*cov/reps:.1f}%, median RV {np.median(rv):.4f} (paper {0.0115 if phi==0 else 0.0084}), {time.time()-t0:.0f}s")
