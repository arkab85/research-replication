import numpy as np, json
from svarcore import var_fit, var_regenerate, whiten, coskew, demand_elasticity, mbb_indices
y=np.loadtxt('oil_extended.txt'); p=24; cols=[0,2]; T=len(y)
dates=np.array([(1974+i//12)*100+(i%12+1) for i in range(T)])[p:]
th=np.deg2rad(np.arange(0,90,2.0)); crit=5.991
def summarize(cs,CS,L):
    wald=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(th))]); a=wald<=crit
    el=np.array([demand_elasticity(L,t) for t in th]); e=el[a]; e=e[np.isfinite(e)]
    return dict(share=round(float(a.mean()),3),el=[round(float(e.min()),3),round(float(e.max()),3)] if len(e) else None,zero_rejected=bool(not a[0]),wald0=round(float(wald[0]),2))
def fit_drop(drop):
    Tn,k=y.shape; X=np.column_stack([np.ones(Tn-p)]+[y[p-j:Tn-j] for j in range(1,p+1)])
    Xd=np.column_stack([X,np.eye(Tn-p)[:,np.where(drop)[0]]]) if drop.any() else X
    Bd=np.linalg.lstsq(Xd,y[p:],rcond=None)[0]; U=y[p:]-Xd@Bd; return Bd[:1+p*k], U[~drop]
res={}; rng=np.random.default_rng(21)
# (a) baseline: moment-level block bootstrap WITH re-whitening
B,U=var_fit(y,p); Z,L=whiten(U[:,cols]); cs=coskew(Z,th); CS=[]
for b in range(600):
    idx=mbb_indices(len(U),12,rng); Zs,_=whiten(U[idx][:,cols]); CS.append(coskew(Zs,th))
res['baseline, block bootstrap of innovations with re-whitening']=summarize(cs,np.array(CS),L)
# (b) residual MBB (regenerate, re-estimate VAR, re-whiten) with pandemic months dummied out
for name,drop in [('none',np.zeros(len(dates),bool)),('2020:03-2020:06',(dates>=202003)&(dates<=202006)),
                  ('2020:05 only',dates==202005),('2020:01-2021:12',(dates>=202001)&(dates<=202112))]:
    Bd,Ud=fit_drop(drop); Zd,Ld=whiten(Ud[:,cols]); cs=coskew(Zd,th); Uc=Ud-Ud.mean(0); CS=[]
    for b in range(400):
        idx=mbb_indices(len(Uc),12,rng); ys=var_regenerate(Bd,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us[:,cols]); CS.append(coskew(Zs,th))
    res[f'residual MBB, dummies: {name}']=summarize(cs,np.array(CS),Ld)
for k,v in res.items(): print(k,json.dumps(v))
json.dump(res,open('pandemic2.json','w'),indent=1)
