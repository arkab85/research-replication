# Is the co-skewness information structural or an episode? Diagnostics around 2020.
import numpy as np, json
from svarcore import var_fit, whiten, coskew, demand_elasticity, mbb_indices, rot
y=np.loadtxt('oil_extended.txt'); p=24; cols=[0,2]; T=len(y)
dates=np.array([(1974+i//12)*100+(i%12+1) for i in range(T)])[p:]      # yyyymm of each innovation
th=np.deg2rad(np.arange(0,90,2.0))
def var_fit_dummies(y,p,drop):                   # VAR with impulse dummies for the dropped innovation months
    Tn,k=y.shape; X=np.column_stack([np.ones(Tn-p)]+[y[p-j:Tn-j] for j in range(1,p+1)])
    D=np.zeros((Tn-p,drop.sum())); D[np.where(drop)[0],np.arange(drop.sum())]=1
    Xd=np.column_stack([X,D]) if drop.any() else X
    Bd=np.linalg.lstsq(Xd,y[p:],rcond=None)[0]; U=y[p:]-Xd@Bd
    return U[~drop]
def coskew_set(Z,L,rng,B=600,block=12):
    n=len(Z); cs=coskew(Z,th); CS=[]
    for b in range(B):
        idx=mbb_indices(n,block,rng); CS.append(coskew(Z[idx],th))       # moment-level block bootstrap (first stage fixed)
    CS=np.array(CS); wald=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(th))])
    a=wald<=5.991; el=np.array([demand_elasticity(L,t) for t in th]); e=el[a]; e=e[np.isfinite(e)]
    return dict(share=float(a.mean()),el=[float(e.min()),float(e.max())] if len(e) else None,zero_rejected=bool(not a[0]),wald0=float(wald[0]),n=n)
rng=np.random.default_rng(5); out={}
def run(name,drop=None,keep=None):
    drop=np.zeros(len(dates),bool) if drop is None else drop
    U=var_fit_dummies(y,p,drop)
    d=dates[~drop]
    if keep is not None: U=U[keep(d)]
    Z,L=whiten(U[:,cols]); out[name]=coskew_set(Z,L,rng); print(name,json.dumps(out[name]))
run('baseline (moment bootstrap)')
run('dummies 2020:03-2020:06',drop=(dates>=202003)&(dates<=202006))
run('dummies 2020:01-2021:12',drop=(dates>=202001)&(dates<=202112))
run('dummies 2020:01-2025:09 (drop all post-2019)',drop=dates>=202001)
run('innovations 2008:01-2025:09 only',keep=lambda d: d>=200801)
# influence: contribution of each month to the co-skewness moments at the recursive rotation
B,U=var_fit(y,p); Z,L=whiten(U[:,cols]); E=(Z-Z.mean(0))/Z.std(0)
g=np.column_stack([E[:,0]**2*E[:,1],E[:,0]*E[:,1]**2]); share=np.abs(g).sum(1)/np.abs(g).sum()
top=np.argsort(-np.abs(g).sum(1))[:8]
print("largest contributions to |co-skewness| at the recursive rotation:",[(int(dates[i]),round(float(share[i]),3)) for i in top])
print("share of total from 2020:01-2020:12:",float(share[(dates>=202001)&(dates<=202012)].sum()))
# common-volatility gauge: rolling 12-month standard deviations of the Cholesky shocks at the recursive rotation
r=np.array([E[max(0,t-6):t+6].std(0) for t in range(len(E))])
rel=r/np.sqrt((r**2).mean(0))
print("RMS deviation of rolling 12-month volatility from its mean level (production, price):",np.sqrt(((rel-1)**2).mean(0)).round(3).tolist(),
      "; correlation of the two rolling volatilities:",round(float(np.corrcoef(r.T)[0,1]),3))
out['top_months']=[(int(dates[i]),float(share[i])) for i in top]; out['share_2020']=float(share[(dates>=202001)&(dates<=202012)].sum())
out['vol_rms']=np.sqrt(((rel-1)**2).mean(0)).tolist(); out['vol_corr']=float(np.corrcoef(r.T)[0,1])
json.dump(out,open('pandemic.json','w'),indent=1)
