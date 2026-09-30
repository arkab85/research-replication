import numpy as np, json
from svarcore import *
y=np.loadtxt('oil_extended.txt'); p=24; cols=[0,2]; T=len(y)
dates=np.array([(1974+i//12)*100+(i%12+1) for i in range(T)])[p:]; th=np.deg2rad(np.arange(0,90,2.0))
drop=dates==202005
Tn,k=y.shape; X=np.column_stack([np.ones(Tn-p)]+[y[p-j:Tn-j] for j in range(1,p+1)])
Xd=np.column_stack([X,np.eye(Tn-p)[:,np.where(drop)[0]]]); Bd=np.linalg.lstsq(Xd,y[p:],rcond=None)[0]
U=(y[p:]-Xd@Bd)[~drop]; Bv=Bd[:1+p*k]; Z,L=whiten(U[:,cols]); nm=op_norms(Z,th); el=np.array([demand_elasticity(L,t) for t in th])
rng=np.random.default_rng(33); Uc=U-U.mean(0); D=[]
for b in range(100):
    idx=mbb_indices(len(Uc),12,rng); ys=var_regenerate(Bv,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us[:,cols]); D.append(op_diff_norms(Zs,Z,th))
q=float(np.quantile(np.array(D).max(1),0.95)); exc=np.maximum(nm-q,0); a0=exc<=0; e0=el[a0]; e0=e0[np.isfinite(e0)]
m=np.isfinite(el)&(el>=0.10)
res=dict(q=q,ind_share=float(a0.mean()),ind_el=[float(e0.min()),float(e0.max())],bd010=float(np.sqrt(exc[m].min())) if m.any() else None,rho_all=float(np.sqrt(exc.max())))
print(json.dumps(res)); json.dump(res,open('pandemic3.json','w'))
