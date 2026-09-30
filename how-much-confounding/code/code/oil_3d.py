# Three-variable co-skewness set (robust to latent common-state confounding) for Kilian's VAR(24).
import numpy as np, json, time
from svarcore import var_fit, var_regenerate, mbb_indices
from scipy.stats import chi2
t0=time.time(); y=np.loadtxt('oil_extended.txt'); p=24
B,U=var_fit(y,p); n=len(U)
def whiten3(U):
    L=np.linalg.cholesky(np.cov(U.T)); return np.linalg.solve(L,U.T).T, L
def T3(Z):
    Zc=Z-Z.mean(0); return np.einsum('ta,tb,tc->abc',Zc,Zc,Zc).ravel()/len(Z)
Z,L=whiten3(U); T=T3(Z)
rng=np.random.default_rng(11); Uc=U-U.mean(0); TB=[]
for r in range(500):
    idx=mbb_indices(n,12,rng); ys=var_regenerate(B,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten3(Us); TB.append(T3(Zs))
Om=np.cov(np.array(TB).T)
# rotations: identity (recursive Cholesky ordering: production, activity, price) plus uniform random rotations
N=60000; G=rng.standard_normal((N,3,3)); Qs,Rs=np.linalg.qr(G); Qs=Qs*np.sign(np.einsum('nii->ni',Rs))[:,None,:]
Qs=np.concatenate([np.eye(3)[None],Qs])
pairs=[(i,i,j) for i in range(3) for j in range(3) if i!=j]+[(0,1,2)]
W=np.stack([np.einsum('na,nb,nc->nabc',Qs[:,:,i],Qs[:,:,j],Qs[:,:,k]).reshape(len(Qs),27) for i,j,k in pairs],1)   # (N,7,27)
m=W@T; V=np.einsum('nkp,pq,nlq->nkl',W,Om,W)
wald=np.einsum('nk,nk->n',m,np.linalg.solve(V,m[...,None])[...,0]); crit=chi2.ppf(0.95,7); adm=wald<=crit
A=np.einsum('ab,nbc->nac',L,Qs)                      # impact matrices, columns = shocks
same=A[:,0,:]*A[:,2,:]>0                              # demand-type columns: production and price move together
el=np.where(same,A[:,0,:]/A[:,2,:],np.nan)
ea=el[adm]; vals=ea[np.isfinite(ea)]
maxdem=np.nanmax(np.abs(np.where(same,el,np.nan)),axis=1)   # largest demand elasticity in the rotation
res=dict(n=n,rotations=int(len(Qs)),share_admitted=float(adm.mean()),recursive_wald=float(wald[0]),crit=float(crit),recursive_admitted=bool(adm[0]),
         el_range=[float(np.min(vals)),float(np.max(vals))] if len(vals) else None,
         el_q05_q95=[float(np.quantile(vals,0.05)),float(np.quantile(vals,0.95))] if len(vals) else None,
         min_maxdem_admitted=float(np.nanmin(maxdem[adm])) if adm.any() else None, sec=round(time.time()-t0))
two=(np.isfinite(el).sum(1)==2)
e_sorted=np.sort(np.where(np.isfinite(el),el,np.inf),axis=1)[:,:2]
consistent=two&(np.abs(e_sorted[:,0]-e_sorted[:,1])<=0.25*np.maximum(e_sorted[:,1],1e-9))
sc=adm&consistent; eta=e_sorted[sc].mean(1) if sc.any() else np.array([])
res.update(supply_curve_admitted=int(sc.sum()),supply_curve_share_of_admitted=float(sc.sum()/max(adm.sum(),1)),
           eta_range=[float(eta.min()),float(eta.max())] if len(eta) else None,
           eta_q05_q95=[float(np.quantile(eta,0.05)),float(np.quantile(eta,0.95))] if len(eta) else None)
print(json.dumps(res)); json.dump(res,open('oil3d.json','w'))
