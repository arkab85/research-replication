import numpy as np, pandas as pd, json
from svarcore import *
Y=pd.read_csv('weekly.csv',index_col=0,parse_dates=True); y=Y.values; p=4; dates=Y.index[p:]
th=np.deg2rad(np.arange(0,90,3.0)); P=np.load('vix_point.npz'); nm=P['nm']; wald=P['wald']; a_vix=float(P['a_vix'])
D=np.concatenate([np.load(f'vix_boot_{c}.npy') for c in (0,1)]); q=float(np.quantile(D.max(1),0.95))
exc=np.maximum(nm-q,0); adm0=exc[:-1]<=0
res=dict(B=len(D),q=q,ind_admitted_deg=np.degrees(th[adm0]).round().astype(int).tolist(),ind_share=float(adm0.mean()),
         rho_returns_first=float(np.sqrt(exc[0])),rho_vix_first=float(np.sqrt(exc[-1])),rho_all=float(np.sqrt(exc[:-1].max())),
         cs_admitted_deg=np.degrees(th[wald[:-1]<=5.991]).round().astype(int).tolist(),cs_wald_returns_first=float(wald[0]),cs_wald_vix_first=float(wald[-1]),a_vix_deg=float(np.degrees(a_vix)))
print(json.dumps(res))
# influence: impulse dummies for the most influential weeks (co-skewness sets and independence point statistics)
def fit_drop(drop):
    Tn,k=y.shape; X=np.column_stack([np.ones(Tn-p)]+[y[p-j:Tn-j] for j in range(1,p+1)])
    Xd=np.column_stack([X,np.eye(Tn-p)[:,np.where(drop)[0]]]); Bd=np.linalg.lstsq(Xd,y[p:],rcond=None)[0]
    return Bd[:1+p*k],(y[p:]-Xd@Bd)[~drop]
rng=np.random.default_rng(9); inf={}
for name,weeks in (('2008-10-10',['2008-10-10']),('top five weeks',['2008-10-10','2015-08-21','2010-05-07','2001-09-21','2009-03-06']),
                   ('2008-09 to 2009-03',None)):
    drop=np.isin(dates.strftime('%Y-%m-%d'),weeks) if weeks else (dates>='2008-09-01')&(dates<='2009-03-31')
    Bd,Ud=fit_drop(drop); Zd,Ld=whiten(Ud); ad=np.arctan2(-Ld[1,0],Ld[1,1])%(np.pi/2); grid=np.r_[th,ad]
    cs=coskew(Zd,grid); Uc=Ud-Ud.mean(0); CS=[]
    for b in range(300):
        idx=mbb_indices(len(Uc),10,rng); ys=var_regenerate(Bd,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us); CS.append(coskew(Zs,grid))
    CS=np.array(CS); wd=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(grid))])
    nmd=op_norms(Zd,np.r_[0.0,ad])
    inf[name]=dict(dropped=int(drop.sum()),cs_admitted_deg=np.degrees(th[wd[:-1]<=5.991]).round().astype(int).tolist(),cs_wald_rf=round(float(wd[0]),2),cs_wald_vf=round(float(wd[-1]),2),
                   ind_norm_rf=round(float(nmd[0]),4),ind_norm_vf=round(float(nmd[1]),4),rf_admitted_ind=bool(nmd[0]<=q),vf_admitted_ind=bool(nmd[1]<=q))
    print(name,json.dumps(inf[name]))
res['influence']=inf; json.dump(res,open('vix_results.json','w'),indent=1)
