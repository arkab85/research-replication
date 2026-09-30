import numpy as np, pandas as pd, sys, json, time
from svarcore import *
Y=pd.read_csv('weekly.csv',index_col=0,parse_dates=True); y=Y.values; p=4; dates=Y.index[p:]
th=np.deg2rad(np.arange(0,90,3.0))
B,U=var_fit(y,p); Z,L=whiten(U); n=len(Z)
def rec_angles(L):
    a0=0.0                                               # returns first: VIX shock has no impact on returns
    t=np.arctan2(-L[1,0],L[1,1])%(np.pi/2)               # VIX first: return shock has no impact on VIX
    return a0,t
a_ret,a_vix=rec_angles(L)
mode=sys.argv[1]
if mode=='point':
    nm=op_norms(Z,np.r_[th,a_vix]); cs=coskew(Z,np.r_[th,a_vix]); rng=np.random.default_rng(4); Uc=U-U.mean(0); CS=[]
    for b in range(400):
        idx=mbb_indices(n,10,rng); ys=var_regenerate(B,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us); CS.append(coskew(Zs,np.r_[th,a_vix]))
    CS=np.array(CS); wald=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(cs))])
    np.savez('vix_point.npz',nm=nm,wald=wald,a_vix=a_vix)
    print(f"n={n}; VIX-first angle {np.degrees(a_vix):.1f} deg; ||C|| returns-first {nm[0]:.4f}, VIX-first {nm[-1]:.4f}, min {nm[:-1].min():.4f} at {np.degrees(th[nm[:-1].argmin()]):.0f}")
    print(f"co-skewness Wald: returns-first {wald[0]:.2f}, VIX-first {wald[-1]:.2f}; share of rotations admitted {np.mean(wald[:-1]<=5.991):.2f}; admitted angles {np.degrees(th[wald[:-1]<=5.991]).round().astype(int).tolist()}")
    # influence at each recursive rotation
    for nmk,a in (('returns-first',a_ret),('VIX-first',a_vix)):
        E=Z@rot(a); E=(E-E.mean(0))/E.std(0); g=np.abs(np.column_stack([E[:,0]**2*E[:,1],E[:,0]*E[:,1]**2])).sum(1); sh=g/g.sum(); top=np.argsort(-sh)[:5]
        print(nmk,"top weeks:",[(str(dates[i].date()),round(float(sh[i]),3)) for i in top])
else:
    c=int(mode); reps=int(sys.argv[2]); grid=np.r_[th,a_vix]; rng=np.random.default_rng(100+c); Uc=U-U.mean(0); D=[]; t0=time.time()
    for r in range(reps):
        idx=mbb_indices(n,10,rng); ys=var_regenerate(B,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us); D.append(op_diff_norms(Zs,Z,grid))
    np.save(f'vix_boot_{c}.npy',np.array(D)); print(f"chunk {c}: {reps} reps in {time.time()-t0:.0f}s")
