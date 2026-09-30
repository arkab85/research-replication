import numpy as np, sys, time, json
from svarcore import *
y=np.loadtxt('oil_extended.txt'); p=24; cols=[0,2]          # production growth, log real price (VAR has all 3)
thetas=np.deg2rad(np.arange(0,90,2.0))
B,U=var_fit(y,p); Z,L=whiten(U[:,cols])
chunk=int(sys.argv[1]); reps=int(sys.argv[2]); block=12
if chunk==0:
    np.savez('oil_orig.npz',norms=op_norms(Z,thetas),cs=coskew(Z,thetas),L=L,thetas=thetas,n=len(Z))
rng=np.random.default_rng(1000+chunk); Uc=U-U.mean(0); sup=[]; cs=[]; t0=time.time()
for r in range(reps):
    idx=mbb_indices(len(Uc),block,rng)
    ys=var_regenerate(B,y[:p],Uc[idx]); Bs,Us=var_fit(ys,p); Zs,_=whiten(Us[:,cols])
    sup.append(op_diff_norms(Zs,Z,thetas)); cs.append(coskew(Zs,thetas))
np.savez(f'oil_boot_{chunk}.npz',diff=np.array(sup),cs=np.array(cs))
print(f"chunk {chunk}: {reps} reps in {time.time()-t0:.0f}s")
