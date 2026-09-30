# First-stage-aware Monte Carlo: VAR(1) estimated, Cholesky whitening estimated, residual moving-block bootstrap.
import numpy as np, sys, json, time
from svarcore import var_fit, var_regenerate, mbb_indices, whiten, rot, _center, gram, op_norms, op_diff_norms, coskew
delta=float(sys.argv[1]); reps=int(sys.argv[2]); seed=int(sys.argv[3])
n=300; burn=100; Phi=np.array([[0.5,0.1],[0.2,0.4]]); A=np.array([[1.0,0.5],[0.3,1.0]])
Lt=np.linalg.cholesky(A@A.T); Q0=np.linalg.solve(Lt,A); th0=np.arctan2(Q0[1,0],Q0[0,0])%(np.pi/2)
grid=np.sort(np.r_[np.deg2rad(np.arange(0,90,6.0)),th0]); i0=int(np.argmin(abs(grid-th0)))
rho2=float(delta**2/(1+delta**2)); rng=np.random.default_rng(seed)
cnt=dict(ind_mbb=0,ind_fixed=0,bud_mbb=0,cs_mbb=0,cs_an=0); size=dict(ind_mbb=0.0,ind_fixed=0.0,bud_mbb=0.0,cs_mbb=0.0,cs_an=0.0); t0=time.time()
for r in range(reps):
    m=n+burn; zeta=rng.choice([-1.0,1.0],size=m); sig=(1+delta*zeta)/np.sqrt(1+delta**2)
    eps=np.column_stack([sig*(rng.exponential(size=m)-1),sig*(rng.exponential(size=m)-1)]); u=eps@A.T
    y=np.zeros((m,2))
    for t in range(1,m): y[t]=Phi@y[t-1]+u[t]
    y=y[burn:]; B,U=var_fit(y,1); Z,L=whiten(U); nn=len(Z)
    nm=op_norms(Z,grid); cs=coskew(Z,grid); Uc=U-U.mean(0); D=[]; CS=[]
    for b in range(39):
        idx=mbb_indices(nn,5,rng); ys=var_regenerate(B,y[:1],Uc[idx]); _,Us=var_fit(ys,1); Zs,_=whiten(Us)
        D.append(op_diff_norms(Zs,Z,grid)); CS.append(coskew(Zs,grid))
    q=np.quantile(np.max(np.array(D),1),0.95); CS=np.array(CS)
    W=rng.standard_normal((199,nn)); st=[]
    for t in grid:
        E=Z@rot(t); M=_center(gram(E[:,0]))*_center(gram(E[:,1])); Ac=M-M.mean(0,keepdims=True)-M.mean(1,keepdims=True)+M.mean()
        st.append(np.sqrt(np.maximum(np.sum((W@Ac)*W,1),0))/nn)
    qf=np.quantile(np.max(np.array(st),0),0.95)
    wald=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(grid))])
    wan=[]
    for t in grid:
        E=Z@rot(t); Es=(E-E.mean(0))/E.std(0); g=np.column_stack([Es[:,0]**2*Es[:,1],Es[:,0]*Es[:,1]**2]); mm=g.mean(0); wan.append(float(mm@np.linalg.solve(np.cov(g.T)/nn,mm)))
    wan=np.array(wan)
    sets=dict(ind_mbb=nm<=q,ind_fixed=nm<=qf,bud_mbb=nm-q<=rho2,cs_mbb=wald<=5.991,cs_an=wan<=5.991)
    for k,s in sets.items(): cnt[k]+=bool(s[i0]); size[k]+=float(s.mean())
out=dict(delta=delta,rho=float(np.sqrt(rho2)),reps=reps,**{f"cov_{k}":cnt[k]/reps for k in cnt},**{f"size_{k}":size[k]/reps for k in size},sec=round(time.time()-t0))
print(json.dumps(out)); json.dump(out,open(f'simfs_{delta}_{seed}.json','w'))
