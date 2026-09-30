import numpy as np, sys, json, time
from svarcore import *
y_all=np.loadtxt('oil_extended.txt'); cols=[0,2]
def run(name,p=24,block=12,start=0,end=None,Bk=100,Bc=400,step=2.0,seed=7):
    t0=time.time(); y=y_all[start:end]; th=np.deg2rad(np.arange(0,90,step))
    B,U=var_fit(y,p); Z,L=whiten(U[:,cols]); n=len(Z); Uc=U-U.mean(0); rng=np.random.default_rng(seed)
    nm=op_norms(Z,th) if Bk>0 else None; cs=coskew(Z,th); el=np.array([demand_elasticity(L,t) for t in th])
    CS=[]; D=[]
    for r in range(max(Bk,Bc)):
        idx=mbb_indices(n,block,rng); ys=var_regenerate(B,y[:p],Uc[idx]); _,Us=var_fit(ys,p); Zs,_=whiten(Us[:,cols])
        if r<Bc: CS.append(coskew(Zs,th))
        if r<Bk: D.append(op_diff_norms(Zs,Z,th))
    CS=np.array(CS); wald=np.array([cs[i]@np.linalg.solve(np.cov(CS[:,i,:].T),cs[i]) for i in range(len(th))])
    a=wald<=5.991; e=el[a]; e=e[np.isfinite(e)]
    out=dict(name=name,n=n,p=p,block=block,cs_share=float(a.mean()),cs_el=[float(e.min()),float(e.max())] if len(e) else None,cs_zero_excluded=bool(not a[0]))
    if Bk>0:
        q=float(np.quantile(np.array(D).max(1),0.95)); exc=np.maximum(nm-q,0); a0=exc<=0; e0=el[a0]; e0=e0[np.isfinite(e0)]
        m=np.isfinite(el)&(el>=0.10)
        out.update(q=q,ind_share=float(a0.mean()),ind_el=[float(e0.min()),float(e0.max())] if len(e0) else None,
                   bd010=float(np.sqrt(exc[m].min())) if m.any() else None,rho_all=float(np.sqrt(exc.max())))
    out['sec']=round(time.time()-t0); print(json.dumps(out)); json.dump(out,open(f'rob_{name}.json','w'))
if __name__=='__main__':
    specs={'lag12':dict(p=12),'block6':dict(block=6),'block24':dict(block=24),'pre2008':dict(end=408),'post1986':dict(start=144),
           'grid1':dict(step=1.0,Bk=0,Bc=400),'pre2020':dict(end=552)}
    for nm_ in sys.argv[1:]: run(nm_,**specs[nm_])
