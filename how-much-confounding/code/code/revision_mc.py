"""Targeted calibration of the revised finite-grid and joint-whitening event.
100 replications per design, 99 iid residual bootstrap draws, fixed 6-degree
mesh, n=300 estimated VAR. No true angle is inserted into the inference mesh.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from svarcore import var_fit,var_regenerate,whiten,rot
from certified_revision import distances,stats,op_inner,CURV
OUT=Path(__file__).resolve().parent/'revision_results'
G=np.linspace(0,np.pi/2,16);A=np.array([[1.,.5],[.3,1.]])
L0=np.linalg.cholesky(A@A.T);Q0=np.linalg.solve(L0,A);theta=np.arctan2(Q0[1,0],Q0[0,0])%(np.pi/2);cell=min(int(theta/(np.pi/30)),14)
def one(task):
    delta,seed=task;rng=np.random.default_rng(seed);m=401;n=300;Phi=np.array([[.5,.1],[.2,.4]])
    sig=(1+delta*rng.choice([-1,1],m))/np.sqrt(1+delta**2)
    eps=(rng.exponential(size=(m,2))-1)*sig[:,None];u=eps@A.T;y=np.zeros((m,2))
    for t in range(1,m):y[t]=Phi@y[t-1]+u[t]
    y=y[100:];B,U=var_fit(y,1);Z,L=whiten(U);orig=stats(Z,G);D=[];R=[]
    for b in range(99):
        ys=var_regenerate(B,y[:1],U[rng.integers(0,n,n)]);_,ub=var_fit(ys,1);zb,lb=whiten(ub)
        D.append(np.max(distances(zb,Z,G,G,orig)));R.append(np.linalg.norm(np.linalg.solve(L,lb-L),'fro'))
    q=np.quantile(D,.975,method='higher');r=np.quantile(R,.975,method='higher');low=[]
    for j in range(15):
        a,b=orig[j:j+2];c=op_inner(Z@rot(G[j]),Z@rot(G[j+1]));v=max(a+b-2*c,0);t=np.clip((a-c)/v,0,1) if v>1e-16 else 0
        low.append(max(0,np.sqrt(max(a+2*t*(c-a)+t*t*v,0))-q-CURV*(G[j+1]-G[j])**2/8))
    low=np.array(low);rho2=delta**2/(1+delta**2)
    return dict(delta=delta,rho=float(np.sqrt(rho2)),seed=int(seed),q=float(q),r=float(r),ind_cover=bool(low[cell]<=0),budget_cover=bool(low[cell]<=rho2),L_cover=bool(np.linalg.norm(np.linalg.solve(L,L0-L),'fro')<=r),joint_parameter_cover=bool(low[cell]<=rho2 and np.linalg.norm(np.linalg.solve(L,L0-L),'fro')<=r),ind_share=float(np.mean(low<=0)),budget_share=float(np.mean(low<=rho2)))
if __name__=='__main__':
    tasks=[(d,int(s)) for d in (0.,.4) for s in np.random.SeedSequence(260927+int(d*100)).generate_state(100)]
    rows=[];t=time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,row in enumerate(ex.map(one,tasks,chunksize=1)):
            rows.append(row)
            if (i+1)%10==0:
                (OUT/'mc_raw.json').write_text(json.dumps(rows,indent=2));print(i+1,'/200 seconds',round(time.time()-t),flush=True)
    summary=[]
    for d in (0.,.4):
        rr=[x for x in rows if x['delta']==d];summary.append(dict(delta=d,rho=rr[0]['rho'],reps=len(rr),bootstrap=99,n=300,**{k:float(np.mean([x[k] for x in rr])) for k in ('ind_cover','budget_cover','L_cover','joint_parameter_cover','ind_share','budget_share')}))
    (OUT/'mc_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
