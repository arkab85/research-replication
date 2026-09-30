# Monte Carlo: coverage and size of identification sets for a bivariate SVAR under latent common volatility.
import numpy as np, sys, json, time
from svarcore import rot, _center, gram
cell=sys.argv[1]; reps=int(sys.argv[2]); shape,delta=cell.split('_'); delta=float(delta)
n=500; A=np.array([[1.0,0.5],[0.3,1.0]]); S=A@A.T; L=np.linalg.cholesky(S); Q0=np.linalg.solve(L,A)
th0=np.arctan2(Q0[1,0],Q0[0,0])%(np.pi/2)
grid=np.sort(np.r_[np.deg2rad(np.arange(0,90,2.0)),th0]); i0=int(np.argmin(abs(grid-th0)))
rho2=float(delta**2/(1+delta**2)); rng=np.random.default_rng(abs(hash(cell))%2**32)
def draw_eta(m):
    return rng.laplace(scale=1/np.sqrt(2),size=m) if shape=='lap' else rng.exponential(size=m)-1.0
cov_ind=cov_bud=cov_cs=0; size_ind=size_bud=size_cs=0
for r in range(reps):
    zeta=rng.choice([-1.0,1.0],size=n); sig=(1+delta*zeta)/np.sqrt(1+delta**2)
    eps=np.column_stack([sig*draw_eta(n),sig*draw_eta(n)]); u=eps@A.T; Z=np.linalg.solve(L,u.T).T   # known Sigma
    W=rng.standard_normal((199,n)); norms=[]; stats=[]; wald=[]
    for t in grid:
        E=Z@rot(t); Kc=_center(gram(E[:,0])); Lc=_center(gram(E[:,1])); M=Kc*Lc
        norms.append(np.sqrt(max(M.sum(),0))/n)
        Ac=M-M.mean(0,keepdims=True)-M.mean(1,keepdims=True)+M.mean()
        stats.append(np.sqrt(np.maximum(np.sum((W@Ac)*W,1),0))/n)
        Es=(E-E.mean(0))/E.std(0); g=np.column_stack([Es[:,0]**2*Es[:,1],Es[:,0]*Es[:,1]**2]); m=g.mean(0); V=np.cov(g.T)/n
        wald.append(float(m@np.linalg.solve(V,m)))
    norms=np.array(norms); q=np.quantile(np.max(np.array(stats),0),0.95); wald=np.array(wald)
    ind=norms-q<=0; bud=norms-q<=rho2; cs=wald<=5.991
    cov_ind+=ind[i0]; cov_bud+=bud[i0]; cov_cs+=cs[i0]; size_ind+=ind.mean(); size_bud+=bud.mean(); size_cs+=cs.mean()
res=dict(cell=cell,rho=float(np.sqrt(rho2)),reps=reps,cov_ind=cov_ind/reps,cov_bud=cov_bud/reps,cov_cs=cov_cs/reps,
         size_ind=size_ind/reps,size_bud=size_bud/reps,size_cs=size_cs/reps,theta0_deg=float(np.degrees(th0)))
json.dump(res,open(f'sim_{cell}.json','w')); print(res)
