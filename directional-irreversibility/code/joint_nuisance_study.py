from pathlib import Path
import json,numpy as np,pandas as pd,itertools
from dii_joint_nuisance import prepare_direction,joint_nuisance_dii
from full_study import wilson
P=Path(__file__).resolve().parent.parent
rng=np.random.default_rng(20260918);m=204;n=120;gap=39;N=1+m+gap+n;te=np.arange(m+gap,m+gap+n);R=300;B=399;rows=[]
def ar(rho):
 x=np.empty(N);x[0]=rng.normal();e=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
 for t in range(1,N):x[t]=rho*x[t-1]+e[t]
 return x
def poly(z):return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])
def eqbasis(z):return np.column_stack([np.ones(len(z)),z[:,0],z[:,0]**2,z[:,1],z[:,0]*z[:,1]])
def missbasis(z):return np.column_stack([np.ones(len(z)),z,z[:,0]**2,z[:,1]**2])
def correctbasis(z):return np.column_stack([missbasis(z),z[:,1]**3,z[:,1]**4])
for rho in (0.,.6):
 for design in ('double','equal_positive','weak','strong','omitted_quartic_null'):
  for rep in range(R):
   if design=='equal_positive':
    a=ar(rho);b=ar(rho);c=rng.choice([-1.,1.],N);x=((1+.65*c)*a)[1:];y=((1+.65*c)*b)[1:];zf=np.column_stack([x,c[1:]]);zb=np.column_stack([y,c[1:]]);basis=eqbasis
   elif design=='omitted_quartic_null':
    a=ar(rho);b=ar(rho);c=rng.normal(size=N);x=(.3*(c**4-6*c**2+3)+a)[1:];y=b[1:];zf=np.column_stack([x,c[1:]]);zb=np.column_stack([y,c[1:]]);basis=missbasis
   else:
    s=rng.normal(size=N);e=ar(rho);theta={'double':0,'weak':.5,'strong':1}[design];yy=theta*(s*s-1)+e;x=s[1:];y=yy[1:];c=np.column_stack([s[:-1],yy[:-1]]);zf=np.column_stack([x,c]);zb=np.column_stack([y,c]);basis=poly
   directions=[prepare_direction(y,zf,m,te,basis),prepare_direction(x,zb,m,te,basis)];seed=20260918+rep
   if design=='omitted_quartic_null':
    variants=[('misspecified_joint',directions,True),('correct_basis_joint',[prepare_direction(y,zf,m,te,correctbasis),prepare_direction(x,zb,m,te,correctbasis)],True)]
   else:variants=[('coefficient_only',directions,False),('joint_scales',directions,True)]
   for name,dd,scales in variants:
    r=joint_nuisance_dii({'h':dd},B,seed,scales)['comparisons']['h'];rows.append(dict(rho=rho,design=design,rep=rep,method=name,**r,reject=r['p_intersection']<=.05))
  print('Completed',rho,design,flush=True)
  pd.DataFrame(rows).to_csv(P/'joint_nuisance_study_raw.csv',index=False)
df=pd.DataFrame(rows);summary=[]
for (rho,design,method),g in df.groupby(['rho','design','method']):
 k=int(g.reject.sum());summary.append(dict(rho=rho,design=design,method=method,rejections=k,replications=R,rate=k/R,wilson95=wilson(k,R)))
(P/'joint_nuisance_study_results.json').write_text(json.dumps(dict(seed=20260918,datasets=3000,m=m,n=n,gap=gap,draws=B,summary=summary),indent=2));print(json.dumps(summary,indent=2))
