from pathlib import Path
import numpy as np,pandas as pd,json
from dii_joint_nuisance import prepare_direction,joint_nuisance_dii
from dii_moment_benchmark import moment_diagnostic
from full_study import wilson
P=Path(__file__).resolve().parent.parent;rng=np.random.default_rng(20260919);R=200;B=399;rows=[]
def affine(z):return np.column_stack([np.ones(len(z)),z])
def draws(law,N):
 if law=='Gaussian':return rng.normal(size=N)
 if law=='Laplace':return rng.laplace(scale=1/np.sqrt(2),size=N)
 pick=rng.random(N)<2/7
 return np.where(pick,rng.laplace(scale=1/np.sqrt(2),size=N),rng.uniform(-np.sqrt(3),np.sqrt(3),size=N))
for n in (120,480):
 m=int(1.7*n);gap=int(np.ceil(8*np.log(n+1)));N=m+gap+n+1001;te=np.arange(m+gap,m+gap+n)
 for rho in (0.,.6):
  for law in ('Gaussian','Laplace','Moment_matched'):
   for rep in range(R):
    a=draws(law,N)*np.sqrt(1-rho*rho);b=draws(law,N)*np.sqrt(1-rho*rho);x=np.zeros(N);e=np.zeros(N)
    for t in range(1,N):x[t]=rho*x[t-1]+a[t];e[t]=rho*e[t-1]+b[t]
    y=x+e;C=np.column_stack([x[1000:-1],y[1000:-1]]);xx=x[1001:];yy=y[1001:]
    f=prepare_direction(yy,np.column_stack([xx,C]),m,te,affine);back=prepare_direction(xx,np.column_stack([yy,C]),m,te,affine);seed=20260919+rep
    rr=joint_nuisance_dii({'h':[f,back]},B,seed)['comparisons']['h'];mf=moment_diagnostic(f,B,seed);mb=moment_diagnostic(back,B,seed)
    rows.append(dict(n=n,m=m,rho=rho,law=law,rep=rep,dii=rr['dii'],dii_p=rr['p_intersection'],forward_moment=mf['moment'],forward_p=mf['p'],reverse_moment=mb['moment'],reverse_p=mb['p']))
   print('Completed',n,rho,law,flush=True);pd.DataFrame(rows).to_csv(P/'decision_use_study_raw.csv',index=False)
df=pd.DataFrame(rows);summary=[]
for (n,rho,law),g in df.groupby(['n','rho','law']):
 r=dict(n=int(n),rho=rho,law=law,replications=R)
 for label in ['dii','forward','reverse']:
  k=int((g[label+'_p']<=.05).sum());r[label]=dict(rejections=k,rate=k/R,wilson95=wilson(k,R))
 summary.append(r)
(P/'decision_use_study_results.json').write_text(json.dumps(dict(seed=20260919,datasets=2400,draws=B,summary=summary),indent=2));print(json.dumps(summary,indent=2))
