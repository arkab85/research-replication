"""New post-results validation seed: 800 datasets, four laws x two persistence
levels x 100. Fixed design before execution. Original DII point estimates are
compared under old NBB, evaluation-only linearized and training-aware calibration.
Orthogonal fit is retained as a fourth comparator. No empirical tuning.
"""
import numpy as np,pandas as pd,json
from pathlib import Path
from full_study import kernel,wilson
from persistent_diagnostic import poly
from dii_orthogonal import corrected_gram,nbb_dii_from_grams
from dii_training_aware import training_aware_dii
P=Path(__file__).resolve().parent.parent;rng=np.random.default_rng(20260917);m=204;n=120;gap=39;B=399;R=100;rows=[]
for rho in (0.,.6):
 for design in ('double','equal_positive','weak','strong'):
  for rep in range(R):
   N=1+m+gap+n;theta={'double':0,'weak':.5,'strong':1}.get(design,0)
   if design=='equal_positive':
    a=np.empty(N);b=np.empty(N);a[0]=rng.normal();b[0]=rng.normal();ea=rng.normal(scale=np.sqrt(1-rho*rho),size=N);eb=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
    for t in range(1,N):a[t]=rho*a[t-1]+ea[t];b[t]=rho*b[t-1]+eb[t]
    c=rng.choice([-1.,1.],N);x=(1+.65*c)*a;y=(1+.65*c)*b
    zf=np.column_stack([x,c])[1:];zb=np.column_stack([y,c])[1:];x=x[1:];y=y[1:]
    def basis(z):return np.column_stack([np.ones(len(z)),z[:,0],z[:,0]**2,z[:,1],z[:,0]*z[:,1]])
    Pf=basis(zf);Pb=basis(zb)
   else:
    s=rng.normal(size=N);e=np.empty(N);e[0]=rng.normal();innov=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
    for t in range(1,N):e[t]=rho*e[t-1]+innov[t]
    yraw=theta*(s*s-1)+e;sd=np.sqrt(2*theta*theta+1)
    arr=np.column_stack([s[1:],yraw[1:]/sd,s[:-1],yraw[:-1]/sd]);x=arr[:,0];y=arr[:,1]
    zf=np.column_stack([x,arr[:,2:]]);zb=np.column_stack([y,arr[:,2:]]);Pf=poly(zf);Pb=poly(zb)
   te=np.arange(m+gap,m+gap+n);rf=y-Pf@np.linalg.lstsq(Pf[:m],y[:m],rcond=None)[0];rb=x-Pb@np.linalg.lstsq(Pb[:m],x[:m],rcond=None)[0]
   dirs=[{'residual_train':r[:m],'design_train':pp[:m],'residual_eval':r[te],'design_eval':pp[te],'regressors_eval':zz[te]} for r,pp,zz in [(rf,Pf,zf),(rb,Pb,zb)]]
   lf=kernel(zf[te]);lb=kernel(zb[te]);seed=20260917+rep
   outputs={}
   outputs['original_nbb']=nbb_dii_from_grams({'h':(kernel(rf[te]),lf,kernel(rb[te]),lb)},B,seed)['comparisons']['h']
   outputs['evaluation_only_linearized']=training_aware_dii({'h':dirs},B,seed,False)['comparisons']['h']
   outputs['training_aware']=training_aware_dii({'h':dirs},B,seed,True)['comparisons']['h']
   outputs['orthogonal']=nbb_dii_from_grams({'h':(corrected_gram(rf[te],zf[te],rf[:m],zf[:m]),lf,corrected_gram(rb[te],zb[te],rb[:m],zb[:m]),lb)},B,seed)['comparisons']['h']
   for method,r in outputs.items():rows.append({'rho':rho,'design':design,'rep':rep,'method':method,**r,'reject':r['p_intersection']<=.05})
  print('Completed',rho,design,flush=True)
df=pd.DataFrame(rows);summary=[]
for (rho,design,method),d in df.groupby(['rho','design','method']):
 k=int(d.reject.sum());summary.append({'rho':rho,'design':design,'method':method,'rate':k/R,'wilson95':wilson(k,R)})
df.to_csv(P/'training_aware_study_raw.csv',index=False)
(P/'training_aware_study_results.json').write_text(json.dumps({'seed':20260917,'datasets':800,'m':m,'n':n,'gap':gap,'draws':B,'summary':summary,'scope':'Post-results new-seed validation. Fixed population scales and correctly specified finite-dimensional mean models; not empirical power.'},indent=2));print(json.dumps(summary,indent=2))
