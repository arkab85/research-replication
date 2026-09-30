"""Fixed post-results audit of correction, six cells and equal algorithms.
Population scales are fixed; m=204,n=120; gap=ceil(8 log(n+1)); 100 datasets/cell.
Both estimators use identical nonoverlapping blocks/draws, fitted means and data.
No empirical dataset is used for method selection. All declared cells are kept.
"""
import numpy as np,json,pandas as pd
from pathlib import Path
from persistent_diagnostic import poly
from full_study import kernel,wilson
from dii_orthogonal import corrected_gram,nbb_dii_from_grams
P=Path(__file__).resolve().parent.parent
rng=np.random.default_rng(20260915);rows=[];m=204;n=120;gap=int(np.ceil(8*np.log(n+1)));R=100;B=199
for rho in (0.,.6):
 for theta in (0.,.5,1.):
  for rep in range(R):
   N=1+m+gap+n;s=rng.normal(size=N);e=np.empty(N);e[0]=rng.normal();innov=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
   for t in range(1,N):e[t]=rho*e[t-1]+innov[t]
   y=theta*(s*s-1)+e;sd=np.sqrt(2*theta*theta+1)
   z=np.column_stack([s[1:],y[1:]/sd,s[:-1],y[:-1]/sd]);te=np.arange(m+gap,m+gap+n)
   xf=np.column_stack([z[:,0],z[:,2:]]);xb=np.column_stack([z[:,1],z[:,2:]])
   f=poly(xf);b=poly(xb);rf=z[:,1]-f@np.linalg.lstsq(f[:m],z[:m,1],rcond=None)[0];rb=z[:,0]-b@np.linalg.lstsq(b[:m],z[:m,0],rcond=None)[0]
   lf=kernel(xf[te]);lb=kernel(xb[te]);gram_f=corrected_gram(rf[te],xf[te],rf[:m],xf[:m]);gram_b=corrected_gram(rb[te],xb[te],rb[:m],xb[:m])
   for method,kf,kb in [('raw',kernel(rf[te]),kernel(rb[te])),('orthogonal',gram_f,gram_b)]:
    r=nbb_dii_from_grams({'h':(kf,lf,kb,lb)},B,20260915+rep)['comparisons']['h']
    rows.append({'rho':rho,'theta':theta,'rep':rep,'method':method,**r,'reject':r['p_intersection']<=.05})
  print('Completed',rho,theta,flush=True)
df=pd.DataFrame(rows);summary=[]
for (rho,theta,method),d in df.groupby(['rho','theta','method']):
 k=int(d.reject.sum());summary.append({'rho':rho,'theta':theta,'method':method,'rejections':k,'datasets':len(d),'rate':k/len(d),'wilson95':wilson(k,len(d))})
df.to_csv(P/'orthogonal_diagnostic_raw.csv',index=False)
(P/'orthogonal_diagnostic_results.json').write_text(json.dumps({'seed':20260915,'m':m,'n':n,'gap':gap,'datasets':600,'draws':B,'summary':summary,'scope':'Synthetic small-sample diagnostic using fixed population scales. Both methods share nonoverlapping block4. No claim of empirical power or uniform size.'},indent=2))
print(json.dumps(summary,indent=2))
