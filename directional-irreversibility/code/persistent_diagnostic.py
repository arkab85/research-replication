"""Post-results diagnostic, fixed six cells; not an empirical specification search.

Seed/design fixed before this script's execution. Assess n=120,m=204 under
correctly specified polynomial nuisance models, persistence rho in {0,.6},
convex exposure theta in {0,.5,1}. 100 replications, 199 bootstrap draws.
Original intersection uses block12; new simultaneous bounds use dyadic block4.
Different block schemes: this is not an isolated test of interval versus p-value.
"""
import numpy as np,itertools,json,pandas as pd
from pathlib import Path
from dii_core import dii_profile
from dii_confidence import simultaneous_dii_bounds
P=Path(__file__).resolve().parent.parent

def poly(z):
 return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(z.shape[1]),2)])

def run():
 rng=np.random.default_rng(20260914);rows=[];m=204;n=120;gap=32;reps=100;B=199
 for rho in (0.,.6):
  for theta in (0.,.5,1.):
   for rep in range(reps):
    N=1+m+gap+n;s=rng.normal(size=N);e=np.empty(N);e[0]=rng.normal()
    innovations=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
    for t in range(1,N):e[t]=rho*e[t-1]+innovations[t]
    y=theta*(s*s-1)+e
    arr=np.column_stack([s[1:],y[1:],s[:-1],y[:-1]])
    mu=arr[:m].mean(0);sd=arr[:m].std(0);z=(arr-mu)/sd
    tr=np.arange(m);te=np.arange(m+gap,m+gap+n)
    xf=np.column_stack([z[:,0],z[:,2:]]);xb=np.column_stack([z[:,1],z[:,2:]])
    pf=poly(xf);pb=poly(xb)
    rf=z[te,1]-pf[te]@np.linalg.lstsq(pf[tr],z[tr,1],rcond=None)[0]
    rb=z[te,0]-pb[te]@np.linalg.lstsq(pb[tr],z[tr,0],rcond=None)[0]
    for method,rf0,rb0 in [('oracle',innovations[te+1]/sd[1],s[te+1]/sd[0]),('fitted',rf,rb)]:
     comp={'h':(rf0,xf[te],rb0,xb[te])};seed=20260914+rep
     old=dii_profile(comp,12,B,seed)['comparisons']['h']
     new=simultaneous_dii_bounds(comp,B,seed)['comparisons']['h']
     rows.append({'rho':rho,'theta':theta,'rep':rep,'residuals':method,
      'wild_reject':old['p_wild']<=.05,'paired_reject':old['p_paired']<=.05,
      'intersection_reject':old['p_intersection']<=.05,
      'bounds_reject':new['dii_lower']>0,'bounds_width':new['dii_upper']-new['dii_lower'],
      'forward_upper':new['forward_upper']})
   print('Completed',rho,theta,flush=True)
 df=pd.DataFrame(rows);summary=[]
 for (rho,theta,method),d in df.groupby(['rho','theta','residuals']):
  out={'rho':rho,'theta':theta,'residuals':method,'replications':len(d)}
  for col in ['wild_reject','paired_reject','intersection_reject','bounds_reject']:
   p=d[col].mean();nn=len(d);zq=1.959963984540054;den=1+zq*zq/nn
   mid=(p+zq*zq/(2*nn))/den;half=zq*np.sqrt(p*(1-p)/nn+zq*zq/(4*nn*nn))/den
   out[col]={'rate':float(p),'wilson95':[float(mid-half),float(mid+half)]}
  out['mean_interval_width']=float(d.bounds_width.mean());summary.append(out)
 df.to_csv(P/'persistent_diagnostic_raw.csv',index=False)
 (P/'persistent_diagnostic_results.json').write_text(json.dumps({'seed':20260914,'m':m,'n':n,'gap':gap,'draws':B,'replications_per_cell':reps,'total_datasets':600,'summary':summary,'scope':'Post-results synthetic diagnostic. Both conditional means correctly specified; no empirical power estimate; n/m is not asymptotically vanishing in an equal-allocation sequence. Random training scaling is frozen but not justified by the fixed-kernel theorem here.'},indent=2))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
