"""Post-results attribution audit; no change to any earlier study.
Replays the 600 fixed orthogonal-study datasets. Adds known-mean and numerically
integrated population-derivative benchmarks, unavailable in empirical work.
"""
import numpy as np,pandas as pd,json
from pathlib import Path
from numpy.polynomial.legendre import leggauss
from persistent_diagnostic import poly
from full_study import kernel,wilson
from dii_orthogonal import nbb_dii_from_grams
P=Path(__file__).resolve().parent.parent

def gaussian_derivative_gram(u,variance):
 u=np.asarray(u);cross=u/(1+variance)**1.5*np.exp(-u*u/(2*(1+variance)))
 return kernel(u)-np.outer(cross,u)-np.outer(u,cross)+(1+2*variance)**(-1.5)*np.outer(u,u)

def posterior_basis(Q):
 nodes,weights=leggauss(Q);nodes=9*nodes;weights=9*weights
 dv=nodes[:,None]-nodes[None,:];F=(1-dv*dv)*np.exp(-dv*dv/2)
 return nodes,weights,F

def reverse_gram(u,a,theta,noisevariance,basis):
 if theta==0:return gaussian_derivative_gram(u,1.)
 v,basew,F=basis
 logw=np.log(basew)[None,:]-.5*v[None,:]**2-(a[:,None]-theta*(v[None,:]**2-1))**2/(2*noisevariance)
 logw-=logw.max(1)[:,None];w=np.exp(logw);w/=w.sum(1)[:,None]
 A=u[:,None]*w;uv=u[:,None]-v[None,:];E=uv*np.exp(-uv*uv/2);cross=E@A.T
 K=kernel(u)-cross-cross.T+A@F@A.T
 return (K+K.T)/2

# Convergence check over a deliberately broad signal range, before study execution.
rngcheck=np.random.default_rng(818);u=rngcheck.normal(size=80);a=np.linspace(-7,26,80)
b256=posterior_basis(256);b512=posterior_basis(512);b768=posterior_basis(768)
errors=[]
for theta in (.5,1.):
 for rho in (0.,.6):
  k256=reverse_gram(u,a,theta,1-rho*rho,b256);k512=reverse_gram(u,a,theta,1-rho*rho,b512);k768=reverse_gram(u,a,theta,1-rho*rho,b768)
  errors.append({'theta':theta,'rho':rho,'256_vs_512':float(np.max(abs(k256-k512))),'512_vs_768':float(np.max(abs(k512-k768)))})
assert max(x['512_vs_768'] for x in errors)<1e-9,errors
rng=np.random.default_rng(20260915);rows=[];m=204;n=120;gap=39;R=100;B=199
previous=pd.read_csv(P/'orthogonal_diagnostic_raw.csv');replayerrors=[]
for rho in (0.,.6):
 for theta in (0.,.5,1.):
  for rep in range(R):
   N=1+m+gap+n;s=rng.normal(size=N);e=np.empty(N);e[0]=rng.normal();innov=rng.normal(scale=np.sqrt(1-rho*rho),size=N)
   for t in range(1,N):e[t]=rho*e[t-1]+innov[t]
   y=theta*(s*s-1)+e;sd=np.sqrt(2*theta*theta+1)
   z=np.column_stack([s[1:],y[1:]/sd,s[:-1],y[:-1]/sd]);te=np.arange(m+gap,m+gap+n)
   xf=np.column_stack([z[:,0],z[:,2:]]);xb=np.column_stack([z[:,1],z[:,2:]])
   f=poly(xf);b=poly(xb);rf=z[:,1]-f@np.linalg.lstsq(f[:m],z[:m,1],rcond=None)[0];rb=z[:,0]-b@np.linalg.lstsq(b[:m],z[:m,0],rcond=None)[0]
   lf=kernel(xf[te]);lb=kernel(xb[te]);of=innov[te+1]/sd;ob=s[te+1];a=y[te+1]-rho*e[te]
   cases=[('known_means_raw',kernel(of),kernel(ob)),
    ('known_means_derivative',gaussian_derivative_gram(of,(1-rho*rho)/sd**2),reverse_gram(ob,a,theta,1-rho*rho,b512)),
    ('fitted_means_known_derivative',gaussian_derivative_gram(rf[te],(1-rho*rho)/sd**2),reverse_gram(rb[te],a,theta,1-rho*rho,b512))]
   if rep==0:
    replay=nbb_dii_from_grams({'h':(kernel(rf[te]),lf,kernel(rb[te]),lb)},B,20260915)['comparisons']['h']
    old=previous[(previous.rho==rho)&(previous.theta==theta)&(previous.rep==0)&(previous.method=='raw')].iloc[0]
    replayerrors.append(max(abs(replay[k]-old[k]) for k in ['dii','p_intersection']))
   for method,kf,kb in cases:
    r=nbb_dii_from_grams({'h':(kf,lf,kb,lb)},B,20260915+rep)['comparisons']['h']
    rows.append({'rho':rho,'theta':theta,'rep':rep,'method':method,**r,'reject':r['p_intersection']<=.05})
  print('Completed',rho,theta,flush=True)
assert max(replayerrors)<1e-12
new=pd.DataFrame(rows);combined=pd.concat([previous,new],ignore_index=True);summary=[]
for (rho,theta,method),d in combined.groupby(['rho','theta','method']):
 k=int(d.reject.sum());summary.append({'rho':rho,'theta':theta,'method':method,'rate':k/len(d),'wilson95':wilson(k,len(d)),'mean_dii':float(d.dii.mean()),'sd_dii':float(d.dii.std())})
new.to_csv(P/'oracle_derivative_audit_raw.csv',index=False)
(P/'oracle_derivative_audit_results.json').write_text(json.dumps({'seed':20260915,'datasets_replayed':600,'new_datasets':0,'draws':B,'quadrature_nodes':512,'quadrature_checks':errors,'replay_max_error':max(replayerrors),'summary':summary,'scope':'Post-results synthetic attribution audit. Population derivatives use the known DGP, numerically integrated on [-9,9]; not an empirically available estimator.'},indent=2))
print(json.dumps(summary,indent=2))
