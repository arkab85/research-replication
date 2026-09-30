"""Verify analytical network propositions on constructed inputs, not city data."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares, minimize
R=Path(__file__).resolve().parent

def objects(n,B,a,out,theta):
 z=out+a@n**theta;p=a*n[None,:]**theta/z[:,None];D=p.T@B;V=p/n
 A=(1-theta)*np.diag(D/n**2)+theta*V.T@np.diag(B)@V
 return p,D,V,A

def solve(B,a,out,theta,k):
 def fun(t):
  n=np.exp(t);p,D,V,A=objects(n,B,a,out,theta)
  return (D/n-k)/k
 def jac(t):
  n=np.exp(t);p,D,V,A=objects(n,B,a,out,theta)
  return -A*n[None,:]/k[:,None]
 x=least_squares(fun,np.log(np.sum(B)/len(k)/k),jac=jac,xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=400)
 assert np.max(np.abs(fun(x.x)))<1e-9
 return np.exp(x.x)

rng=np.random.default_rng(20260913);err=0;res=0;mineig=1;count=0;scaleerr=0;spectralerr=0
for M in [2,5,9,12]:
 for theta in [.2,.6,.9]:
  for opened in [False,True]:
   for heterogeneous in [False,True]:
    I=M+2;B=rng.uniform(2,15,I);a=np.exp(rng.uniform(-3,0,(I,M)));out=rng.uniform(.3,2,I) if opened else np.zeros(I);k=rng.uniform(.7,1.5,M) if heterogeneous else np.ones(M)
    n=solve(B,a,out,theta,k);p,D,V,A=objects(n,B,a,out,theta);res=max(res,float(np.max(abs(D/n-k))));mineig=min(mineig,float(np.linalg.eigvalsh(A).min()))
    dB=rng.normal(size=I);dB-=dB.mean();dk=rng.normal(size=M)*.05;h=1e-4
    analytic=np.linalg.solve(A,V.T@dB-dk)
    numeric=(solve(B+h*dB,a,out,theta,k+h*dk)-solve(B-h*dB,a,out,theta,k-h*dk))/(2*h)
    err=max(err,float(np.max(abs(analytic-numeric))));assert np.max(abs(analytic-numeric))<2e-6
    if not opened:
     assert abs(k@analytic+n@dk-dB.sum())<1e-9
    if not opened and not heterogeneous:
     q=n/n.sum();w=B/B.sum();Q=np.diag(q)-p.T@np.diag(w)@p;Rq=Q/np.sqrt(q[:,None]*q[None,:]);ev=np.linalg.eigvalsh(Rq);assert ev.min()>-1e-12 and ev.max()<1+1e-12
     b=(p.T@(dB/B.sum()))/np.sqrt(q);y=np.linalg.solve(np.eye(M)-theta*Rq,b)
     direct=np.linalg.solve(A,V.T@dB)
     spectralerr=max(spectralerr,float(np.max(abs(y-np.sqrt(q)*direct/n))))
     ns=solve(3.7*B,a,np.zeros(I),theta,k);scaleerr=max(scaleerr,float(np.max(abs(ns/3.7-n))))
    if opened and not heterogeneous:
     p0=out/(out+a@n**theta);direct=np.linalg.solve(A,V.T@dB)
     rhs=(1-p0)@dB+theta*(p.T@(B*p0))@(direct/n)
     assert abs(direct.sum()-rhs)<1e-9
    count+=1

# Capacity-constrained potential and scale correspondence.
M=5;I=6;B=rng.uniform(5,15,I);a=np.exp(rng.uniform(-2,0,(I,M)));out=np.zeros(I);theta=.6;k=np.ones(M)
n0=solve(B,a,out,theta,k);S=n0*np.array([.45,2,.6,2,2])
def capacity(B,S):
 def fj(n):
  p,D,V,A=objects(n,B,a,out,theta);C=B@np.log(out+a@n**theta)
  return -C+theta*(k@n),-theta*(D/n-k)
 x=minimize(fj,np.minimum(n0,S)*.9,jac=True,bounds=[(1e-8,s) for s in S],method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-10,'maxiter':2000,'maxls':50})
 n=x.x;p,D,V,A=objects(n,B,a,out,theta);gap=D/n-k;full=np.isclose(n,S,atol=1e-6);assert np.all(gap[full]>-1e-6) and np.max(abs(gap[~full]))<1e-6
 return n,full,gap
nc,full,gap=capacity(B,S);nscale,_,_=capacity(2*B,2*S);assert np.max(abs(nscale-2*nc))<1e-4

# Transparent hypothetical city structures, all with nine zones.
xy=np.array([(x,y) for x in range(3) for y in range(3)]);dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
home=np.ones(9)/9;job1=np.full(9,.025);job1[4]=.8;job2=np.full(9,.04);job2[1]=job2[7]=.36;job3=np.array([.07,.13,.08,.15,.12,.1,.08,.16,.11]);job3/=job3.sum()
examples=[]
for label,job,tau in [('One workplace center',job1,1.2),('Two workplace centers',job2,.9),('Dispersed workplaces',job3,.45)]:
 B=100*(.5*home+.5*job);dB=8*(home-job);a=np.exp(-tau*dist);k=np.ones(9);theta=.65
 for opened in [False,True]:
  out=(.5+np.linalg.norm(xy-[1,1],axis=1))*1.2 if opened else np.zeros(9)
  n=solve(B,a,out,theta,k);nn=solve(B+dB,a,out,theta,k)
  examples.append({'structure':label,'outside_option':opened,'baseline_entry':float(n.sum()),'change_in_total_entry':float((nn-n).sum()),'percent_total_change':float(100*(nn.sum()/n.sum()-1)),'zone_entry_changes':(nn-n).tolist()})
report={'interpretation':'Constructed model verification and hypothetical cities; no empirical observations or estimated city parameters.','configurations':count,'maximum_derivative_error':err,'maximum_equilibrium_residual':res,'minimum_A_eigenvalue':mineig,'maximum_scale_error':scaleerr,'maximum_spectral_formula_error':spectralerr,'capacity_checks':{'full_destinations':int(full.sum()),'max_scale_error':float(np.max(abs(nscale-2*nc)))},'city_examples':examples,'passed':True}
(R/'general_theory_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='city_examples'},indent=2))
