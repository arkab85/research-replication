"""Numerical checks of derived formulas; these are not empirical observations."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq, minimize_scalar
R=Path(__file__).resolve().parent

def equilibrium(B1,B2,k,a,theta):
    N=(B1+B2)/k;w=B1/(B1+B2)
    def equation(m):
        p=w*expit(theta*m+np.log(a))+(1-w)*expit(theta*m-np.log(a))
        return m-logit(p)
    # Finite symmetric bracket avoids numerical saturation in logit.
    m=brentq(equation,-30,30,xtol=1e-13)
    n1=N*expit(m);n2=N-n1
    p=expit(theta*m+np.log(a));q=expit(theta*m-np.log(a))
    H=B1*p*(1-p)+B2*q*(1-q)
    lam=theta*H*N/(k*n1*n2)
    derivative=(p-q)/(k*(1-lam))
    return n1,n2,p,q,lam,derivative

errors=[];residuals=[];gains=[]
for theta in [.1,.4,.7,.9]:
 for a in [1.,1.2,3.,10.]:
  for share in [.15,.3,.5,.8]:
   b=100*share;k=1.;v=equilibrium(b,100-b,k,a,theta)
   h=1e-3
   fd=(equilibrium(b+h,100-b-h,k,a,theta)[0]-equilibrium(b-h,100-b+h,k,a,theta)[0])/(2*h)
   errors.append(abs(fd-v[-1]));residuals.append(abs(v[0]-(b*v[2]+(100-b)*v[3])))
   assert -1e-12<=v[4]<=theta+1e-12
   assert abs(v[0]+v[1]-100)<1e-10
   # Conditional consumer accessibility is maximized at the free-entry allocation.
   def C(n):
    return b*np.log(n**theta+(100-n)**theta/a)+(100-b)*np.log(n**theta/a+(100-n)**theta)
   opt=minimize_scalar(lambda n:-C(n),bounds=(1e-7,100-1e-7),method='bounded',options={'xatol':1e-8})
   gains.append(abs(C(v[0])-C(opt.x)))
   if share==.5:
    G=(a*a-1)/((a+1)**2-4*theta*a)
    assert abs(G-v[-1])<1e-9
assert max(errors)<1e-7 and max(residuals)<1e-8 and max(gains)<1e-7
peaks=[]
for theta in [.6,.75,.9]:
 h=2*theta-1;astar=(1+np.sqrt(1-h*h))/h
 G=lambda a:(a*a-1)/((a+1)**2-4*theta*a)
 assert G(astar)>G(astar*.95) and G(astar)>G(astar*1.05)
 peaks.append({'theta':theta,'a_star':astar,'gain':G(astar)})
report={'interpretation':'Analytical formula verification on constructed model parameters; no empirical data and no calibration.', 'parameter_configurations':len(errors),'max_derivative_error':max(errors),'max_equilibrium_residual':max(residuals),'max_conditional_accessibility_objective_gap':max(gains),'interior_mobility_peaks':peaks,'formulas_pass':True}
(R/'theory_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
