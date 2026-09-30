from pathlib import Path
from fractions import Fraction
import numpy as np,json
from dii_joint_nuisance import prepare_direction
from dii_moment_benchmark import moment_components
rng=np.random.default_rng(717);m=80;n=24;z=rng.normal(size=(m+n,3));y=z[:,0]+rng.laplace(scale=1/np.sqrt(2),size=m+n)
def basis(z):return np.column_stack([np.ones(len(z)),z])
d=prepare_direction(y,z,m,np.arange(m,m+n),basis);value,g,grad=moment_components(d);v=rng.normal(size=m);v-=v.mean();eps=1e-5

def refit(sign):
 w=1+sign*eps*v;my=np.average(y[:m],weights=w);sy=np.sqrt(np.average((y[:m]-my)**2,weights=w));mz=np.average(z[:m],axis=0,weights=w);sz=np.sqrt(np.average((z[:m]-mz)**2,axis=0,weights=w));zz=(z-mz)/sz;yy=(y-my)/sy;P=basis(zz)
 beta=np.linalg.solve(P[:m].T@(w[:,None]*P[:m]),P[:m].T@(w*yy[:m]));u=(yy-P@beta)[m:];q=zz[m:,0];return np.mean((u*u-np.mean(u*u))*(q*q-np.mean(q*q)))
refit_der=(refit(1)-refit(-1))/(2*eps);analytic=(v@d['score']/m)@grad;err=float(abs(refit_der-analytic));assert err<1e-7,err
m4=Fraction(2,7)*6+Fraction(5,7)*Fraction(9,5);m6=Fraction(2,7)*90+Fraction(5,7)*Fraction(27,7);witness=(m6-3*m4-6)/2
assert m4==3 and m6==Fraction(1395,49) and witness==Fraction(330,49)
out=dict(moment_full_weighted_refit_derivative_error=err,mixture_fourth_moment=str(m4),mixture_sixth_moment=str(m6),dependence_witness=str(witness))
(Path(__file__).resolve().parent.parent/'decision_use_verification.json').write_text(json.dumps(out,indent=2));print(out)
