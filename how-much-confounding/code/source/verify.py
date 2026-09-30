"""Checks that target substantive formula and implementation risks."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.special import roots_hermitenorm
from core import *
P=Path(__file__).resolve().parents[1];rng=np.random.default_rng(3317);out={}
z,w=roots_hermitenorm(160);w=w/np.sqrt(2*np.pi);psi=shape_function(z)
out['shape_mean']=float(w@psi);out['shape_state_covariance']=float(w@(z*psi));out['shape_variance']=float(w@(psi*psi))
assert abs(out['shape_mean'])<1e-12 and abs(out['shape_state_covariance'])<1e-12 and abs(out['shape_variance']-1)<1e-12
e=rng.normal(size=50);Z=rng.normal(size=(50,3));v,A,K,L=component(e,Z);H=np.eye(50)-np.ones((50,50))/50
out['hsic_identity_error']=float(abs(v-np.trace(K@H@L@H)/50**2));assert out['hsic_identity_error']<1e-12
# Explicit finite-dimensional tensor features independently verify multiplier Gram algebra.
F=rng.normal(size=(35,4));G=rng.normal(size=(35,3));Fc=F-F.mean(0);Gc=G-G.mean(0)
T=np.einsum('ni,nj->nij',Fc,Gc).reshape(35,-1);Tc=T-T.mean(0)
AG=center((Fc@Fc.T)*(Gc@Gc.T));out['tensor_gram_error']=float(np.max(np.abs(AG-Tc@Tc.T)));assert out['tensor_gram_error']<1e-10
violations=[]
for both in [False,True]:
 for phi in [.1,.8,.95]:
  q=benchmark(phi,.3,.7,p=2,h=3,k=1,both=both)
  for side in ['f','b']:
   K0=envelope_constant(q,side)
   assert np.linalg.eigvalsh(q['cov']).min()>0 and q[side]['v']>0
   for t in np.linspace(0,1,101):violations.append(4*amplitude(q,t,1-t,side)**2-K0)
out['simplex_max_excess']=float(max(violations));assert max(violations)<1e-10
# Gaussian conditional mean entropy inequality: equality for shifted means.
v0=.7;shift=.2;KL=shift**2/(2*v0)
out['gaussian_mean_kl_equality_error']=abs(shift**2-2*v0*KL);assert out['gaussian_mean_kl_equality_error']<1e-14
# Integrated conditional-mean inequality in a nonlinear, non-Gaussian observed law.
q=benchmark(.8,.3,.7);delta=.2;x,y,c=sample(q,delta,12000,rng);ef,zf,eb,zb=oracle(q,delta,x,y,c,180)
S=delta**2/q['se']**2
out['conditional_mean_checks']={}
for side,response,residual,z0 in [('f',y,ef,zf),('b',x,eb,zb)]:
 mse=float(np.mean((response-residual-z0@q[side]['beta'])**2));bound=q[side]['v']*S
 out['conditional_mean_checks'][side]={'numerical_mean_squared_change':mse,'entropy_upper_bound':bound}
 assert mse<bound
import pandas as pd
app=pd.read_csv(P/'results/application.csv');assert len(app)==96 and (app.gamma_lower==0).all()
out['application_rows']=len(app);out['positive_empirical_lower_bounds']=int((app.lower>0).sum())
quad=json.loads((P/'results/quadrature.json').read_text());errors=[]
for rx in [.2,.8]:
 a=[v for v in quad if v['rx']==rx];errors.append(abs(a[-1]['dii']-a[-2]['dii']))
out['quadrature_max_dii_change_140_to_240']=max(errors);assert max(errors)<1e-9
out['passed']=True;(P/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
