"""Deterministic checks of the numerical certification and projection code."""
import numpy as np,json
from certified_revision import op_inner,stats,rot,CURV,elasticity_cells,TH,OUT
from svarcore import whiten,op_norms,op_diff_norms
rng=np.random.default_rng(260929);z=rng.normal(size=(80,2));z-=z.mean(0);L=np.linalg.cholesky(z.T@z/len(z));z=np.linalg.solve(L,z.T).T
# Exact identity and signed-permutation invariance.
assert np.max(op_diff_norms(z,z,TH))<1e-7
assert np.max(abs(op_norms(z,TH)-op_norms(z,TH+np.pi/2)))<1e-12
# Interpolation inequality for an exactly standardized finite population.
worst=0
for a,b in zip(TH[:-1],TH[1:]):
 E=z@rot(a);F=z@rot(b);aa=op_inner(E,E);bb=op_inner(F,F);ab=op_inner(E,F)
 for t in (.2,.5,.8):
  G=z@rot((1-t)*a+t*b)
  d2=op_inner(G,G)+(1-t)**2*aa+t*t*bb+2*t*(1-t)*ab-2*((1-t)*op_inner(G,E)+t*op_inner(G,F))
  actual=np.sqrt(max(d2,0));bound=CURV*(b-a)**2/8;worst=max(worst,actual/bound);assert actual<=bound+1e-10
# Outer economic projection must contain every sampled positive ratio.
L=np.array([[.8,0],[-.2,1.3]]);r=.2;bounds=elasticity_cells(L,r)
for _ in range(2000):
 j=int(rng.integers(30));t=rng.uniform(TH[j],TH[j+1]);h=np.tril(rng.normal(size=(2,2)));h*=r*rng.uniform()/np.linalg.norm(h);A=L@(np.eye(2)+h)@rot(t)
 for k in range(2):
  if A[0,k]*A[1,k]>0:
   e=A[0,k]/A[1,k];assert bounds[j,0]-1e-12<=e<=bounds[j,1]+1e-12
# Projection-defined two-point volatility budget.
for d in (.01,.4,.8):
 sig=np.array([1-d,1+d])/np.sqrt(1+d*d);assert abs(np.var(sig)-d*d/(1+d*d))<1e-14
result={'exact_operator_identity':'pass','signed_permutation_invariance':'pass','interpolation_finite_population':'pass','largest_interpolation_error_over_bound':worst,'economic_projection_2000_draws':'pass','projection_budget_identity':'pass'}
(OUT/'validation.json').write_text(json.dumps(result,indent=2));print(result)
# Regenerate the first retained bootstrap draw in each empirical specification.
from certified_revision import spec,init,draw
replication=[]
for name in ('oil','equity','oil_may','equity_top1','equity_top5','equity_crisis'):
    data=np.load(OUT/(name+'.npz'));init(spec(name));actual=draw(data['seeds'][0])
    for key,v in zip(('D','DR','R','CS','Lboot'),actual):
        assert np.allclose(v,data[key][0],rtol=1e-10,atol=1e-12),(name,key)
    replication.append(name)
result['first_bootstrap_draw_reproduced']=replication
(OUT/'validation.json').write_text(json.dumps(result,indent=2));print('All six first draws reproduced.')
