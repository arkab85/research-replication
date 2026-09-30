from pathlib import Path
import numpy as np,json
from dii_joint_nuisance import prepare_direction,operator_matrices,joint_nuisance_dii
from dii_training_aware import training_aware_dii
from full_study import center
rng=np.random.default_rng(818);m=80;n=24;z=rng.normal(size=(m+n,2));y=.4*z[:,0]**2+rng.normal(size=m+n)
def basis(z):return np.column_stack([np.ones(len(z)),z,z[:,0]**2,z[:,1]**2])
d=prepare_direction(y,z,m,np.arange(m,m+n),basis);A,M,J=operator_matrices(d);q=J.shape[0];p=d['p']
def features(e):return (d['u']-d['P']@e[:p])*np.exp(-e[p]),(d['z']-e[p+1+d['z'].shape[1]:])*np.exp(-e[p+1:p+1+d['z'].shape[1]])
def cross(a,b):
 ua,za=features(a);ub,zb=features(b)
 K=np.exp(-.5*(ua[:,None]-ub[None,:])**2);L=np.exp(-.5*np.sum((za[:,None,:]-zb[None,:,:])**2,axis=2))
 return center(K)*center(L)
eps=1e-4;eye=np.eye(q)*eps;zero=np.zeros(q)
MM=np.column_stack([(cross(zero,e)-cross(zero,-e)).sum(1)/(2*eps*n) for e in eye])
JJ=np.array([[(cross(a,b).sum()-cross(a,-b).sum()-cross(-a,b).sum()+cross(-a,-b).sum())/(4*eps**2*n*n) for b in eye] for a in eye])
errM=float(np.max(abs(MM-M)));errJ=float(np.max(abs(JJ-J)))
assert errM<2e-7 and errJ<2e-7,(errM,errJ)
# Coefficient-only implementation must agree with the previous exact formula.
ty=(y-y[:m].mean())/y[:m].std();zz=(z-z[:m].mean(0))/z[:m].std(0);P=basis(zz);u=ty-P@np.linalg.lstsq(P[:m],ty[:m],rcond=None)[0]
do=dict(residual_train=u[:m],residual_eval=u[m:],design_train=P[:m],design_eval=P[m:],regressors_eval=zz[m:])
a=joint_nuisance_dii({'h':[d,d]},99,88,False);b=training_aware_dii({'h':[do,do]},99,88)
assert a['comparisons']['h']==b['comparisons']['h'] or all(abs(a['comparisons']['h'][k]-v)<1e-12 for k,v in b['comparisons']['h'].items())
# Use a distinct reverse direction as a nonzero-contrast check.
dr=prepare_direction(z[:,0],np.column_stack([y,z[:,1]]),m,np.arange(m,m+n),basis)
rr=(z[:,0]-z[:m,0].mean())/z[:m,0].std();zr=np.column_stack([y,z[:,1]]);zr=(zr-zr[:m].mean(0))/zr[:m].std(0);Pr=basis(zr);ur=rr-Pr@np.linalg.lstsq(Pr[:m],rr[:m],rcond=None)[0]
dor=dict(residual_train=ur[:m],residual_eval=ur[m:],design_train=Pr[:m],design_eval=Pr[m:],regressors_eval=zr[m:])
a=joint_nuisance_dii({'h':[d,dr]},399,88,False)['comparisons']['h'];b=training_aware_dii({'h':[do,dor]},399,88)['comparisons']['h']
assert all(abs(a[k]-v)<1e-12 for k,v in b.items())
r=dict(cross_operator_derivative_max_error=errM,mixed_derivative_max_error=errJ,coefficient_only_agreement=True,finite_difference_step=eps)

# Full weighted refit: independently recompute OLS, means and scales and compare
# its operator derivative with the joint score/Jacobian implementation.
v=rng.normal(size=m);v-=v.mean();ep=1e-4
def refit(sign):
 w=1+sign*ep*v
 my=np.average(y[:m],weights=w);sy=np.sqrt(np.average((y[:m]-my)**2,weights=w))
 mz=np.average(z[:m],axis=0,weights=w);sz=np.sqrt(np.average((z[:m]-mz)**2,axis=0,weights=w))
 yy=(y-my)/sy;zz=(z-mz)/sz;pp=basis(zz)
 beta=np.linalg.solve(pp[:m].T@(w[:,None]*pp[:m]),pp[:m].T@(w*yy[:m]))
 return (yy-pp@beta)[m:],zz[m:]
def inner_feat(a,b):
 ua,za=a;ub,zb=b
 return np.sum(center(np.exp(-.5*(ua[:,None]-ub[None,:])**2))*center(np.exp(-.5*np.sum((za[:,None,:]-zb[None,:,:])**2,axis=2))))/n**2
plus=refit(1);minus=refit(-1);direction=v@d['score']/m
refit_norm=(inner_feat(plus,plus)+inner_feat(minus,minus)-2*inner_feat(plus,minus))/(4*ep**2)
analytic_norm=direction@J@direction
err_refit=abs(refit_norm-analytic_norm);assert err_refit<1e-7,err_refit
from dii_joint_nuisance import certify_original_dii
example={'joint_radius95':.01,'comparisons':{'h':dict(forward_hsic=.01,reverse_hsic=.09)}}
assert not certify_original_dii(example)['h']['positive_original_dii']
assert certify_original_dii(example,{'h':(0,0)})['h']['positive_original_dii']
assert not certify_original_dii(example,{'h':(.1,.1)})['h']['positive_original_dii']
r.update(full_weighted_refit_norm_error=float(err_refit),missing_approximation_bound_withholds_original_claim=True)
(Path(__file__).resolve().parent.parent/'joint_nuisance_verification.json').write_text(json.dumps(r,indent=2));print(r)
