"""Original Gaussian-kernel DII: joint coefficient and training-scale bootstrap.
Fixed polynomial bases. Common block draws retain all within-period covariance.
The conditional-mean target requires correct specification; see sensitivity result.
"""
import numpy as np
from full_study import kernel,center
from dii_training_aware import count_weights

def prepare_direction(target,regressors,m,te,basis):
 target=np.asarray(target);z=np.asarray(regressors)
 ty=(target-target[:m].mean())/target[:m].std()
 zz=(z-z[:m].mean(0))/z[:m].std(0)
 P=basis(zz); beta=np.linalg.lstsq(P[:m],ty[:m],rcond=None)[0]
 u=ty-P@beta; Q=P[:m].T@P[:m]/m
 if np.linalg.matrix_rank(Q)!=Q.shape[0]:raise ValueError('Singular design')
 score_beta=(P[:m]*u[:m,None])@np.linalg.inv(Q)
 score_scale=np.column_stack([.5*(ty[:m]**2-1),.5*(zz[:m]**2-1)])
 return dict(u=u[te],z=zz[te],P=P[te],score=np.column_stack([score_beta,score_scale,zz[:m]]),p=P.shape[1])

def operator_matrices(d):
 u=d['u'];z=d['z'];P=d['P'];n=len(u);p=P.shape[1];k=z.shape[1];q=p+1+2*k
 K=kernel(u);L=kernel(z);Kc=center(K);Lc=center(L);A=Kc*Lc
 delta=u[:,None]-u[None,:];E=delta*K;Ec=E-E.mean(0)
 # Each column is a derivative of the standardized residual with respect to
 # a regression coefficient or the logarithm of its training target scale.
 du=np.column_stack([-P,-u]);M=np.zeros((n,q));J=np.zeros((q,q))
 M[:,:p+1]=(Ec*Lc)@du/n
 J[:p+1,:p+1]=du.T@(((1-delta**2)*K)*Lc)@du/n**2
 Efirst=-E;Efirst-=Efirst.mean(1)[:,None]
 dz=z[:,None,:]-z[None,:,:]
 # Include learned regressor centers: kernel distances are translation-invariant,
 # but covariance-operator uncertainty lives in a common feature-space frame.
 zder=[(r,-z[:,r]) for r in range(k)]+[(r,-np.ones(n)) for r in range(k)]
 for r,(axis,g) in enumerate(zder):
  j=p+1+r
  Lder=L*dz[:,:,axis]*g[None,:]
  Lder-=Lder.mean(0)
  M[:,j]=(Kc*Lder).sum(1)/n
  cross=du.T@(Efirst*Lder).sum(1)/n**2
  J[:p+1,j]=cross;J[j,:p+1]=cross
  for ss,(axis_s,g_s) in enumerate(zder):
   inner=(g[:,None]*g_s[None,:])*(float(axis==axis_s)-dz[:,:,axis]*dz[:,:,axis_s])
   J[j,p+1+ss]=np.sum(Kc*L*inner)/n**2
 return A,M,(J+J.T)/2

def joint_nuisance_dii(comparisons,bootstrap_draws=399,seed=20260918,include_scales=True,return_bounds=False):
 names=list(comparisons);d0=comparisons[names[0]][0];n=len(d0['u']);m=len(d0['score']);B=bootstrap_draws
 rng=np.random.default_rng(seed);W=count_weights(rng,n,B);WT=count_weights(rng,m,B)
 result={};norms=[]
 for name,directions in comparisons.items():
  out=[]
  for d in directions:
   A,M,J=operator_matrices(d);score=d['score'].copy()
   if not include_scales:score[:,d['p']:]=0
   db=WT@score/m
   H=float(A.sum()/n**2)
   quad=np.sum((W@A)*W,axis=1)/n**2+2*np.sum((W@M)*db,axis=1)/n+np.sum((db@J)*db,axis=1)
   if quad.min() < -1e-9:raise ArithmeticError('Negative squared operator norm')
   quad=np.maximum(quad,0)
   lin=2*(W@A.sum(0)/n**2+db@M.mean(0))
   out.append((H,quad,lin));norms.append(np.sqrt(quad))
  f,b=out;D=b[0]-f[0];pq=(1+np.count_nonzero(b[1]-f[1]>=D-1e-12))/(B+1);pl=(1+np.count_nonzero(b[2]-f[2]>=D-1e-12))/(B+1)
  result[name]=dict(forward_hsic=f[0],reverse_hsic=b[0],dii=D,p_quadratic=float(pq),p_linear=float(pl),p_intersection=float(max(pq,pl)))
 output=dict(m=m,n=n,draws=B,seed=seed,include_scales=include_scales,comparisons=result)
 if return_bounds:
  # Joint 95% norm radius, common across every direction and comparison.
  radius=float(np.quantile(np.max(norms,axis=0),.95,method='higher'));output['joint_radius95']=radius
  for v in result.values():
   lf=max(0,np.sqrt(max(0,v['forward_hsic']))-radius);uf=np.sqrt(max(0,v['forward_hsic']))+radius
   lb=max(0,np.sqrt(max(0,v['reverse_hsic']))-radius);ub=np.sqrt(max(0,v['reverse_hsic']))+radius
   v.update(forward_lower95=lf**2,forward_upper95=uf**2,reverse_lower95=lb**2,reverse_upper95=ub**2,dii_lower95=lb**2-uf**2,dii_upper95=ub**2-lf**2,
            common_mean_error_radius=max(0,(np.sqrt(max(0,v['reverse_hsic']))-np.sqrt(max(0,v['forward_hsic']))-2*radius)/4))
 return output

def certify_original_dii(output,mean_error_bounds=None):
 """Conditional 95% sign certificates for original conditional-mean DII.
 Bounds are externally justified L2 errors in population-standardized outcome
 units, not fitted residual RMSE. Missing bounds never imply zero error.
 They must hold jointly; any probability of their failure adds to noncoverage.
 """
 if 'joint_radius95' not in output:raise ValueError('Run with return_bounds=True')
 t=output['joint_radius95'];cert={}
 for name,v in output['comparisons'].items():
  if mean_error_bounds is None or name not in mean_error_bounds:
   cert[name]={'status':'unverified_mean_approximation','positive_original_dii':False};continue
  rf,rb=map(float,mean_error_bounds[name])
  if not np.isfinite([rf,rb]).all() or min(rf,rb)<0:raise ValueError('Invalid mean error bounds')
  lower=max(0,np.sqrt(max(0,v['reverse_hsic']))-t-2*rb)**2-(np.sqrt(max(0,v['forward_hsic']))+t+2*rf)**2
  cert[name]=dict(status='conditional_on_supplied_joint_error_bounds',forward_error_bound=rf,reverse_error_bound=rb,dii_lower95=lower,positive_original_dii=bool(lower>0))
 return cert
