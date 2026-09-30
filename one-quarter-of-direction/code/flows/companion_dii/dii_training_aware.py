"""Training-aware linearized nonoverlapping bootstrap for original DII.
Fixed finite-dimensional correctly specified mean models, fixed target kernels.
Common training path across components; independent common evaluation path.
See theorem for moments, train/evaluation gap, positive-definite design matrices.
"""
import numpy as np
from full_study import kernel,center
from dii_confidence import dyadic_block_length

def count_weights(rng,N,B):
 ell=dyadic_block_length(N)
 if N%ell:raise ValueError('Trim rows to a multiple of the dyadic block before fitting')
 return np.repeat(rng.multinomial(N//ell,np.ones(N//ell)/(N//ell),size=B),ell,axis=1)-1

def training_aware_dii(comparisons,bootstrap_draws=399,seed=20260917,include_training=True):
 """name -> pair of direction dictionaries with residual_train, design_train,
 residual_eval, design_eval, regressors_eval. Fits must use exactly these rows.
 No nuisance function is estimated inside this function; no bandwidth tuning.
 """
 B=bootstrap_draws;names=list(comparisons);d0=comparisons[names[0]][0]
 m=len(d0['residual_train']);n=len(d0['residual_eval']);rng=np.random.default_rng(seed)
 W=count_weights(rng,n,B);WT=count_weights(rng,m,B);result={}
 for name,directions in comparisons.items():
  out=[]
  for d in directions:
   u=np.asarray(d['residual_eval']);v=np.asarray(d['residual_train']);P=np.asarray(d['design_eval']);PT=np.asarray(d['design_train']);z=np.asarray(d['regressors_eval'])
   if len(u)!=n or len(v)!=m or P.shape[0]!=n or PT.shape[0]!=m or P.shape[1]!=PT.shape[1]:raise ValueError('Calendar/design mismatch')
   Q=PT.T@PT/m
   if np.linalg.matrix_rank(Q)<Q.shape[0]:raise ValueError('Singular design; remove redundant columns')
   db=(WT@(PT*v[:,None])/m)@np.linalg.inv(Q)
   V=db@P.T if include_training else np.zeros((B,n))
   K=kernel(u);Lc=center(kernel(z));A=center(K)*Lc
   delta=u[:,None]-u[None,:]
   E=delta*K;E-=E.mean(0)[None,:]
   C=E*Lc;F=(1-delta*delta)*K*Lc
   H=float(A.sum()/n**2)
   quadratic=(np.sum((W@A)*W,axis=1)-2*np.sum((W@C)*V,axis=1)+np.sum((V@F)*V,axis=1))/n**2
   linear=2*(W@A.sum(0)-V@C.sum(0))/n**2
   out.append((H,quadratic,linear))
  f,b=out;D=b[0]-f[0];q=b[1]-f[1];l=b[2]-f[2]
  pq=(1+np.count_nonzero(q>=D-1e-12))/(B+1);pl=(1+np.count_nonzero(l>=D-1e-12))/(B+1)
  result[name]={'forward_hsic':f[0],'reverse_hsic':b[0],'dii':D,'p_quadratic':float(pq),'p_linear':float(pl),'p_intersection':float(max(pq,pl))}
 return {'m':m,'n':n,'seed':seed,'draws':B,'include_training':include_training,'comparisons':result}
