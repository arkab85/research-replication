"""Simple residual moment comparator, with the same joint training influence.
Tests Cov(u^2,z_focal^2)>0, not DII or full residual independence.
"""
import numpy as np
from dii_training_aware import count_weights

def moment_components(d):
 u=d['u'];z=d['z'][:,0];P=d['P'];p=d['p'];k=d['z'].shape[1];n=len(u);m=len(d['score']);q=d['score'].shape[1]
 a=u*u-np.mean(u*u);b=z*z-np.mean(z*z);value=float(np.mean(a*b));g=a*b-value
 du=np.zeros((n,q));du[:,:p]=-P;du[:,p]=-u
 dz=np.zeros((n,q));dz[:,p+1]=-z;dz[:,p+1+k]=-1
 grad=np.mean(2*u[:,None]*du*b[:,None]+2*z[:,None]*dz*a[:,None],axis=0)
 return value,g,grad

def moment_diagnostic(d,draws=399,seed=20260919):
 value,g,grad=moment_components(d);n=len(g);m=len(d['score'])
 rng=np.random.default_rng(seed);W=count_weights(rng,n,draws);WT=count_weights(rng,m,draws)
 boot=W@g/n+(WT@d['score']/m)@grad
 return dict(moment=value,p=(1+int(np.count_nonzero(boot>=value-1e-12)))/(draws+1))
