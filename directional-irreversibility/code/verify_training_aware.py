import numpy as np,json
from pathlib import Path
from numpy.polynomial.hermite import hermgauss
from full_study import kernel,center
rng=np.random.default_rng(912);n=18;u=rng.normal(size=n);z=rng.normal(size=n);w=rng.normal(size=n);w-=w.mean();v=rng.normal(size=n)*.05
nodes,weights=hermgauss(100);o=nodes*np.sqrt(2);s=np.sqrt(weights/np.sqrt(np.pi))
def f(x):return np.concatenate([np.cos(x[:,None]*o)*s,np.sin(x[:,None]*o)*s],axis=1)
def fp(x):return np.concatenate([-np.sin(x[:,None]*o)*o*s,np.cos(x[:,None]*o)*o*s],axis=1)
phi=f(u);phi-=phi.mean(0);psi=f(z);psi-=psi.mean(0);der=fp(u)
Aexp=np.einsum('ni,nj->nij',phi,psi);Dexp=np.einsum('ni,nj->nij',der,psi)
Cobs=Aexp.mean(0);Delta=(np.einsum('n,nij->ij',w,Aexp)-np.einsum('n,nij->ij',v,Dexp))/n
K=kernel(u);L=center(kernel(z));A=center(K)*L;diff=u[:,None]-u[None,:];E=diff*K;E-=E.mean(0)[None,:];C=E*L;F=(1-diff*diff)*K*L
q=(w@A@w-2*w@C@v+v@F@v)/n**2;l=2*(w@A.sum(0)-v@C.sum(0))/n**2
errq=abs(q-np.sum(Delta**2));errl=abs(l-2*np.sum(Cobs*Delta));assert max(errq,errl)<1e-11
out={'quadratic_vs_explicit_feature_error':float(errq),'linear_vs_explicit_feature_error':float(errl)}
(Path(__file__).resolve().parent.parent/'training_aware_verification.json').write_text(json.dumps(out,indent=2));print(out)
