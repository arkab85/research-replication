import numpy as np,json
from numpy.polynomial.hermite import hermgauss
from pathlib import Path
from dii_orthogonal import corrected_gram
rng=np.random.default_rng(731)
u=rng.normal(size=12);z=rng.normal(size=(12,2));v=rng.normal(size=19);zt=rng.normal(size=(19,2))
K=corrected_gram(u,z,v,zt);mineig=np.linalg.eigvalsh(K).min();assert mineig>-1e-10
# Independent feature-space check: Gaussian spectral representation integrated
# with 100 Gauss-Hermite frequency nodes. No approximation in main estimator.
nodes,weights=hermgauss(100);omega=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
def f(x):return np.concatenate([np.cos(x[:,None]*omega)*np.sqrt(weights),np.sin(x[:,None]*omega)*np.sqrt(weights)],axis=1)
def fp(x):return np.concatenate([-np.sin(x[:,None]*omega)*omega*np.sqrt(weights),np.cos(x[:,None]*omega)*omega*np.sqrt(weights)],axis=1)
d=((z[:,None,:]-zt[None,:,:])**2).sum(2);bandwidth=len(v)**(-1/(z.shape[1]+4));w=np.exp(-d/(2*bandwidth**2));w/=w.sum(1)[:,None]
T=f(u)-u[:,None]*(w@fp(v));err=np.max(np.abs(K-T@T.T));assert err<1e-10
# Population Taylor audit in a finite smooth feature map, integrated over
# independent Gaussian X and epsilon. This checks the sign/order of correction.
n,w=hermgauss(35);x=n*np.sqrt(2);w=w/np.sqrt(np.pi)
X,E=np.meshgrid(x,x,indexing='ij');X=X.ravel();E=E.ravel();ww=np.outer(w,w).ravel()
psi=np.column_stack([np.sin(X),np.cos(X)]);psi-=ww@psi
phi=lambda u:np.column_stack([np.sin(u),np.cos(u)])
g=np.array([np.exp(-.5),0.])
rows=[]
for delta in [.2,.1,.05,.025,.0125]:
 residual=E-delta*X
 raw=np.einsum('n,ni,nj->ij',ww,phi(residual),psi)
 ort=np.einsum('n,ni,nj->ij',ww,phi(residual)-residual[:,None]*g,psi)
 rows.append({'delta':delta,'raw_operator_norm':float(np.linalg.norm(raw)),'orthogonal_operator_norm':float(np.linalg.norm(ort))})
assert rows[-1]['orthogonal_operator_norm']/rows[-2]['orthogonal_operator_norm']<.27
out={'minimum_gram_eigenvalue':float(mineig),'exact_gram_vs_spectral_quadrature_max_error':float(err),'population_bias_audit':rows}
(Path(__file__).resolve().parent.parent/'orthogonal_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
