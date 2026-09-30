"""DII engine, full-vector version: rho_f = HSIC(eps_f, Z^f), rho_b = HSIC(eps_b, Z^b), with
Z^f = (X_t,...,X_{t-p+1}, Y_t,...,Y_{t-q+1}) and Z^b = (Y_{t+h},...,Y_{t+h-p'+1}, X_{t-1},...,X_{t-q'}),
exactly as in the paper's equations (3)-(4). Gaussian kernels, median heuristic, doubly centered Gram matrices."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def cgram(Z):
    Z=np.asarray(Z,float); 
    if Z.ndim==1: Z=Z.reshape(-1,1)
    n=len(Z); sq=(Z**2).sum(1); d2=np.maximum(sq[:,None]+sq[None,:]-2*Z@Z.T,0.0)
    med=np.median(d2[d2>0]) if np.any(d2>0) else 1.0; K=np.exp(-d2/(2*(med/2.0)+1e-12))
    H=np.eye(n)-np.ones((n,n))/n; return H@K@H
def hsic2(a,B):
    n=len(a); return float(np.sum(cgram(a)*cgram(B)))/(n-1)**2
def sieve(Z): return np.hstack([Z,Z**2,np.sign(Z)*np.abs(Z)**3,np.tanh(Z),np.tanh(2*Z)])

def build(X,Y,h,p,q=None,**kw):
    """Design rows for t. Common conditioning set C_t=(Y_t, X_{t-1..t-p}, Y_{t-1..t-q}).
    Forward: target Y_{t+h}, regressors Zf=(X_t, C_t). Backward: target X_t, regressors Zb=(Y_{t+h}, C_t)."""
    q=p if q is None else q; lo=max(p,q); rows=[]
    for t in range(lo, len(X)-h):
        C=[Y[t]]+list(X[t-p:t][::-1])+list(Y[t-q:t][::-1])
        rows.append([Y[t+h],X[t]]+[X[t]]+C+[Y[t+h]]+C)
    k=1+p+q; fcols=[f'Zf{i}' for i in range(1+k)]; bcols=[f'Zb{i}' for i in range(1+k)]
    d=pd.DataFrame(rows,columns=['Y_th','X_t']+fcols+bcols); d.attrs['f']=fcols; d.attrs['b']=bcols; return d

class Stage1:
    def fit(self,d,alpha=1.0):
        self.f=d.attrs['f']; self.b=d.attrs['b']
        self.scf=StandardScaler().fit(d[self.f].values); self.scb=StandardScaler().fit(d[self.b].values)
        self.mf=Ridge(alpha=alpha).fit(sieve(self.scf.transform(d[self.f].values)),d['Y_th'].values)
        self.mb=Ridge(alpha=alpha).fit(sieve(self.scb.transform(d[self.b].values)),d['X_t'].values); return self
    def resid(self,d):
        ef=d['Y_th'].values-self.mf.predict(sieve(self.scf.transform(d[self.f].values)))
        eb=d['X_t'].values-self.mb.predict(sieve(self.scb.transform(d[self.b].values))); return ef,eb
def dii(ef,eb,d): return hsic2(eb,d[d.attrs['b']].values)-hsic2(ef,d[d.attrs['f']].values)
def core_q(ef,eb,d):
    n=len(d); return (cgram(eb)*cgram(d[d.attrs['b']].values)-cgram(ef)*cgram(d[d.attrs['f']].values))*(n**2)/(n-1)**2
def wild_weights(n,ell,rng):
    a=np.exp(-1.0/ell); W=np.empty(n); W[0]=rng.normal()
    for t in range(1,n): W[t]=a*W[t-1]+np.sqrt(1-a*a)*rng.normal()
    return W
def wild_p(Q,n,B,rng,two=False):
    obs=Q.sum()/n**2; ell=max(2,int(round(n**(1/3)))); bo=np.array([(lambda W: W@Q@W/n**2)(wild_weights(n,ell,rng)) for _ in range(B)])
    pr=(np.sum(bo>=obs)+1)/(B+1); pl=(np.sum(bo<=obs)+1)/(B+1); return obs,pr,pl
def mbb_rows(n,ell,rng):
    starts=rng.randint(0,n-ell+1,size=int(np.ceil(n/ell))); return np.concatenate([np.arange(s,s+ell) for s in starts])[:n]
def sub(d,idx):
    e=d.iloc[idx].reset_index(drop=True); e.attrs=d.attrs; return e
def paired_p(s1,d,B,rng):
    ef,eb=s1.resid(d); obs=dii(ef,eb,d); n=len(d); ell=max(2,int(round(n**(1/3)))); bo=np.empty(B)
    for b in range(B):
        e=sub(d,mbb_rows(n,ell,rng)); e1,e2=s1.resid(e); bo[b]=dii(e1,e2,e)
    c=bo-obs; return obs,(np.sum(c>=obs)+1)/(B+1),(np.sum(c<=obs)+1)/(B+1)
def shift_p(s1,X_te,Y_te,h,p,B,rng,**kw):
    d=build(X_te,Y_te,h,p,**kw); ef,eb=s1.resid(d); obs=dii(ef,eb,d); bo=np.empty(B)
    for b in range(B):
        e=build(X_te,np.roll(Y_te,rng.randint(1,len(Y_te))),h,p,**kw); e1,e2=s1.resid(e); bo[b]=dii(e1,e2,e)
    return obs,(np.sum(bo>=obs)+1)/(B+1),(np.sum(bo<=obs)+1)/(B+1)
def honest(X,Y,h,p,rule='admissible',gap=None,**kw):
    T=len(X); gap=gap if gap is not None else 10
    n=int(np.floor(T/np.log(T))) if rule=='admissible' else (int(T**(2/3)) if rule=='t23' else int(0.25*T) if rule=='quarter' else int(0.5*T))
    tr=T-n-gap; dtr=build(X[:tr],Y[:tr],h,p,**kw); dte=build(X[tr+gap:],Y[tr+gap:],h,p,**kw); return Stage1().fit(dtr),dte,tr,gap,n
# ---- DGPs ----
def ar1(T,rng,phi=0.5):
    x=np.zeros(T)
    for t in range(1,T): x[t]=phi*x[t-1]+np.sqrt(1-phi**2)*rng.normal()
    return x
def dgp_null(T,rng): return ar1(T,rng),ar1(T,rng)
def dgp_lingauss(T,rng): X=ar1(T,rng); Y=np.zeros(T); Y[1:]=0.8*X[:-1]+rng.normal(0,1,T-1); return X,Y
def dgp_anm(T,rng,s=0.8): X=ar1(T,rng); Y=np.zeros(T); Y[1:]=s*X[:-1]**2+rng.normal(0,1,T-1); return X,Y
def dgp_anm_rev(T,rng,s=0.8): Y,X=dgp_anm(T,rng,s); return X,Y
def dgp_symconf(T,rng): U=ar1(T,rng,0.7); return U**2+0.5*rng.normal(size=T),U**2+0.5*rng.normal(size=T)
def dgp_conf(T,rng,phi=0.9):
    U=ar1(T,rng,phi); X=U+rng.normal(0,0.5,T); Y=np.zeros(T); Y[1:]=np.tanh(U[:-1])+rng.normal(0,0.3,T-1); return X,Y
# oracle conditional means for h=1,p=q=1: Zf=(X_t, Y_t, X_{t-1}, Y_{t-1}), Zb=(Y_{t+1}, Y_t, X_{t-1}, Y_{t-1})
def oracle_null(d): return d['Y_th'].values-0.5*d['Zf1'].values, d['X_t'].values-0.5*d['Zb2'].values
def oracle_lingauss(d):
    ef=d['Y_th'].values-0.8*d['Zf0'].values
    eb=d['X_t'].values-0.5*d['Zb2'].values-(0.6/1.48)*(d['Zb0'].values-0.4*d['Zb2'].values); return ef,eb

def dgp_symconf(T,rng,w=0.53):
    """Stratum-(c) design: latent AR(1) confounder U; X_t=U_t^2+e1, Y_t=w U_t^2+(1-w) U_{t-1}^2+e2, w calibrated so rho_f=rho_b>0 at h=1."""
    U=ar1(T,rng,0.7); U2=U**2; Ul=np.roll(U2,1); Ul[0]=U2[0]
    return U2+0.5*rng.normal(size=T), w*U2+(1-w)*Ul+0.5*rng.normal(size=T)

class Stage1LS(Stage1):
    """Location-scale first stage: sieve ridge for the mean, sieve ridge on log squared residuals for the variance."""
    def fit(self,d,alpha=1.0,floor=0.05):
        super().fit(d,alpha)
        Sf=sieve(self.scf.transform(d[self.f].values)); rf=d['Y_th'].values-self.mf.predict(Sf); self.vf=Ridge(alpha=alpha).fit(Sf,np.log(rf**2+1e-8)); self.flf=floor*np.var(rf)
        Sb=sieve(self.scb.transform(d[self.b].values)); rb=d['X_t'].values-self.mb.predict(Sb); self.vb=Ridge(alpha=alpha).fit(Sb,np.log(rb**2+1e-8)); self.flb=floor*np.var(rb); return self
    def resid(self,d):
        Sf=sieve(self.scf.transform(d[self.f].values)); Sb=sieve(self.scb.transform(d[self.b].values))
        ef=(d['Y_th'].values-self.mf.predict(Sf))/np.sqrt(np.maximum(np.exp(self.vf.predict(Sf)),self.flf))
        eb=(d['X_t'].values-self.mb.predict(Sb))/np.sqrt(np.maximum(np.exp(self.vb.predict(Sb)),self.flb)); return ef,eb
def dgp_lsnm(T,rng):
    X=ar1(T,rng); Y=np.zeros(T); g=0.4+1.2*np.abs(X[:-1])/(1+np.abs(X[:-1])); Y[1:]=0.8*X[:-1]**2+g*rng.normal(0,1,T-1); return X,Y

from sklearn.neural_network import MLPRegressor
class Stage1NN(Stage1):
    """Deep-network first stage (Corollary 3): two hidden layers, ReLU, early stopping."""
    def __init__(self,hidden=(32,16),seed=0): self.hidden=hidden; self.seed=seed
    def fit(self,d,**kw):
        self.f=d.attrs['f']; self.b=d.attrs['b']
        self.scf=StandardScaler().fit(d[self.f].values); self.scb=StandardScaler().fit(d[self.b].values)
        self.mf=MLPRegressor(hidden_layer_sizes=self.hidden,activation='relu',max_iter=500,early_stopping=True,n_iter_no_change=20,random_state=self.seed).fit(self.scf.transform(d[self.f].values),d['Y_th'].values)
        self.mb=MLPRegressor(hidden_layer_sizes=self.hidden,activation='relu',max_iter=500,early_stopping=True,n_iter_no_change=20,random_state=self.seed+1).fit(self.scb.transform(d[self.b].values),d['X_t'].values); return self
    def resid(self,d):
        return d['Y_th'].values-self.mf.predict(self.scf.transform(d[self.f].values)), d['X_t'].values-self.mb.predict(self.scb.transform(d[self.b].values))
