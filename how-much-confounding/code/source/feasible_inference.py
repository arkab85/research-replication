"""Feasible joint operator inference for estimated regression coefficients.
Confidence calibration is asymptotic under the stated projection, moment,
and independent-path/mixing assumptions. Specification error remains separate.
"""
import numpy as np
from scipy.stats import chi2
from core import gram,center,component,radius,interval

def derivative_geometry(e,z,degree=2,features=None):
    """Gram of J and inner products <T_i-C,J>; intercept removed by invariance."""
    n=len(e);P=np.vander(z,degree+1,increasing=True)[:,1:] if features is None else features
    K=gram(e);Lc=center(gram(z));diff=e[:,None]-e[None,:]
    mixed=(1-diff**2)*K
    G=P.T@(mixed*Lc)@P/n**2
    D=diff*K
    B=-((D-D.mean(axis=0,keepdims=True))*Lc)@P/n
    B-=B.mean(axis=0,keepdims=True)
    return (G+G.T)/2,B

def joint_training_covariance(xt,yt,degree=2,dependent=False):
    n=len(xt);betas=[];scores=[]
    for z,v in [(xt,yt),(yt,xt)]:
        P=np.vander(z,degree+1,increasing=True);inv=np.linalg.inv(P.T@P)
        b=inv@P.T@v;res=v-P@b;h=np.einsum('ij,jk,ik->i',P,inv,P)
        # HC3 finite-sample leverage adjustment; asymptotically equivalent to OLS IF.
        scores.append((P*(res/np.maximum(1-h,1e-8))[:,None])@inv[:,1:]);betas.append(b)
    S=np.concatenate(scores,axis=1);V=S.T@S
    lag=int(np.floor(4*(n/100)**(2/9))) if dependent else 0
    for j in range(1,lag+1):
        a=S[j:].T@S[:-j];V+=(1-j/(lag+1))*(a+a.T)
    return betas,(V+V.T)/2,lag

def joint_analyze(xt,yt,x,y,degree=2,dependent=False,bootstrap=399,seed=1):
    bs,V,lag=joint_training_covariance(xt,yt,degree,dependent)
    es=[v-np.vander(z,degree+1,increasing=True)@b for z,v,b in [(x,y,bs[0]),(y,x,bs[1])]]
    comps=[component(e,z) for e,z in zip(es,[x,y])]
    n=len(x);ell=max(1,round(n**(1/3)));starts=np.arange(0,n,ell)
    rng=np.random.default_rng(seed);w=rng.normal(size=(bootstrap,len(starts)))
    ev,vec=np.linalg.eigh(V);root=(vec*np.sqrt(np.maximum(ev,0)))@vec.T
    db=rng.normal(size=(bootstrap,2*degree))@root.T
    mx=np.zeros(bootstrap);mx0=np.zeros(bootstrap);terms=[]
    for k,(e,z,comp) in enumerate(zip(es,[x,y],comps)):
        G,B=derivative_geometry(e,z,degree)
        AC=center(comp[1]);block=np.add.reduceat(np.add.reduceat(AC,starts,axis=0),starts,axis=1)
        BC=np.add.reduceat(B,starts,axis=0)
        v=db[:,k*degree:(k+1)*degree]
        samp=np.einsum('bi,ij,bj->b',w,block,w,optimize=True)/n**2
        param=np.einsum('bi,ij,bj->b',v,G,v,optimize=True)
        cross=2*np.einsum('bi,ij,bj->b',w,BC,v,optimize=True)/n
        mx=np.maximum(mx,np.sqrt(np.maximum(0,samp+param+cross)))
        mx0=np.maximum(mx0,np.sqrt(np.maximum(0,samp)))
        terms.append(float(np.trace(G@V[k*degree:(k+1)*degree,k*degree:(k+1)*degree])))
    q=float(np.quantile(mx,.95,method='higher'));q0=float(np.quantile(mx0,.95,method='higher'))
    hf,hb=[c[0] for c in comps];lo,hi=interval(hf,hb,q);l0,u0=interval(hf,hb,q0)
    # A feasible asymptotic RMS bound under the specified conditional-mean model.
    # This conservative comparator uses a .01 training / .04 sampling allocation.
    rhat=[]
    for k,z in enumerate([x,y]):
        P=np.vander(z,degree+1,increasing=True)[:,1:];M=P.T@P/n
        evm,vm=np.linalg.eigh(M);rm=(vm*np.sqrt(np.maximum(evm,0)))@vm.T
        W=V[k*degree:(k+1)*degree,k*degree:(k+1)*degree]
        rhat.append(float(np.sqrt(chi2.ppf(.995,degree)*max(0,np.linalg.eigvalsh(rm@W@rm).max()))*(1+n**(-.25))))
    q04=float(np.quantile(mx0,.96,method='higher'));lc,uc=interval(hf,hb,q04,*rhat)
    return dict(Hf=hf,Hb=hb,D_hat=hb-hf,q=q,q_ignore=q0,lower=lo,upper=hi,
        lower_ignore=l0,upper_ignore=u0,lower_feasible_r=lc,upper_feasible_r=uc,
        feasible_r_f=rhat[0],feasible_r_b=rhat[1],variance_training_f=terms[0],variance_training_b=terms[1],
        lag=lag,block=ell),bs
