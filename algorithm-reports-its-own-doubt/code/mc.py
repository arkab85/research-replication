"""Monte Carlo study calibrated to the application. Compares oracle, naive, homoskedastic regression
calibration (RC), interval-calibrated RC (ICRC, proposed) and interval-calibrated SIMEX (ICSIMEX)."""
import numpy as np, json, sys
from scipy.stats import norm
from prep import load
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)]
S_EMP=pl.s.values; H_EMP=pl.hiconf.values
N=len(S_EMP); C2=0.195; SM2=0.0208; MU=0.02
B0=dict(a=.05,bp=-.08,bd=.07,g=.01,dp=0.,dd=0.)
def ols(X,y):
    XtX=X.T@X; b=np.linalg.solve(XtX,X.T@y); e=y-X@b
    V=np.linalg.inv(XtX)@((X*e[:,None]**2).T@X)@np.linalg.inv(XtX)*len(y)/(len(y)-X.shape[1])
    return b,np.sqrt(np.diag(V))
def design(p,dsc,h): return np.column_stack([np.ones_like(p),p,dsc,h,p*h,dsc*h])
def post(g,s,c2,sm2,mu):
    ve=c2*s**2; lam=sm2/(sm2+ve); mp=mu+lam*(g-mu); tau=np.sqrt(np.maximum(lam*ve,1e-12)); z=mp/tau
    Ep=mp*norm.cdf(z)+tau*norm.pdf(z); return Ep,Ep-mp
def one(rng,sc):
    idx=rng.integers(0,N,N); s=S_EMP[idx]; h=H_EMP[idx].astype(float)
    sm=np.sqrt(SM2)
    if sc.get("sm_s"): sm=np.sqrt(SM2*(0.5+0.5*(s/np.median(S_EMP))**2))   # Var(m) rises with s
    if sc.get("tails"): m=MU+sm*rng.standard_t(4,N)/np.sqrt(2)             # t4 scaled to same variance
    else: m=MU+sm*rng.standard_normal(N)
    e=np.sqrt(C2)*s*rng.standard_normal(N)
    if sc.get("rho"): e=np.sqrt(C2)*s*(sc["rho"]*(m-MU)/np.std(m)+np.sqrt(1-sc["rho"]**2)*rng.standard_normal(N))
    g=m+e; P=sc["par"]
    mp_,md_=np.maximum(m,0),np.maximum(-m,0)
    pr=np.clip(P["a"]+P["bp"]*mp_+P["bd"]*md_+P["g"]*h+P["dp"]*mp_*h+P["dd"]*md_*h,0,1); y=(rng.random(N)<pr).astype(float)
    R={}
    R["oracle"]=ols(design(mp_,md_,h),y)
    R["naive"]=ols(design(np.maximum(g,0),np.maximum(-g,0),h),y)
    gm=g.mean(); r2=(g-gm)**2
    Xv=np.column_stack([np.ones(N),s**2]); cv=np.linalg.lstsq(Xv,r2,rcond=None)[0]; c2h=cv[1]; sm2h=max(r2.mean()-c2h*(s**2).mean(),1e-4)
    Ep,Ed=post(g,np.full(N,np.sqrt((s**2).mean())),c2h,sm2h,gm); R["rc_homo"]=ols(design(Ep,Ed,h),y)
    Ep,Ed=post(g,s,c2h,sm2h,gm); R["icrc"]=ols(design(Ep,Ed,h),y)
    Z=[0,.5,1,1.5,2]; path=[R["naive"][0]]
    for z in Z[1:]:
        bs=[ols(design(np.maximum(gg,0),np.maximum(-gg,0),h),y)[0] for gg in (g+np.sqrt(z*c2h)*s*rng.standard_normal(N) for _ in range(10))]
        path.append(np.mean(bs,0))
    X=np.vstack([np.ones(5),Z,np.square(Z)]).T; R["icsimex"]=(np.array([1,-1,1])@np.linalg.lstsq(X,np.array(path),rcond=None)[0],None)
    return R,c2h
SC={"S1_H0":dict(par=B0),
    "S2_H1":dict(par={**B0,"dp":-.07,"dd":-.12}),
    "S3_H0_tails":dict(par=B0,tails=1),
    "S4_H0_varm_rises":dict(par=B0,sm_s=1),
    "S5_H0_differential":dict(par=B0,rho=.3)}
if __name__=="__main__":
    name=sys.argv[1]; Rn=int(sys.argv[2]); sc=SC[name]; rng=np.random.default_rng(abs(hash(name))%2**32)
    truth=np.array([sc["par"][k] for k in ["a","bp","bd","g","dp","dd"]])
    est={k:[] for k in ["oracle","naive","rc_homo","icrc","icsimex"]}; rej={k:[] for k in est}; c2s=[]
    for r in range(Rn):
        R,c2h=one(rng,sc); c2s.append(c2h)
        for k,(b,se) in R.items():
            est[k].append(b)
            if se is not None: rej[k].append(abs(b[4]-0)/se[4]>1.96)
    out={"truth":truth.tolist(),"c2_mean":float(np.mean(c2s)),"c2_sd":float(np.std(c2s))}
    for k,v in est.items():
        v=np.array(v); out[k]={"mean":v.mean(0).tolist(),"bias":(v.mean(0)-truth).tolist(),"rmse":np.sqrt(((v-truth)**2).mean(0)).tolist(),
                               "rej_dp0":float(np.mean(rej[k])) if rej[k] else None}
    json.dump(out,open(f"mc_{name}.json","w"),indent=1); print(name,"done")
