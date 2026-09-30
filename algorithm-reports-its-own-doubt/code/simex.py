"""Interval-calibrated heteroskedastic SIMEX for the AVM-gap regressions.
Error model: log A_i = log V_i + u_i, u_i ~ N(0, kappa^2 s_i^2), s_i = half-width of vendor log interval.
kappa^2 estimated from the slope of squared residual gap on s^2 (kappa.py)."""
from prep import *; import pyfixest as pf, json, sys
rng=np.random.default_rng(20260919)
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)].copy().reset_index(drop=True)
pc,dc=d.attrs["prem_cap"],d.attrs["disc_cap"]
SPECS={"inter":(f"y ~ prem + disc + prem:hiconf + disc:hiconf + hiconf + {C} | {FE}",["prem","disc","prem:hiconf","disc:hiconf"]),
       "cont":(f"y ~ prem + disc + prem:conf10 + disc:conf10 + conf10 + {C} | {FE}",["prem","disc","prem:conf10","disc:conf10"]),
       "base":(f"y ~ prem + disc + conf10 + {C} | {FE}",["prem","disc"])}
def fit(dat,spec):
    f,ks=SPECS[spec]; m=pf.feols(f,data=dat,vcov={"CRV1":"city"},fixef_rm="none")
    b=m.coef()[ks].values; V=m.vcov.loc[ks,ks].values if hasattr(m.vcov,"loc") else pd.DataFrame(m._vcov,index=m._coefnames,columns=m._coefnames).loc[ks,ks].values
    return b,V
def simex(spec,k2,zetas=(0.5,1.0,1.5,2.0),B=40):
    b0,V0=fit(pl,spec); out={0.0:(b0,V0)}
    for z in zetas:
        bs=[];Vs=[]
        for _ in range(B):
            g=pl.g+rng.normal(0,np.sqrt(z*k2)*pl.s)
            dd=pl.assign(prem=g.clip(lower=0).clip(upper=pc),disc=(-g).clip(lower=0).clip(upper=dc))
            b,V=fit(dd,spec); bs.append(b);Vs.append(V)
        bs=np.array(bs); out[z]=(bs.mean(0), np.mean(Vs,0)-np.cov(bs.T))   # Stefanski-Cook
    Z=np.array(sorted(out)); Bm=np.array([out[z][0] for z in Z]); Vm=np.array([out[z][1] for z in Z])
    X=np.vstack([np.ones_like(Z),Z,Z**2]).T; xe=np.array([1,-1,1])
    coef=np.linalg.lstsq(X,Bm,rcond=None)[0]; bse=xe@coef
    Vflat=Vm.reshape(len(Z),-1); Ve=(xe@np.linalg.lstsq(X,Vflat,rcond=None)[0]).reshape(len(b0),len(b0))
    se=np.sqrt(np.clip(np.diag(Ve),0,None))
    # linear extrapolant as a less aggressive alternative
    cl=np.linalg.lstsq(X[:,:2],Bm,rcond=None)[0]; blin=np.array([1,-1])@cl
    return dict(keys=SPECS[spec][1],naive=b0.tolist(),naive_se=np.sqrt(np.diag(V0)).tolist(),
                path={str(z):out[z][0].tolist() for z in Z},simex_quad=bse.tolist(),simex_se=se.tolist(),simex_lin=blin.tolist())
if __name__=="__main__":
    k2=float(sys.argv[1]); B=int(sys.argv[2]); specs=sys.argv[3].split(",")
    res={s:simex(s,k2,B=B) for s in specs}; res["k2"]=k2
    json.dump(res,open(f"simex_k{k2}_B{B}.json","w"),indent=1); print(json.dumps(res,indent=1))
