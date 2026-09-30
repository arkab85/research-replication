"""Interval-calibrated regression calibration (ICRC) for regressors built from an ML benchmark.
Step 1: calibrate the scale c of the vendor's interval from dispersion of the human-benchmark gap.
Step 2: posterior expectations of the kinked regressors (premium, discount) given the observed gap.
Step 3: second-stage regression; inference by city-cluster bootstrap of the whole pipeline."""
from prep import *; import pyfixest as pf
from scipy.stats import norm
def pos(x): return x
def stage1(pl,w=None):
    kw={} if w is None else {"weights":"w"}
    dat=pl if w is None else pl.assign(w=w)
    dat=dat[dat.w>0] if w is not None else dat
    r=pf.feols(f"g ~ {C} | {FE}",data=dat,fixef_rm="none",**kw)
    res=pd.Series(r.resid(),index=dat.index); mu=dat.g-res
    v=pf.feols(f"res2 ~ s2 + {C} | {FE}",data=dat.assign(res2=res**2,s2=dat.s**2),fixef_rm="none",**kw)
    c2=float(v.coef()["s2"])
    ww=np.ones(len(dat)) if w is None else dat.w.values
    sm2=float(np.average(res**2,weights=ww)-c2*np.average(dat.s**2,weights=ww))
    return dat,mu,c2,sm2
def posterior(g,mu,s,c2,sm2):
    ve=c2*s**2; lam=sm2/(sm2+ve); mp=mu+lam*(g-mu); tau=np.sqrt(np.maximum(lam*ve,1e-12))
    z=mp/tau; Ep=mp*norm.cdf(z)+tau*norm.pdf(z); Ed=Ep-mp
    return Ep,Ed,mp,tau,lam
SPEC={"inter":("prem + disc + prem:hiconf + disc:hiconf + hiconf",["prem","disc","prem:hiconf","disc:hiconf"]),
      "base":("prem + disc + conf10",["prem","disc"]),
      "cont":("prem + disc + prem:conf10 + disc:conf10 + conf10",["prem","disc","prem:conf10","disc:conf10"])}
def stage2(dat,spec,w=None,outcome="y"):
    rhs,ks=SPEC[spec]; kw={} if w is None else {"weights":"w"}
    m=pf.feols(f"{outcome} ~ {rhs} + {C} | {FE}",data=dat,fixef_rm="none",vcov={"CRV1":"city"},**kw)
    return m.coef()[ks].values, m.se()[ks].values
def icrc(pl,spec="inter",w=None,c2_override=None,outcome="y"):
    dat,mu,c2,sm2=stage1(pl,w)
    if c2_override is not None:
        ww=np.ones(len(dat)) if w is None else dat.w.values
        res=dat.g-mu; sm2=float(np.average(res**2,weights=ww)-c2_override*np.average(dat.s**2,weights=ww)); c2=c2_override
    Ep,Ed,mp,tau,lam=posterior(dat.g.values,mu.values,dat.s.values,c2,sm2)
    dd=dat.assign(prem=Ep,disc=Ed)
    b,se=stage2(dd,spec,w,outcome)
    return b,se,c2,sm2,dd.assign(mp=mp,tau=tau,lam=lam)
def sample():
    d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&(d.s>0)].copy().reset_index(drop=True); return d,pl
