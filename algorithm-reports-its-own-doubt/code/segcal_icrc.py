"""Web Appendix B.0: segment-calibrated ICRC -- c^2 estimated within 9 thickness x DOM cells,
interaction model re-estimated with cell-specific error variances. Reproduces segcal_icrc.json."""
from icrc import *
import numpy as np, json
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&d.s.notna()&(d.s>0)].copy()
raw=pd.read_csv("/mnt/user-data/uploads/FA_Luxury.csv",thousands=",",low_memory=False); raw.columns=[c.strip() for c in raw.columns]
pl["dom"]=pd.to_numeric(raw.loc[pl.index,"DOM"],errors="coerce")
pl["thick"]=pd.to_numeric(raw.loc[pl.index,"PROP COUNT OF CITY/STATE"],errors="coerce")
pl["cell"]=pd.qcut(pl.thick,3,labels=False).astype(str)+"_"+pd.qcut(pl.dom,3,labels=False).astype(str)
dat,mu,c2g,sm2g=stage1(pl)
dat=dat.assign(res=dat.g-mu); dat["res2"]=dat.res**2; dat["s2"]=dat.s**2
c2=dat.groupby("cell").apply(lambda g: max(float(pf.feols("res2 ~ s2",data=g).coef()["s2"]),0.02),include_groups=False)
dat["c2i"]=dat.cell.map(c2); sm2=float(np.mean(dat.res2-dat.c2i*dat.s2))
Ep,Ed,mpst,tau,lam=posterior(dat.g,mu,dat.s,dat.c2i.values,sm2)
dat["prem"]=Ep; dat["disc"]=mpst-Ep
b,se=stage2(dat,"inter")
print("segment-calibrated ICRC:",dict(zip(["prem","disc","prem:hi","disc:hi"],np.round(b,3))))
