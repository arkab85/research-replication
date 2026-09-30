"""Web Appendix B.0: c^2 by segment (DOM terciles, price tiers, thickness terciles, vintage) with
markup-side moments and joint equality tests. Reproduces seg_stability.json."""
from prep import *; import pyfixest as pf, numpy as np, json
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&d.s.notna()&(d.s>0)].copy()
r=pf.feols(f"g ~ {C} | {FE}",data=pl,fixef_rm="none"); pl["res"]=r.resid(); pl["res2"]=pl.res**2; pl["s2"]=pl.s**2
raw=pd.read_csv("/mnt/user-data/uploads/FA_Luxury.csv",thousands=",",low_memory=False); raw.columns=[c.strip() for c in raw.columns]
pl["dom"]=pd.to_numeric(raw.loc[pl.index,"DOM"],errors="coerce")
pl["thick"]=pd.to_numeric(raw.loc[pl.index,"PROP COUNT OF CITY/STATE"],errors="coerce")
pl["seg_dom"]=pd.qcut(pl.dom,3,labels=["fresh","mid","stale"])
pl["seg_price"]=pd.cut(pd.to_numeric(raw.loc[pl.index,"CurrentListingPrice"],errors="coerce"),[0,750e3,1.5e6,np.inf],labels=["500-750k","750k-1.5M","1.5M+"])
pl["seg_thick"]=pd.qcut(pl.thick,3,labels=["thin","mid","thick"])
pl["seg_time"]=pd.qcut(pd.to_datetime(raw.loc[pl.index,"ListingDate"],errors="coerce").astype("int64"),2,labels=["early","late"])
for name in ["seg_dom","seg_price","seg_thick","seg_time"]:
    for lev,g in pl.groupby(name,observed=True):
        k=pf.feols("res2 ~ s2",data=g,vcov={"CRV1":"city"})
        print(name,lev,len(g),"c2=%.3f(%.3f)"%(k.coef()["s2"],k.se()["s2"]),
              "mean_gap=%.3f share_prem10=%.3f"%(g.g.mean(),(g.g>np.log(1.10)).mean()))
