from prep import *; import pyfixest as pf
d=load(); pl=d[d.plaus&d.ConfidenceScore.notna()&d.s.notna()&(d.s>0)].copy()
print(len(pl), pl.s.describe().round(3).to_dict())
# prem check vs journal
m=pf.feols(f"y ~ prem + disc + prem:hiconf + disc:hiconf + hiconf + {C} | {FE}",data=pl,vcov={"CRV1":"city"},fixef_rm="none"); print(m.tidy().iloc[:4,:3])
# residualize gap on covariates+FE
r=pf.feols(f"g ~ {C} | {FE}",data=pl,fixef_rm="none"); pl["res"]=r.resid()
pl["res2"]=pl.res**2; pl["s2"]=pl.s**2
k=pf.feols("res2 ~ s2",data=pl,vcov={"CRV1":"city"}); print(k.tidy().iloc[:,:3])
k2=pf.feols(f"res2 ~ s2 + {C} | {FE}",data=pl,vcov={"CRV1":"city"},fixef_rm="none"); print(k2.tidy().loc["s2"].iloc[:3])
# by confidence bin
pl["cb"]=pd.cut(pl.ConfidenceScore,[0,60,70,80,90,100])
print(pl.groupby("cb").agg(n=("res","size"),var=("res","var"),s2=("s2","mean")))
print(pl.groupby("hiconf").agg(n=("res","size"),var=("res","var"),s2=("s2","mean")))
