from icrc import *; import json
d,pl=sample(); out={}
# 1. calibration: binned variance of residual gap vs mean s^2
dat,mu,c2,sm2=stage1(pl); res=dat.g-mu
b=pd.cut(dat.ConfidenceScore,[0,60,70,75,80,85,90,95,100])
out["bins"]=[[str(k),int(len(v)),float(v.var()),float((dat.s[v.index]**2).mean()),float(dat.s[v.index].mean())] for k,v in res.groupby(b)]
out["c2"]=c2; out["sm2"]=sm2
# 2. sensitivity of ICRC to c^2
G={}
for k2 in [0.0,0.05,0.10,0.145,c2,0.23,0.30]:
    bi,se,_,s2,dd=icrc(pl,"inter",c2_override=k2); bb,seb,_,_,_=icrc(pl,"base",c2_override=k2)
    G[f"{k2:.3f}"]=dict(inter=bi.tolist(),inter_se=se.tolist(),base=bb.tolist(),base_se=seb.tolist(),sm2=s2,
        lam_lo=float(dd.lam[dd.hiconf==0].mean()),lam_hi=float(dd.lam[dd.hiconf==1].mean()))
out["grid"]=G
# 3. reliability summary
_,_,_,_,dd=icrc(pl,"inter")
out["lam"]={"lo":float(dd.lam[dd.hiconf==0].mean()),"hi":float(dd.lam[dd.hiconf==1].mean()),"all":float(dd.lam.mean()),
            "q":[float(x) for x in dd.lam.quantile([.1,.5,.9])]}
# 4. uncertainty-aware overpricing flag vs raw-gap flag (threshold: 10 percent overpricing)
from scipy.stats import norm
pi=1-norm.cdf((np.log(1.10)-dd.mp)/dd.tau); raw=dd.g>np.log(1.10); postf=pi>0.5
tab={}
for a in [0,1]:
    for bb_ in [0,1]:
        m=(raw==a)&(postf==bb_); tab[f"raw{a}_post{bb_}"]=[int(m.sum()),float(dd.y[m].mean()*100) if m.sum() else None,
            float(dd.hiconf[m].mean()*100) if m.sum() else None, float(dd.ConfidenceScore[m].mean()) if m.sum() else None]
out["flags"]=tab; out["flag_counts"]=[int(raw.sum()),int(postf.sum())]
# flag coefficients with FE
dd2=dd.assign(rawf=raw.astype(int),postf=postf.astype(int),pi=pi)
for f in ["rawf","postf","pi"]:
    m=pf.feols(f"y ~ {f} + conf10 + {C} | {FE}",data=dd2,vcov={"CRV1":"city"},fixef_rm="none")
    out["flag_"+f]=[float(m.coef()[f]),float(m.se()[f]),float(m.pvalue()[f])]
m=pf.feols(f"y ~ rawf + postf + conf10 + {C} | {FE}",data=dd2,vcov={"CRV1":"city"},fixef_rm="none")
out["flag_horse"]={k:[float(m.coef()[k]),float(m.se()[k]),float(m.pvalue()[k])] for k in ["rawf","postf"]}
# 5. naive exact p-values for the main table
for sp in ["base","inter"]:
    rhs,ks=SPEC[sp]; m=pf.feols(f"y ~ {rhs} + {C} | {FE}",data=pl,vcov={"CRV1":"city"},fixef_rm="none")
    out["naive_"+sp]={k:[float(m.coef()[k]),float(m.se()[k]),float(m.pvalue()[k])] for k in ks}; out["N"]=int(m._N)
out["ybar"]=float(pl.y.mean()*100); out["n_hi"]=int(pl.hiconf.sum()); out["n_cities"]=int(pl.city.nunique())
out["s_q"]=[float(x) for x in pl.s.quantile([.1,.25,.5,.75,.9])]
json.dump(out,open("grid_flags.json","w"),indent=1); print(json.dumps({k:v for k,v in out.items() if k!="grid"},indent=1)[:4000])
for k,v in G.items(): print(k,np.round(v["inter"],4),np.round(v["base"],4),round(v["lam_lo"],3),round(v["lam_hi"],3))
