"""Replication script for "Valuation Disagreement, AVM Confidence, and Marketability".
Usage: python replication.py FA_Luxury.csv   ->  writes results.json with every number reported
in the manuscript and Online Resource 1. Requires pandas, numpy, pyfixest, scikit-learn.
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import pyfixest as pf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# ---------------- load and clean
d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "FA_Luxury.csv",dtype=str,low_memory=False)
d.columns=[c.strip() for c in d.columns]
money=lambda s: pd.to_numeric(s.str.replace(r"[\$,\s]","",regex=True).replace("NULL",np.nan),errors="coerce")
for c in ["MaxListPrice","MinListPrice","CurrentListingPrice","FACurrentAVM","Low_Value","High_Value","SoldPrice"]: d[c]=money(d[c])
for c in ["LotSizeSqFt","HomeSize","YearBuilt","ConfidenceScore","DOM"]:
    d[c]=pd.to_numeric(d[c].str.replace(",","").str.strip().replace("NULL",np.nan),errors="coerce")

# ---------------- variables
d["y"]=d.Status.isin(["Pending","Contingent"]).astype(int)
P,A=d.CurrentListingPrice,d.FACurrentAVM
w=lambda x: x.clip(x.quantile(.01),x.quantile(.99))
lr=np.log(P/A)
d["absd"]=w(lr.abs()); d["prem"]=w(lr.clip(lower=0)); d["disc"]=w((-lr).clip(lower=0))
d["outside"]=w(pd.Series(np.maximum.reduce([np.log(P/d.High_Value).values,np.log(d.Low_Value/P).values,np.zeros(len(d))]),index=d.index))
d["above10"]=(P>1.1*A).astype(int)
d["conf10"]=d.ConfidenceScore/10
d["lp"]=w(np.log(P))
for v,src in [("lsq","HomeSize"),("llot","LotSizeSqFt")]:
    x=np.log(d[src].where(d[src]>0)); d[v+"_m"]=x.isna().astype(int); d[v]=w(x).fillna(0)
age=2019-d.YearBuilt; d["age_m"]=age.isna().astype(int); d["age"]=w(age).fillna(0)
d["city"]=d["CITY/STATE"].fillna("NA"); d["zip"]=d.PropertyZip.fillna("NA"); d["state"]=d.PropertyState
d["ptype"]=d.PropertyType.fillna("NA")
d["month"]=pd.to_datetime(d.ListingDate).dt.to_period("M").astype(str)
d["stmonth"]=d.state+"_"+d.month
d["plaus"]=(A/P).between(.5,2)
d["hiconf"]=(d.ConfidenceScore>=80).astype(int)
C="lp + lsq + llot + age + lsq_m + llot_m + age_m"

# ---------------- estimation
R={}
def est(lab,f,dat,ks,cl="city"):
    m=pf.feols(f,data=dat,vcov={"CRV1":cl},fixef_rm="none",fixef_maxiter=100000); t=m.tidy()
    R[lab]={"N":int(m._N),**{k:[float(round(t.loc[k,"Estimate"],4)),float(round(t.loc[k,"Std. Error"],4)),float(round(t.loc[k,"Pr(>|t|)"],4))] for k in ks}}
FE="city + ptype + month"; pl=d[d.plaus]
# ---- descriptives
R["desc"]=dict(N=len(d),active=int((d.Status=="Active").sum()),pending=int((d.Status=="Pending").sum()),contingent=int((d.Status=="Contingent").sum()),
  share=round(d.y.mean()*100,2),plaus=int(d.plaus.sum()),ge1m=int((d.CurrentListingPrice>=1e6).sum()),
  minprice=float(d.CurrentListingPrice.min()),medprice=float(d.CurrentListingPrice.median()),
  missconf=int(d.ConfidenceScore.isna().sum()),inside=round((d.outside==0).mean()*100,1),
  y_inside=round(d.y[d.outside==0].mean()*100,2),y_outside=round(d.y[d.outside>0].mean()*100,2),
  hiconf_plaus=int(pl.hiconf.sum()))
r=d.CurrentListingPrice/d.FACurrentAVM
b=pd.cut(r,[0,.9,1.1,1.25,1.5,2,np.inf],labels=["lt90","ref","b110_125","b125_150","b150_200","gt200"])
bf=pd.cut(r,[0,.9,1.1,1.25,2,np.inf],labels=["<0.90","0.90-1.10","1.10-1.25","1.25-2.00",">2.00"])
R["fig1"]={k:round(v*100,2) for k,v in d.groupby(bf).y.mean().items()}
R["fig1_n"]={k:int(v) for k,v in bf.value_counts().items()}
for l in ["lt90","b110_125","b125_150","b150_200","gt200"]: d["bin_"+l]=(b==l).astype(int)
# ---- Table 2
est("T2c1",f"y ~ absd + conf10 + {C} | {FE}",d,["absd","conf10"])
est("T2c2",f"y ~ prem + disc + conf10 + {C} | {FE}",d,["prem","disc","conf10"])
est("T2c3",f"y ~ prem + disc + conf10 + {C} | {FE}",pl,["prem","disc","conf10"])
est("T2c4",f"y ~ prem + disc + conf10 + {C} | {FE}",d[d.CurrentListingPrice>=1e6],["prem","disc","conf10"])
# ---- Table 3 (all with conf control where available -> N 50,395 / 43,316)
est("T3outside",f"y ~ outside + conf10 + {C} | {FE}",d,["outside"])
est("T3above10",f"y ~ above10 + conf10 + {C} | {FE}",d,["above10"])
est("T3lowconf",f"y ~ prem + disc + conf10 + {C} | {FE}",pl[(pl.hiconf==0)&pl.ConfidenceScore.notna()],["prem","disc"])
est("T3hiconf",f"y ~ prem + disc + conf10 + {C} | {FE}",pl[pl.hiconf==1],["prem","disc"])
est("T3inter",f"y ~ prem + disc + prem:hiconf + disc:hiconf + hiconf + {C} | {FE}",pl[pl.ConfidenceScore.notna()],["prem","disc","prem:hiconf","disc:hiconf"])
est("T3inter_cont",f"y ~ prem + disc + prem:conf10 + disc:conf10 + conf10 + {C} | {FE}",pl,["prem","disc","prem:conf10","disc:conf10"])
# ---- Table 4 bins
est("T4",f"y ~ bin_lt90 + bin_b110_125 + bin_b125_150 + bin_b150_200 + bin_gt200 + conf10 + {C} | {FE}",d,["bin_lt90","bin_b110_125","bin_b125_150","bin_b150_200","bin_gt200"])
# ---- Table 5 robustness (conf control included throughout)
est("T5zip",f"y ~ absd + conf10 + {C} | zip + ptype + month",d,["absd"],"zip")
est("T5zipstm",f"y ~ absd + conf10 + {C} | zip + ptype + stmonth",d,["absd"],"zip")
est("T5pl_zip",f"y ~ prem + disc + conf10 + {C} | zip + ptype + month",pl,["prem","disc"],"zip")
est("T5pl_500k5m",f"y ~ prem + disc + conf10 + {C} | {FE}",pl[pl.CurrentListingPrice<=5e6],["prem","disc"])
est("T5pl_sfr",f"y ~ prem + disc + conf10 + {C} | city + month",pl[pl.ptype=="SFR"],["prem","disc"])
est("T5_nowins",f"y ~ absd_raw + conf10 + {C} | {FE}",d.assign(absd_raw=np.abs(np.log(d.CurrentListingPrice/d.FACurrentAVM))),["absd_raw"])
# ---- DOM diagnostic
d["ldom"]=np.log1p(d.DOM)
est("DOM_nomonth",f"ldom ~ absd + conf10 + {C} | city + ptype",d,["absd","conf10"])
est("DOM_month",f"ldom ~ absd + conf10 + {C} | {FE}",d,["absd","conf10"])
# ---- price revisions
R["revised_share"]=round((d.CurrentListingPrice<d.MaxListPrice).mean()*100,1)
R["maxprice_missing"]=int(d.MaxListPrice.isna().sum())
# ---- disclosure split
nd={"AK","ID","KS","LA","MS","MO","MT","NM","ND","TX","UT","WY"}
pl2=pl.assign(nd=pl.state.isin(nd))
est("DISC_disclosure",f"y ~ prem + disc + conf10 + {C} | {FE}",pl2[~pl2.nd],["prem"])
est("DISC_nondisclosure",f"y ~ prem + disc + conf10 + {C} | {FE}",pl2[pl2.nd],["prem"])
R["disc_conf"]=[round(pl2.ConfidenceScore[~pl2.nd].mean(),1),round(pl2.ConfidenceScore[pl2.nd].mean(),1)]
R["disc_n"]=[int((~pl2.nd).sum()),int(pl2.nd.sum())]
# ---- round thresholds
th=[6e5,7e5,7.5e5,8e5,9e5,1e6,1.25e6,1.5e6,1.75e6,2e6,2.5e6,3e6,3.5e6,4e6,4.5e6,5e6]
P=d.CurrentListingPrice.values
bun={}
for wdt in [.01,.02,.03]:
    below=sum(((P>=t*(1-wdt))&(P<t)).sum() for t in th); above=sum(((P>=t)&(P<t*(1+wdt))).sum() for t in th)
    bun[f"{int(round(wdt*100))}pct"]=[int(below),int(above),round(below/above,2)]
R["bunching"]=bun
rd={}
for wdt in [.02,.03,.05,.10]:
    rows=[]
    for i,t in enumerate(th):
        m=(P>=t*(1-wdt))&(P<t*(1+wdt)); s=d[m].copy(); s["x"]=np.log(s.CurrentListingPrice/t); s["thr"]=str(i); rows.append(s)
    s=pd.concat(rows); s["above"]=(s.x>=0).astype(int); s["xa"]=s.x*s.above
    m=pf.feols("y ~ above + x + xa | thr + state",data=s,vcov={"CRV1":"city"},fixef_rm="none"); t=m.tidy()
    rd[f"{int(round(wdt*100))}pct"]=[float(round(t.loc["above","Estimate"],4)),float(round(t.loc["above","Std. Error"],4)),int(m._N)]
R["rd"]=rd
# ---- AIPW
s=d[(d.CurrentListingPrice>d.High_Value)|(d.outside==0)].copy()
s["T"]=(s.CurrentListingPrice>s.High_Value).astype(int)
X=pd.DataFrame({"lavm":np.log(s.FACurrentAVM),"lwidth":np.log(s.High_Value/s.Low_Value),"lsq":s.lsq,"llot":s.llot,"age":s.age,
  "lsq_m":s.lsq_m,"llot_m":s.llot_m,"age_m":s.age_m,"conf":s.ConfidenceScore,"lat":pd.to_numeric(s.SitusLatitude,errors="coerce"),
  "lon":pd.to_numeric(s.SitusLongitude,errors="coerce"),"state":s.state.astype("category").cat.codes,
  "ptype":s.ptype.where(s.ptype.map(s.ptype.value_counts())>=50,"OTHER").astype("category").cat.codes,
  "month":s.month.astype("category").cat.codes})
cat=[X.columns.get_loc(c) for c in ["state","ptype","month"]]
T=s["T"].values; Y=s.y.values.astype(float); g=s.zip.values
e=np.zeros(len(s)); m1=np.zeros(len(s)); m0=np.zeros(len(s))
for tr,te in GroupKFold(5).split(X,T,g):
    ps=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,categorical_features=cat,random_state=0).fit(X.iloc[tr],T[tr]); e[te]=ps.predict_proba(X.iloc[te])[:,1]
    for tt,arr in [(1,m1),(0,m0)]:
        idx=tr[T[tr]==tt]; mo=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,categorical_features=cat,random_state=0).fit(X.iloc[idx],Y[idx]); arr[te]=mo.predict(X.iloc[te])
k=(e>=.05)&(e<=.95)
Tk,Yk,ek,m1k,m0k,gk=T[k],Y[k],e[k],m1[k],m0[k],g[k]
psi_ate=m1k-m0k+Tk*(Yk-m1k)/ek-(1-Tk)*(Yk-m0k)/(1-ek)
p=Tk.mean(); psi_att=(Tk*(Yk-m0k)-(1-Tk)*ek/(1-ek)*(Yk-m0k))/p
rng=np.random.default_rng(2026); zips=np.unique(gk); gi={z:np.where(gk==z)[0] for z in zips}
bs_ate=[];bs_att=[]
for _ in range(1000):
    idx=np.concatenate([gi[z] for z in rng.choice(zips,len(zips))])
    bs_ate.append(psi_ate[idx].mean()); pp=Tk[idx].mean()
    bs_att.append(((Tk[idx]*(Yk[idx]-m0k[idx])-(1-Tk[idx])*ek[idx]/(1-ek[idx])*(Yk[idx]-m0k[idx]))).mean()/pp)
R["aipw"]=dict(N=int(k.sum()),treated=int(Tk.sum()),ate=round(psi_ate.mean()*100,2),ate_ci=[round(np.percentile(bs_ate,2.5)*100,2),round(np.percentile(bs_ate,97.5)*100,2)],
  att=round(psi_att.mean()*100,2),att_ci=[round(np.percentile(bs_att,2.5)*100,2),round(np.percentile(bs_att,97.5)*100,2)])

# ---- heterogeneity: listing cohort (length-biased sampling) and price tier
def est2(f,dat,ks):
    m=pf.feols(f,data=dat,vcov={"CRV1":"city"},fixef_rm="none",fixef_maxiter=100000); t=m.tidy()
    return {"N":int(m._N),"ybar":round(dat.loc[dat.ConfidenceScore.notna(),"y"].mean()*100,2),**{k:[round(float(t.loc[k,"Estimate"]),4),round(float(t.loc[k,"Std. Error"]),4),round(float(t.loc[k,"Pr(>|t|)"]),4)] for k in ks}}
H={}; ld=pd.to_datetime(d.ListingDate)
for lab,cut in [("recent6",pd.Timestamp("2018-09-01")),("recent12",pd.Timestamp("2018-03-01"))]:
    H[lab+"_full"]=est2(f"y ~ absd + conf10 + {C} | {FE}",d[ld>=cut],["absd"])
    H[lab+"_pl"]=est2(f"y ~ prem + disc + conf10 + {C} | {FE}",pl[pd.to_datetime(pl.ListingDate)>=cut],["prem","disc"])
    H[lab+"_older_full"]=est2(f"y ~ absd + conf10 + {C} | {FE}",d[ld<cut],["absd"])
q=pl.CurrentListingPrice.quantile([1/3,2/3]).values
for lab,m in [("t1",pl.CurrentListingPrice<q[0]),("t2",pl.CurrentListingPrice.between(q[0],q[1],inclusive="left")),("t3",pl.CurrentListingPrice>=q[1])]:
    H["price_"+lab]=est2(f"y ~ prem + disc + conf10 + {C} | {FE}",pl[m],["prem","disc"])
H["terciles"]=[float(x) for x in q]
R["hetero"]=H
# ---- summary statistics
S={}
vv={"Asking price (\\$000)":d.CurrentListingPrice/1e3,"AVM point estimate (\\$000)":d.FACurrentAVM/1e3,"AVM / asking price":d.FACurrentAVM/d.CurrentListingPrice,
    "AVM confidence score":d.ConfidenceScore,"AVM range width / AVM":(d.High_Value-d.Low_Value)/d.FACurrentAVM,
    "Living area (sq.\\ ft.)":d.HomeSize.where(d.HomeSize>0),"Lot size (sq.\\ ft.)":d.LotSizeSqFt.where(d.LotSizeSqFt>0),
    "Property age (years)":(2019-d.YearBuilt).where(d.YearBuilt>1700),"Days on market":d.DOM}
for k,x in vv.items():
    x=x.dropna(); S[k]=[int(len(x)),float(round(x.mean(),2)),float(round(x.std(),2)),float(round(x.quantile(.25),2)),float(round(x.median(),2)),float(round(x.quantile(.75),2))]
R["sumstats"]=S
v=d.outside.notna(); R["desc"]["inside_valid"]=round((d.outside[v]==0).mean()*100,1); R["desc"]["n_range"]=int(v.sum())
R["desc"]["ratio_q"]=[round(x,2) for x in (d.FACurrentAVM/d.CurrentListingPrice).quantile([.01,.5,.99])]
R["thresholds"]=th
json.dump(R,open("results.json","w"),indent=1,default=float)
print("wrote results.json")
