import pandas as pd,numpy as np,warnings;warnings.filterwarnings("ignore")
def load(path="/mnt/user-data/uploads/FA_Luxury.csv"):
    d=pd.read_csv(path,dtype=str,low_memory=False); d.columns=[c.strip() for c in d.columns]
    money=lambda s: pd.to_numeric(s.str.replace(r"[\$,\s]","",regex=True).replace("NULL",np.nan),errors="coerce")
    for c in ["MaxListPrice","MinListPrice","CurrentListingPrice","FACurrentAVM","Low_Value","High_Value"]: d[c]=money(d[c])
    for c in ["LotSizeSqFt","HomeSize","YearBuilt","ConfidenceScore","DOM"]:
        d[c]=pd.to_numeric(d[c].str.replace(",","").str.strip().replace("NULL",np.nan),errors="coerce")
    d["y"]=d.Status.isin(["Pending","Contingent"]).astype(int)
    P,A=d.CurrentListingPrice,d.FACurrentAVM
    w=lambda x: x.clip(x.quantile(.01),x.quantile(.99))
    d["g"]=np.log(P/A)
    d["lp"]=w(np.log(P))
    for v,src in [("lsq","HomeSize"),("llot","LotSizeSqFt")]:
        x=np.log(d[src].where(d[src]>0)); d[v+"_m"]=x.isna().astype(int); d[v]=w(x).fillna(0)
    age=2019-d.YearBuilt; d["age_m"]=age.isna().astype(int); d["age"]=w(age).fillna(0)
    d["city"]=d["CITY/STATE"].fillna("NA"); d["zip"]=d.PropertyZip.fillna("NA"); d["state"]=d.PropertyState
    d["ptype"]=d.PropertyType.fillna("NA")
    d["month"]=pd.to_datetime(d.ListingDate).dt.to_period("M").astype(str)
    d["plaus"]=(A/P).between(.5,2)
    d["hiconf"]=(d.ConfidenceScore>=80).astype(int)
    d["conf10"]=d.ConfidenceScore/10
    d["s"]=np.log(d.High_Value/d.Low_Value)/2      # half-width of vendor interval, log points
    # caps used in journal (full-sample 1/99 pct) so corrected regressors share the original scale
    lr=d.g
    d.attrs["prem_cap"]=float(lr.clip(lower=0).quantile(.99)); d.attrs["disc_cap"]=float((-lr).clip(lower=0).quantile(.99))
    d["prem"]=lr.clip(lower=0).clip(upper=d.attrs["prem_cap"]); d["disc"]=(-lr).clip(lower=0).clip(upper=d.attrs["disc_cap"])
    return d
C="lp + lsq + llot + age + lsq_m + llot_m + age_m"
FE="city + ptype + month"
