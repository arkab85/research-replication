"""Profile the Ginnie Mae option-exercise extract and the issuer classification."""
import pandas as pd, numpy as np, os, json
from config import OUT, EXTRACT as DATA, ISSUER_TYPES

ISS  = ISSUER_TYPES
os.makedirs(OUT, exist_ok=True)

USE = ["as_of_date","state","pool_id","seq_num","issuer_id","agency","interest_rate",
       "opb","upb","loan_age","months_dlq","credit_score","ltv_current","ltv",
       "fb_flag","num_mth_fb","covid_flag","removal_reason","issue_type","pool_type",
       "Judicial","spread","T10_spread","bEBO","year","valid_entry","analy_data",
       "current_liquidation_flag","loan_origination_date"]

df = pd.read_csv(DATA, usecols=USE, low_memory=False)
print("rows", len(df))
print("\n-- months_dlq --"); print(df.months_dlq.value_counts(dropna=False).head(10))
print("\n-- as_of_date range --", df.as_of_date.min(), df.as_of_date.max())
print("\n-- bEBO --"); print(df.bEBO.value_counts(dropna=False))
print("\n-- agency --"); print(df.agency.value_counts(dropna=False))
print("\n-- n issuers --", df.issuer_id.nunique(), " n pools", df.pool_id.nunique())
print("\n-- loans (pool_seq) --", (df.pool_id.astype(str)+"_"+df.seq_num.astype(str)).nunique())
print("\n-- fb_flag --"); print(df.fb_flag.value_counts(dropna=False))
print("\n-- covid_flag --"); print(df.covid_flag.value_counts(dropna=False))
print("\n-- valid_entry --"); print(df.valid_entry.value_counts(dropna=False))
print("\n-- analy_data --"); print(df.analy_data.value_counts(dropna=False))
print("\n-- removal_reason --"); print(df.removal_reason.value_counts(dropna=False).head(12))
print("\n-- issue_type --"); print(df.issue_type.value_counts(dropna=False).head(12))
print("\n-- yearly counts / buyout rate --")
print(df.groupby("year").agg(n=("bEBO","size"), ebo=("bEBO","mean")))

iss = pd.read_csv(ISS)
print("\n-- issuer_info_with_type --", iss.shape)
print(iss.bank_type.value_counts(dropna=False))
print(iss.columns.tolist())

# coverage of the classification
m = df.issuer_id.map(iss.set_index("IssuerID").bank_type.to_dict())
print("\n-- classified share of decisions --"); print(m.isna().mean())
print(m.value_counts(dropna=False))

# largest unclassified issuers
unc = df.loc[m.isna()].groupby("issuer_id").size().sort_values(ascending=False).head(40)
print("\n-- largest unclassified issuer_ids --"); print(unc)
unc.to_csv(os.path.join(OUT,"unclassified_top40.csv"))
df.groupby("issuer_id").size().sort_values(ascending=False).head(60).to_csv(os.path.join(OUT,"issuer_counts.csv"))
