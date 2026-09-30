"""Build the estimation panel: option-exercise decisions, Jan 2019 - Sep 2020, with issuer types."""
import pandas as pd, numpy as np, os
from config import OUT, EXTRACT as DATA, ISSUER_TYPES

ISS  = ISSUER_TYPES
os.makedirs(OUT, exist_ok=True)

USE = ["as_of_date","state","pool_id","seq_num","issuer_id","agency","interest_rate",
       "opb","upb","loan_age","credit_score","ltv_current","fb_flag","num_mth_fb",
       "Judicial","spread","T10_spread","bEBO","year"]
df = pd.read_csv(DATA, usecols=USE, low_memory=False)

df["buyout"]  = df.bEBO.astype(str).str.upper().eq("TRUE").astype(float)
df["ym"]      = df.as_of_date.astype(int)
df["month"]   = pd.to_datetime(df.ym.astype(str), format="%Y%m")
df["forbear"] = df.fb_flag.astype(str).str.upper().eq("Y").astype(float)

# ---------------- issuer classification ----------------
iss  = pd.read_csv(ISS)
base = iss.set_index("IssuerID").bank_type.to_dict()
BASEMAP = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
df["itype_base"] = df.issuer_id.map(lambda x: BASEMAP.get(base.get(x)))

# Hand extension: the largest issuers with no base classification, typed from charter records.
# depository = holds an insured depository charter; nonbank = non-depository mortgage company;
# hfa = state housing finance agency (tax-exempt bond funded).
EXT = {
    4150:"nonbank",    # Lakeview Loan Servicing, LLC
    3871:"nonbank",    # Freedom Mortgage Corp
    3162:"depository",  # MidFirst Bank
    2559:"depository",  # Truist Bank
    4102:"nonbank",    # The Money Source Inc.
    4159:"nonbank",    # Pingora Loan Servicing, LLC
    4158:"nonbank",    # RoundPoint Mortgage Servicing Corp
    3663:"depository",  # M&T Bank
    3770:"nonbank",    # First Guaranty Mortgage Corp
    4135:"nonbank",    # Planet Home Lending, LLC
    4068:"hfa",        # Idaho Housing and Finance Association
    3977:"hfa",        # Alabama Housing Finance Authority
    4033:"depository",  # Gateway First Bank (chartered 2019)
    3266:"depository",  # Citizens Bank, N.A.
    4058:"nonbank",    # Village Capital & Investment, LLC
    3774:"depository",  # BOKF, NA
    3590:"nonbank",    # Matrix Financial Services Corp
    4413:"depository",  # Fifth Third Bank, N.A.
    4034:"nonbank",    # American Financial Resources, Inc.
    4047:"nonbank",    # Embrace Home Loans, Inc.
    4043:"hfa",        # Colorado Housing and Finance Authority
    4060:"hfa",        # Virginia Housing Development Authority
    3177:"depository",  # Trustmark National Bank
    3350:"nonbank",    # Sun West Mortgage Co., Inc.
    1990:"nonbank",    # Mid America Mortgage Inc.
    1798:"nonbank",    # James B. Nutter & Company
    4072:"nonbank",    # AmeriFirst Home Mortgage
    4062:"hfa",        # Pennsylvania Housing Finance Agency
    3967:"nonbank",    # Stearns Lending, LLC
    4123:"nonbank",    # GMFS LLC
    4297:"nonbank",    # Nations Lending Corp
    4061:"hfa",        # Utah Housing Corporation
    4086:"nonbank",    # Selene Finance LP
    4273:"nonbank",    # Arc Home LLC
    3458:"nonbank",    # Towne Mortgage Company
    4408:"depository",  # Truist Bank
    2265:"depository",  # BancorpSouth Bank
    1699:"nonbank",    # Standard Mortgage Corp
    4246:"nonbank",    # Rushmore Loan Management Services, LLC
    4187:"nonbank",    # Nations Direct Mortgage, LLC
}
# keep the 30 largest unclassified issuers inside the estimation window, as in the paper
win = df.ym.between(201901, 202009)
unc = (df.loc[win & df.itype_base.isna()].groupby("issuer_id").size()
         .sort_values(ascending=False))
unc = unc[unc.index.isin(EXT)].head(30)
EXT30 = {k: EXT[k] for k in unc.index}
print("hand-extended issuers (30 largest unclassified in window):")
for k, v in EXT30.items():
    print(f"   {k:>5}  {v:<11} {unc[k]:>8,}")

df["itype_ext"] = df.itype_base.where(df.itype_base.notna(),
                                      df.issuer_id.map(EXT30))

# ---------------- estimation panel ----------------
p = df.loc[win].copy()
p["post"]       = (p.ym >= 202003).astype(float)
p["nonbank"]    = (p.itype_ext == "nonbank").astype(float)
p["state_month"] = p.state.astype(str) + "_" + p.ym.astype(str)
p["coupon"]     = pd.to_numeric(p.interest_rate, errors="coerce")
p["fico"]       = pd.to_numeric(p.credit_score, errors="coerce")
p["cltv"]       = pd.to_numeric(p.ltv_current, errors="coerce")
p["age"]        = pd.to_numeric(p.loan_age, errors="coerce")

for name, col in [("BASE", "itype_base"), ("EXT", "itype_ext")]:
    s = p[p[col].notna()]
    print(f"\n{name}: {len(s):,} decisions, {s.issuer_id.nunique()} issuers")
    print(s[col].value_counts())
    core = s[s[col].isin(["depository", "nonbank", "techfirst"])]
    print(f"  excl. HFA -> {len(core):,} decisions, {core.issuer_id.nunique()} issuers")

p.to_parquet(os.path.join(OUT, "panel.parquet"), index=False)

# full-history file for the by-year table and the pre-2019 placebo
df["itype"] = df.itype_ext
keep = ["ym","year","state","issuer_id","itype","buyout","forbear","interest_rate",
        "credit_score","ltv_current","loan_age","agency","upb","opb","Judicial",
        "num_mth_fb","spread","T10_spread","pool_id","seq_num"]
df[keep].to_parquet(os.path.join(OUT, "full.parquet"), index=False)
print("\nwrote panel.parquet and full.parquet")
