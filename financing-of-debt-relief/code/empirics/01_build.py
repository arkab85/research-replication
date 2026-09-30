"""Build the vesting-level decision panel, 2016m1-2022m9, from the files constructed from Ginnie Mae's
public loan-level disclosures (one row per loan in the month it first reaches three missed payments,
with its removal outcome over the following twelve months and its final outcome through 2022m12).

Exercise x = repurchase (removal reason 2) within WINDOW months of vesting, before any self-cure.
Issuer charter types: base types from the issuer file, plus the largest unclassified issuers typed
by hand from charter records (dictionary EXT)."""
import pandas as pd, numpy as np
from config import VESTING, ISSUERS, WORK, WINDOW, PERIODS, PMMS
BASEMAP = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
EXT = {4150:"nonbank",3871:"nonbank",3162:"depository",2559:"depository",4102:"nonbank",4159:"nonbank",4158:"nonbank",
       3663:"depository",3770:"nonbank",4135:"nonbank",4068:"hfa",3977:"hfa",4033:"depository",3266:"depository",4058:"nonbank",
       3774:"depository",3590:"nonbank",4413:"depository",4034:"nonbank",4047:"nonbank",4043:"hfa",4060:"hfa",3177:"depository",
       3350:"nonbank",1990:"nonbank",1798:"nonbank",4072:"nonbank",4062:"hfa",3967:"nonbank",4123:"nonbank",4297:"nonbank",
       4061:"hfa",4086:"nonbank",4273:"nonbank",3458:"nonbank",4408:"depository",2265:"depository",1699:"nonbank",4246:"nonbank",4187:"nonbank",
       # largest unclassified issuers in 2019-2022, typed by charter
       4205:"nonbank",3219:"nonbank",4173:"nonbank",4289:"depository",4115:"nonbank",3993:"nonbank",4228:"nonbank",4294:"nonbank",
       4368:"nonbank",3676:"hfa",4290:"nonbank",4077:"hfa",4379:"depository",4171:"nonbank",4329:"depository",4343:"depository",
       4063:"hfa",4266:"depository",4179:"nonbank",4193:"nonbank",4320:"nonbank",4018:"depository",4314:"depository",4003:"depository",
       4393:"nonbank",4383:"depository",4274:"nonbank",4261:"nonbank",1997:"depository",4218:"nonbank",4163:"nonbank",4416:"nonbank",
       4162:"nonbank",3937:"other",4144:"depository",4006:"nonbank",4278:"nonbank",4107:"nonbank",4070:"nonbank",4139:"nonbank"}

def read(i):
    for ext in ("parquet", "csv"):
        f = VESTING / f"00c_gnma_dlq_analysis_data_00{i}_twelve_months.{ext}"
        if f.exists():
            return pd.read_parquet(f) if ext == "parquet" else pd.read_csv(f, low_memory=False, dtype={"pool_id": str, "identifier": str})
    return pd.read_parquet(VESTING / f"v00{i}.parquet")   # parquet copies made on the author's machine

df = pd.concat([read(1).assign(gnma=1), read(2).assign(gnma=2)], ignore_index=True)
df["ym"] = pd.to_datetime(df.as_of_date).dt.strftime("%Y%m").astype(int)
df = df[df.ym.between(201402, 202209)].copy()
iss = pd.read_csv(ISSUERS)
tmap = {k: BASEMAP.get(v) for k, v in iss.set_index("IssuerID").bank_type.to_dict().items()}
for k, v in EXT.items():
    if tmap.get(k) is None: tmap[k] = v
df["itype"] = df.issuer_id.map(tmap)
# outcomes
ex = df.removal_reason.eq(2)
df["t_ex"] = np.where(ex, df.total_months_post_dlq, np.nan)
df["x"] = (ex & (df.total_months_post_dlq <= WINDOW)).astype(float)
df["x12"] = ex.astype(float)
fin = pd.to_datetime(df.final_as_of_date); ves = pd.to_datetime(df.as_of_date)
df["t_final"] = (fin.dt.year - ves.dt.year) * 12 + (fin.dt.month - ves.dt.month)
df["x_ever"] = df.final_removal_reason.eq(2).astype(float)          # repurchased by 2022m12
df["t_ever"] = np.where(df.x_ever == 1, df.t_final, np.nan)
df["gone3"] = (df.removal_reason.isin([1, 3, 4, 5, 6]) & (df.total_months_post_dlq <= WINDOW)).astype(float)   # left the pool for another reason within the window
df["foreclosed12"] = df.removal_reason.eq(3).astype(float)
df["cured12"] = df.removal_reason.isin([1, 8]).astype(float)
df["forbear"] = df.forbearance_flag.astype(float)
df["period"] = None
for p, a, b in PERIODS:
    df.loc[df.ym.between(a, b), "period"] = p
df["year"] = df.ym // 100
df["spread"] = df.interest_rate - df.T10
pm = pd.read_csv(PMMS).set_index("ym").rate
df["pmms"] = df.ym.map(pm)
df["mspread"] = df.interest_rate - df.pmms          # note rate minus the contemporaneous 30-year mortgage rate
df["T10_change"] = df.T10 - df.T10_contract_date
df["upb_cash"] = df.upb
keep = ["identifier", "pool_id", "issuer_id", "itype", "gnma", "ym", "year", "period", "x", "x12", "x_ever", "t_ex", "t_ever",
        "gone3", "foreclosed12", "cured12", "forbear", "state", "agency", "loan_purpose", "refi_type", "interest_rate", "spread", "mspread", "pmms", "T10",
        "T10_change", "upb", "opb", "orig_loan_term", "loan_age", "remaining_term", "ltv", "ltv_current", "credit_score",
        "num_borrower", "first_timer", "num_units", "third_party_orig_type", "issue_type", "pool_type", "Judicial",
        "annual_mip", "upfront_mip", "cum_missed_pmt_12", "prev_dlq_history", "down_pay_assist", "removal_reason", "total_months_post_dlq"]
df[keep].to_parquet(WORK / "panel.parquet", index=False)
w = df.ym.between(201901, 202209)
print(df.loc[w].groupby("itype").size())
print(df.loc[w & df.itype.isin(["depository", "nonbank"])].groupby(["period", "itype"]).x.agg(["mean", "size"]).unstack())
