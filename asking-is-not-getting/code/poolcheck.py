import os, pandas as pd, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "out")
L = pd.read_parquet(os.path.join(OUT, "loan_frame.parquet")); K = L[L.in_covid_file == 1].copy()
K["Investor"] = K["Investor"].astype(str)
K["inq"] = (K.inq_first.notna() & (K.inq_first <= "2020-09-30")).astype(int); K["fb"] = (K.fb_agree.notna() & (K.fb_agree <= "2020-09-30")).astype(int)
print("Investor x Gov (linked loans):"); print(pd.crosstab(K.Investor, K.Gov, margins=True))
print("\nAssetType x Gov:"); print(pd.crosstab(K.AssetType, K.Gov, margins=True))
# deal names from Dashboard for a finer pool id
d = pd.read_csv(r"<DATA>/panel\Dashboard_Dec20_2020.csv", usecols=lambda c: c.strip().lstrip("\ufeff") in ("LoanID", "DealName", "AcquisitionDate", "RunDate"), dtype=str, nrows=6_000_000)
d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
deal = d.dropna(subset=["DealName"]).drop_duplicates("LoanID").set_index("LoanID")[["DealName", "AcquisitionDate"]]
K = K.join(deal); K["DealName"] = K.DealName.fillna("(none)")
print("\nDealName x Gov:"); print(pd.crosstab(K.DealName, K.Gov, margins=True))
g = K.groupby(["DealName", "Gov"]).agg(n=("fb", "size"), inq=("inq", "mean"), fb=("fb", "mean"), conv=("fb", lambda s: np.nan)).round(3)
g["conv"] = K[K.inq == 1].groupby(["DealName", "Gov"])["fb"].mean().round(3)
print("\nBy deal x Gov (n, inquiry, FB, FB|inq):"); print(g[g.n >= 30])
print("\nBy boarding year x Gov:"); K["by"] = K.board.dt.year
g2 = K.groupby(["by", "Gov"]).agg(n=("fb", "size"), inq=("inq", "mean"), fb=("fb", "mean")).round(3); g2["conv"] = K[K.inq == 1].groupby(["by", "Gov"])["fb"].mean().round(3); print(g2[g2.n >= 30])
print("\nBy loan_type:"); g3 = K.groupby("loan_type").agg(n=("fb", "size"), inq=("inq", "mean"), fb=("fb", "mean")).round(3); g3["conv"] = K[K.inq == 1].groupby("loan_type")["fb"].mean().round(3); print(g3)
print("\nTop states, conv by Gov:"); top = K.state.value_counts().index[:8]
print(K[(K.inq == 1) & K.state.isin(top)].groupby(["state", "Gov"])["fb"].agg(["mean", "size"]).round(3).unstack())
K[["DealName"]].to_parquet(os.path.join(OUT, "loan_deal.parquet"))
