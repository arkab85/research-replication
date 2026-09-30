"""Monthly eligible stock. From the monthly histories of seriously delinquent loans, keep every
loan-month from 2019m1 to 2022m12 in which the loan is at least three payments delinquent (so the
issuer may repurchase it), flag repurchase in that month (removal reason 2), and attach the loan's
ranking index from its vesting record."""
import pandas as pd, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, glob, datetime as dt
from config import PANEL, WORK
files = sorted(glob.glob(str(WORK / "panel_slim_00*.parquet")))            # written by 00_slim_panel.py
if not files:   # blocks converted on the author's machine (same fields, same filter)
    files = sorted(glob.glob(str(PANEL / "p002_c*.parquet"))) + sorted(glob.glob(str(PANEL / "p001_00.parquet")))
parts = []
cut = pa.scalar(dt.date(2019, 1, 1), type=pa.date32())
for f in files:
    t = pq.read_table(f, columns=["identifier", "issuer_id", "as_of_date", "months_dlq", "removal_reason", "fb_flag", "upb"])
    msk = pc.and_(pc.greater_equal(t["as_of_date"], cut),
                  pc.or_(pc.greater_equal(pc.fill_null(t["months_dlq"], 0), 3), pc.equal(pc.fill_null(t["removal_reason"], 0), 2)))
    t = t.filter(msk)
    parts.append(t.to_pandas())
    print(f.split("/")[-1], t.num_rows, flush=True)
s = pd.concat(parts, ignore_index=True); del parts
s["ym"] = pd.to_datetime(s.as_of_date).dt.strftime("%Y%m").astype(int)
s["ex"] = s.removal_reason.eq(2).astype(np.int8)
s["fb"] = s.fb_flag.eq("Y").astype(np.int8)
print("rows", len(s), "repurchases", int(s.ex.sum()))
print("months_dlq among repurchases", s.loc[s.ex == 1, "months_dlq"].value_counts().head(8).to_dict())
sc = pd.read_parquet(WORK / "scored.parquet", columns=["identifier", "itype", "r_main", "r_gb", "ym", "spread", "upb"]).rename(columns={"ym": "vest_ym", "upb": "upb_vest"})
sc = sc.drop_duplicates("identifier")
s = s.merge(sc, on="identifier", how="inner")
s = s[s.months_dlq.fillna(0).ge(3) | s.ex.eq(1)]
s = s.drop(columns=["as_of_date", "fb_flag", "removal_reason"])
s.to_parquet(WORK / "stock.parquet", index=False)
print("merged rows", len(s), s.groupby([s.ym // 100, "itype"]).ex.mean().unstack().round(4))
