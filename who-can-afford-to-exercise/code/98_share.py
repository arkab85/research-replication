"""How much of the nonbank side of the sample do the four filers cover?"""
import pandas as pd, os
from config import OUT, PAPER as PAP

p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
nm = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"].str.upper().str.strip()
FOUR = ["PENNYMAC LOAN SERVICES, LLC", "NATIONSTAR MORTGAGE, LLC",
        "CALIBER HOME LOANS, INC.", "AMERIHOME MORTGAGE COMPANY,LLC"]
ids = nm[nm.isin(FOUR)].index
nb = p[p.nonbank == 1]
share = nb.issuer_id.isin(ids).mean() * 100
share_upb = nb.loc[nb.issuer_id.isin(ids), "upb"].sum() / nb.upb.sum() * 100
print(f"  four filers = {share:.0f}% of classified nonbank decisions, "
      f"{share_upb:.0f}% of nonbank delinquent principal")

v = f"{share:.0f}\\%"
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
if "\\pWedgeShare}" not in have:
    open(pth, "a", encoding="utf-8").write("\n\\newcommand{\\pWedgeShare}{" + v + "}\n")
    print("  wrote \\pWedgeShare =", v)
