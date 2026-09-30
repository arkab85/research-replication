"""Descriptive numbers quoted in the text that are not already in a table."""
import pandas as pd, numpy as np, os, json
from config import OUT

CTRL = ["coupon", "fico", "cltv", "age"]
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
p = p.dropna(subset=CTRL + ["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank", "techfirst"])].copy()
e = {}

# forbearance incidence at the threshold, 2020
f20 = p[p.ym.between(202001, 202009)].groupby("itype_ext").forbear.mean() * 100
print("forbearance incidence at the buyout threshold, 2020, by issuer type:")
print(f20.round(1))
for k, v in f20.items(): e[f"Fb{k.capitalize()}"] = f"{v:.1f}"

# monthly exercise range Mar-Sep 2020
g = p[p.ym.between(202003, 202009)].groupby(["ym", "itype_ext"]).buyout.mean().unstack() * 100
print("\nMar-Sep 2020 monthly exercise range:")
for t in ["depository", "nonbank"]:
    print(f"  {t}: {g[t].min():.0f} to {g[t].max():.0f}")
    e[f"Range{t.capitalize()}Lo"] = f"{g[t].min():.0f}"
    e[f"Range{t.capitalize()}Hi"] = f"{g[t].max():.0f}"

# coupon-bin changes
B = [0, 3.5, 4, 4.5, 5, 5.5, 99]
L = ["<3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0", "5.0-5.5", "5.5+"]
p["cbin"] = pd.cut(p.coupon, B, labels=L, right=False)
q = p[p.itype_ext.isin(["depository", "nonbank"])]
a = q[q.ym.between(201901, 201912)].pivot_table(index="cbin", columns="itype_ext",
                                                values="buyout", observed=True) * 100
b = q[q.ym.between(202003, 202009)].pivot_table(index="cbin", columns="itype_ext",
                                                values="buyout", observed=True) * 100
ch = (b - a).round(1)
print("\nchange by coupon bin (pp):"); print(ch)
print("depository 2019 monotone:", bool((a.depository.diff().dropna() > 0).all()))
print("depository 2020 monotone:", bool((b.depository.diff().dropna() > 0).all()))
print("nonbank 2019 monotone:", bool((a.nonbank.diff().dropna() > 0).all()))
print("nonbank 2020 monotone:", bool((b.nonbank.diff().dropna() > 0).all()))
nb = ch.nonbank
e["NbFallBot"] = f"{abs(nb.iloc[0]):.1f}"
e["NbFallPeak"] = f"{abs(nb.min()):.1f}"
e["NbFallPeakBin"] = str(nb.idxmin()).replace("-", "--")
e["NbFallTop"] = f"{abs(nb.iloc[-1]):.1f}"
gap19 = (a.depository - a.nonbank); gap20 = (b.depository - b.nonbank)
e["GapLowNineteen"] = f"{gap19.min():.0f}"; e["GapHighNineteen"] = f"{gap19.max():.0f}"
e["GapLowTwenty"] = f"{gap20.min():.0f}"; e["GapHighTwenty"] = f"{gap20.max():.0f}"

# nonbanks that never exercise
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
nbi = iss[iss.ext == "nonbank"]
e["NbTotal"] = str(len(nbi)); e["NbInactive"] = str(int((nbi.ebo19 < .02).sum()))
print(f"\nnonbank issuers {len(nbi)}, of which {int((nbi.ebo19<.02).sum())} "
      f"exercised on <2% of vested options in 2019")

# 2019 nonbank rise
t2 = p[p.itype_ext == "nonbank"].groupby("ym").buyout.mean() * 100
e["NbJanNineteen"] = f"{t2.loc[201901]:.1f}"; e["NbAugNineteen"] = f"{t2.loc[201908]:.1f}"

json.dump(e, open(os.path.join(OUT, "extra_numbers.json"), "w"), indent=1)
print("\nwrote extra_numbers.json:", json.dumps(e, indent=1))
