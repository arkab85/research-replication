"""What geography is actually available? Coordinates, counties, occupation mix, property values.
Reads only loan id, zip, county, lat/long from the property file - never names or street addresses."""
import os, re, pandas as pd, numpy as np
B = r"<DATA>/geo"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
cl = lambda c: re.sub(r"^[^A-Za-z]+", "", c)
KEEP = ("LoanID", "Zip", "County", "CountyFIPS", "fips", "Latitude", "Longitude")
g = pd.read_csv(os.path.join(B, "Geography.csv"), encoding="latin1", dtype=str, usecols=lambda c: cl(c) in KEEP); g.columns = [cl(c) for c in g.columns]
print("geography rows:", len(g), "| cols:", list(g.columns))
for c in ("Latitude", "Longitude"): g[c] = pd.to_numeric(g[c], errors="coerce")
print("lat/long non-missing:", g.Latitude.notna().mean().round(3), "| lat range:", g.Latitude.min(), g.Latitude.max(), "| lon range:", g.Longitude.min(), g.Longitude.max())
print("distinct counties:", g.CountyFIPS.nunique(), "| distinct zips:", g.Zip.nunique())
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); D.index = D.index.astype(str)
gg = g.drop_duplicates("LoanID").set_index("LoanID"); D = D.join(gg)
W = D[(D.r.abs() <= 14) & (D.Gov == 0)]; A = D[(D.Gov == 0)]
print("\nRD window conv loans:", len(W), "| with coords:", W.Latitude.notna().mean().round(3), "| counties:", W.CountyFIPS.nunique(), "| zips:", W.Zip.nunique())
for k in (3, 5, 8, 10):
    vc = W.CountyFIPS.value_counts(); print(f"  counties with >={k} window loans: {(vc>=k).sum()} covering {vc[vc>=k].sum()} loans")
print("all conv inquirers Mar-Sep:", len(A), "| counties:", A.CountyFIPS.nunique())
for k in (5, 10, 20):
    vc = A.CountyFIPS.value_counts(); print(f"  counties with >={k} conv inquirers: {(vc>=k).sum()} covering {vc[vc>=k].sum()}")
z = pd.read_csv(os.path.join(B, "Zip_Demo_Mod.csv"), encoding="latin1", nrows=3)
occ = [c for c in z.columns if c.startswith("OCC_") or "Collar" in c]; print("\noccupation columns:", occ)
hi = [c for c in z.columns if c.startswith("HI_")]; print("income bins:", len(hi))
print("value cols:", [c for c in z.columns if c.startswith("VAL_")][:8], "...")
print("other:", [c for c in z.columns if any(t in c.upper() for t in ("UNEMP", "EDU_", "POV", "RENT", "OWN", "VAC", "HH"))][:25])
m = pd.read_csv(r"<DATA>/panel\Monthly_Loan_Detail.csv", nrows=200000, dtype=str, usecols=["LoanID", "value", "value_date", "value_change", "zip", "current_income"] if True else None)
print("\nMonthly_Loan_Detail sample:", m.shape); print(m.head(3).to_string()); print("value non-missing:", m.value.notna().mean().round(3), "| distinct value_dates:", m.value_date.nunique())
