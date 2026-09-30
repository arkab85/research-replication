"""Check the proofread's claims against the macros and the data before editing."""
import re, os, pandas as pd, numpy as np
from config import OUT, PAPER as PAP

mac = {}
for f in ("numbers.tex", "numbers_pool.tex"):
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*$",
                         open(os.path.join(PAP, f), encoding="utf-8").read(), re.M):
        mac[m.group(1)] = m.group(2)

def show(*ks):
    for k in ks:
        print(f"    {k:<22} = {mac.get(k, '<< MISSING >>')}")

print("[2] DiD main vs within-pool table row")
show("nDiDMain", "pPoolMain", "pPoolMainSE", "pFlatMain")
print("[7] duplicate median-flow macros")
show("pTransferMed", "nTransMed")
print("[15] cutoff percentiles (check 'th' suffix bug)")
show("pCutNbPre", "pCutNbPost", "pCutDepPre", "pCutDepPost")
print("[18] within-pool share denominator")
show("pMixedShare", "pPoolN")
print("[24] pre-period coefficients")
show("pPoolNSig", "nEvMaxPre", "nEvMaxPost")

print("\n[8] Table 10 vs Table A.2: why do 2019 volumes differ?")
p = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
nm = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"].str.upper().str.strip()
pid = nm[nm == "PENNYMAC LOAN SERVICES, LLC"].index[0]
g = p[p.issuer_id == pid]
g19 = g[(g.ym // 100) == 2019]
print(f"    all 2019 rows                     {len(g19):>8,}")
print(f"    with non-missing buyout           {g19.buyout.notna().sum():>8,}")
cov = ["coupon", "fico", "cltv", "age"]
est = g19.dropna(subset=["buyout"] + cov)
print(f"    estimation sample (buyout+covs)   {len(est):>8,}   rate {est.buyout.mean()*100:.1f}%")
print("    -> Table A.2 uses the estimation sample; Table 10 uses all decisions.")

print("\n[11] Table 11: annual MAX vs annual MEAN of the quarterly series")
a = pd.read_csv(os.path.join(OUT, "pennymac_recovery.csv"), index_col=0, parse_dates=True)
yr = a.groupby(a.index.year).agg(mx=("elig", "max"), mn=("elig", "mean"),
                                 amx=("per_assets", "max"), amn=("per_assets", "mean"))
print(yr.round(2).to_string())
pre = a[(a.index >= "2018-01-01") & (a.index <= "2020-02-29")]
post = a[a.index >= "2022-01-01"]
print(f"    pre-shock mean of quarterly elig  {pre.elig.mean():.2f}  (sd {pre.elig.std():.2f})")
print(f"    2022+ mean of quarterly elig      {post.elig.mean():.2f}")
print(f"    pre-shock mean of per-assets      {pre.per_assets.mean():.3f} (sd {pre.per_assets.std():.3f})")

print("\n[1] active nonbanks with the deepest falls (from the issuer-level file)")
il = pd.read_csv(os.path.join(OUT, "issuer_level.csv"))
c = [x for x in il.columns]
print("    columns:", c[:12])
