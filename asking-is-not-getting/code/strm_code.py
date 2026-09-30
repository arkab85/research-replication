"""The servicer's own product and loss-mitigation codes across 7 April 2020, loan level (last snapshot row per loan). Structured fields only; no text."""
import os, json, numpy as np, pandas as pd, statsmodels.formula.api as smf
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out")
p = r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv"
d = pd.read_csv(p, dtype=str, na_values=["NULL"], usecols=["LoanID", "ModificationType", "ModificationDate", "LSMIT_Status", "CreatedAt"]); d["c"] = pd.to_datetime(d.CreatedAt, errors="coerce")
ever = d.assign(strm=d.ModificationType.fillna("").str.contains("PNDMIC STRM", case=False), docs=d.LSMIT_Status.fillna("").str.contains("Trailing Docs|Incomplete Letter|File Received for UW", case=False), reinst=d.LSMIT_Status.fillna("").str.contains("Reinstatement", case=False)).groupby("LoanID")[["strm", "docs", "reinst"]].max().astype(float)
md = d[d.ModificationType.fillna("").str.contains("PNDMIC STRM", case=False)].assign(m=lambda x: pd.to_datetime(x.ModificationDate, errors="coerce")).groupby("LoanID").m.min()
D = pd.read_parquet(os.path.join(OUT, "rd_frame.parquet")); D.index = D.index.astype(str); D = D.join(ever).fillna({"strm": 0, "docs": 0, "reinst": 0}); D["strm_date"] = md.reindex(D.index)
star = lambda p: "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
def ll(x, y, h=14):
    x = x[x.r.abs() <= h].copy(); x["w"] = 1 - x.r.abs() / (h + 1); m = smf.wls(f"{y} ~ post + r + post:r", x, weights=x.w).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.r)[0]}); return 100 * m.params["post"], 100 * m.bse["post"], m.pvalues["post"]
R = {}
for g, nm in [(0, "Conv"), (1, "Gov")]:
    x = D[(D.Gov == g) & (D.r.abs() <= 14)]; print(f"\n{nm}: loans in window {len(x)}")
    for y, lab in [("strm", "pandemic streamlined deferral code"), ("docs", "document-chase status (trailing docs / incomplete letter / file to underwriting)"), ("reinst", "reinstatement status")]:
        a, b = 100 * x[x.post == 0][y].mean(), 100 * x[x.post == 1][y].mean(); j = ll(D[D.Gov == g], y); print(f"  {lab}: before {a:.1f}%  after {b:.1f}%  | local linear jump {j[0]:.1f} ({j[1]:.1f}){star(j[2])}"); R[f"{nm}_{y}"] = [a, b, j[0], j[1]]
x = D[(D.Gov == 0) & (D.r.abs() <= 14)]; print("\nConventional, by day of first inquiry (share with the streamlined-deferral code):"); t = x.groupby("r").agg(n=("strm", "size"), strm=("strm", "mean"), fb=("fb", "mean")); t = t[t.n >= 8]; print((t.assign(strm=(100 * t.strm).round(0), fb=(100 * t.fb).round(0))).loc[-8:8].to_string())
s = x[x.strm == 1]; print("\nAmong conventional loans with the code: share that received forbearance:", round(100 * s.fb.mean(), 1), "| share of forbearance recipients with the code:", round(100 * x[x.fb == 1].strm.mean(), 1)); print("modification dates on those loans:", s.strm_date.dt.to_period("M").value_counts().sort_index().to_dict())
R["fb_given_code"] = 100 * s.fb.mean(); R["code_given_fb"] = 100 * x[x.fb == 1].strm.mean(); json.dump(R, open(os.path.join(OUT, "strm_code.json"), "w"))
