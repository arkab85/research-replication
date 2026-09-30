"""Final recovery series: unexercised options scaled by the servicing book, to 2025."""
import urllib.request, json, os, warnings
import pandas as pd, numpy as np
from config import OUT, SEC_UA
warnings.filterwarnings("ignore")

H = {"User-Agent": SEC_UA, "Accept": "application/json"}

def concept(cik, tag, tax="us-gaap"):
    u = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{tax}/{tag}.json"
    try:
        j = json.loads(urllib.request.urlopen(
            urllib.request.Request(u, headers=H), timeout=60).read())
    except Exception:
        return pd.Series(dtype=float)
    rows = []
    for unit, recs in j.get("units", {}).items():
        if unit != "USD": continue
        for r in recs:
            if r.get("end") and r.get("val") is not None and "start" not in r:
                rows.append((r["end"], float(r["val"])))
    if not rows: return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["end", "v"]).drop_duplicates("end")
    s["end"] = pd.to_datetime(s.end)
    return s.set_index("end").v.sort_index()

CIK = "0001745916"          # PennyMac Financial Services
msr = concept(CIK, "ServicingAssetAtFairValueAmount")
ast = concept(CIK, "Assets")
print(f"MSR quarters {len(msr)} ({msr.index.min():%Y-%m} to {msr.index.max():%Y-%m})")
print(f"Assets quarters {len(ast)} ({ast.index.min():%Y-%m} to {ast.index.max():%Y-%m})")

d = pd.read_parquet(os.path.join(OUT, "sec_panel_raw.parquet"))
d["date"] = pd.to_datetime(d.ddate.astype(str), format="%Y%m%d", errors="coerce")
STOCK = {"LoansEligibleForRepurchases", "LoansEligibleForRepurchase",
         "LiabilityForLoansEligibleForRepurchase"}
e = (d[(d.cik == 1745916) & (d.qtrs == 0) & d.tag.isin(STOCK)]
     .groupby("date").value.max() / 1e9)

a = pd.DataFrame({"elig": e, "msr": msr / 1e9, "assets": ast / 1e9}).sort_index()
a = a[a.elig.notna()]
a["msr"] = a.msr.ffill(); a["assets"] = a.assets.ffill()
a["per_msr"] = a.elig / a.msr
a["per_assets"] = a.elig / a.assets
print("\n" + "=" * 78)
print("PENNYMAC: UNEXERCISED GINNIE MAE BUYOUT OPTIONS, 2017-2025")
print("=" * 78)
yr = a.groupby(a.index.year).agg(elig=("elig", "max"), msr=("msr", "last"),
                                 per_msr=("per_msr", "max"),
                                 per_assets=("per_assets", "max"))
for y, r in yr.iterrows():
    bar = "#" * int(min(46, (r.per_msr / max(yr.per_msr.max(), 1e-9)) * 42))
    print(f"  {y}  ${r.elig:7.2f}bn   MSR ${r.msr:6.2f}bn   per MSR {r.per_msr:5.2f}"
          f"   per assets {r.per_assets:5.2f}  {bar}")

pre = a[(a.index >= "2018-01-01") & (a.index <= "2020-02-29")]
pk = a[(a.index >= "2020-03-01") & (a.index <= "2021-06-30")]
po = a[a.index >= "2022-01-01"]
res = {}
for lab, col in [("per MSR", "per_msr"), ("per assets", "per_assets"),
                 ("dollars (bn)", "elig")]:
    P, K, O = pre[col], pk[col], po[col]
    x = np.arange(len(P)); b, _ = np.polyfit(x, P.values, 1)
    print(f"\n  {lab}")
    print(f"    2018-Feb 2020 mean {P.mean():7.2f}  sd {P.std():5.2f}  "
          f"trend {b*4:+.3f}/yr")
    print(f"    2020-H1 2021 peak  {K.max():7.2f}  ({K.max()/P.mean():4.1f}x)")
    print(f"    2022 onward mean   {O.mean():7.2f}  ({O.mean()/P.mean():4.1f}x)")
    res[lab] = {"pre": round(float(P.mean()), 2), "pre_sd": round(float(P.std()), 2),
                "pre_trend": round(float(b * 4), 3), "peak": round(float(K.max()), 2),
                "peak_mult": round(float(K.max() / P.mean()), 1),
                "post": round(float(O.mean()), 2),
                "post_mult": round(float(O.mean() / P.mean()), 1)}
a.to_csv(os.path.join(OUT, "pennymac_recovery.csv"))
json.dump(res, open(os.path.join(OUT, "results_recovery_final.json"), "w"),
          indent=1, default=str)
print("\nwrote pennymac_recovery.csv, results_recovery_final.json")
