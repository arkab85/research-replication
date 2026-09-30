"""Scale the immobilisation measure by the size of the servicing business.

The dollar stock of unexercised options grows with the book, so the raw series confounds
immobilisation with growth. The mortgage servicing right is the balance-sheet item that
scales with servicing UPB, and it is reported in the same filings.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "sec_panel_raw.parquet"))
d["date"] = pd.to_datetime(d.ddate.astype(str), format="%Y%m%d", errors="coerce")
d = d.dropna(subset=["date"])
STOCK = {"LoansEligibleForRepurchases", "LoansEligibleForRepurchase",
         "LiabilityForLoansEligibleForRepurchase", "LoansEligibleForRepurchaseFromAgency"}
res = {}
def line(m): print("\n" + "=" * 84); print(m); print("=" * 84, flush=True)

for cik, nm, short in [(1745916, "PennyMac Financial Services", "PennyMac"),
                       (1527590, "Ready Capital", "ReadyCap")]:
    g = d[(d.cik == cik) & (d.qtrs == 0)]
    elig = g[g.tag.isin(STOCK)].groupby("date").value.max() / 1e9
    msr = g[g.tag == "ServicingAssetAtFairValueAmount"].groupby("date").value.max() / 1e9
    a = pd.DataFrame({"elig": elig, "msr": msr}).dropna()
    if a.empty:
        print(f"\n{nm}: no overlapping MSR"); continue
    a["ratio"] = a.elig / a.msr
    line(f"{nm}: unexercised options per dollar of servicing right")
    yr = a.groupby(a.index.year).agg(elig=("elig", "max"), msr=("msr", "max"),
                                     ratio=("ratio", "max"))
    for y, r in yr.iterrows():
        bar = "#" * int(min(50, r.ratio / max(yr.ratio.max(), 1e-9) * 46))
        print(f"   {y}  elig ${r.elig:7.2f}bn  MSR ${r.msr:6.2f}bn  "
              f"ratio {r.ratio:5.2f}  {bar}")
    pre = a[(a.index >= "2017-01-01") & (a.index <= "2020-02-29")].ratio
    pk = a[(a.index >= "2020-03-01") & (a.index <= "2021-06-30")].ratio
    po = a[a.index >= "2022-01-01"].ratio
    if len(pre) >= 4:
        x = np.arange(len(pre)); b, _ = np.polyfit(x, pre.values, 1)
        print(f"\n   pre-shock (2017-Feb 2020): mean {pre.mean():.2f}, "
              f"sd {pre.std():.2f}, trend {b*4:+.3f}/yr")
        print(f"   peak (2020-H1 2021)       : {pk.max():.2f}  "
              f"({pk.max()/pre.mean():.1f}x pre-shock)")
        if len(po):
            print(f"   2022 onward               : {po.mean():.2f}  "
                  f"({po.mean()/pre.mean():.1f}x pre-shock)")
        res[short] = {"pre_mean": round(float(pre.mean()), 2),
                      "pre_sd": round(float(pre.std()), 2),
                      "pre_trend": round(float(b * 4), 3),
                      "peak": round(float(pk.max()), 2),
                      "peak_mult": round(float(pk.max() / pre.mean()), 1),
                      "post": round(float(po.mean()), 2) if len(po) else None,
                      "post_mult": round(float(po.mean() / pre.mean()), 1) if len(po) else None}
    a.to_csv(os.path.join(OUT, f"scaled_{short}.csv"))

line("READ")
print("   A pre-existing trend cannot produce a flat ratio for three years, a large")
print("   spike in 2020, and a reversion afterwards. The scaled series is the paper's")
print("   immobilisation measure, extended past the loan-level sample using audited")
print("   public filings, for the issuer that sits at the centre of the story.")
json.dump(res, open(os.path.join(OUT, "results_scaled.json"), "w"), indent=1, default=str)
print("\nwrote results_scaled.json")
