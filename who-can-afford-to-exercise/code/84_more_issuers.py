"""Recover the other filers: BOK Financial (depository) and Ready Capital.
Then widen the tag net in case other Ginnie Mae issuers report the same concept
under a differently-named extension."""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT
warnings.filterwarnings("ignore")

d = pd.read_parquet(os.path.join(OUT, "sec_panel_raw.parquet"))
d["date"] = pd.to_datetime(d.ddate.astype(str), format="%Y%m%d", errors="coerce")
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

line("1. WHICH TAGS DOES EACH FILER USE, AND WITH WHAT qtrs?")
TAGS = [t for t in d.tag.unique() if "ligible" in t or "Repurchas" in t]
sub = d[d.tag.isin(TAGS)]
print(sub.groupby(["name", "tag", "qtrs"]).size().to_string())

line("2. SERIES FOR EVERY FILER, ANY qtrs")
res = {}
for (cik, nm), g in sub.groupby(["cik", "name"]):
    s = g.sort_values("date").groupby("date").value.max() / 1e9
    if len(s) < 6: continue
    yr = s.groupby(s.index.year).max()
    print(f"\n  {nm} (CIK {cik})  tags={sorted(g.tag.unique())}")
    for y, v in yr.items():
        bar = "#" * int(min(55, v / max(yr.max(), 1e-9) * 50))
        print(f"    {y}  ${v:8.3f}bn  {bar}")
    pre = s[(s.index >= "2019-01-01") & (s.index <= "2020-02-29")]
    pk = s[(s.index >= "2020-03-01") & (s.index <= "2021-06-30")]
    po = s[s.index >= "2022-01-01"]
    if len(pre) and len(pk):
        r = {"pre": round(float(pre.mean()), 3), "peak": round(float(pk.max()), 3),
             "peak_mult": round(float(pk.max() / pre.mean()), 1)}
        if len(po):
            r["post"] = round(float(po.mean()), 3)
            r["post_mult"] = round(float(po.mean() / pre.mean()), 1)
        print(f"    -> pre ${r['pre']}bn, peak ${r['peak']}bn ({r['peak_mult']}x)"
              + (f", 2022+ ${r['post']}bn ({r['post_mult']}x)" if len(po) else ""))
        res[nm] = r

line("3. PRE-SHOCK STABILITY: is there a trend before 2020?")
for (cik, nm), g in sub.groupby(["cik", "name"]):
    s = g.sort_values("date").groupby("date").value.max() / 1e9
    pre = s[(s.index >= "2017-01-01") & (s.index <= "2020-02-29")]
    if len(pre) < 6: continue
    x = np.arange(len(pre))
    b, a = np.polyfit(x, pre.values, 1)
    print(f"   {nm[:38]:<38} n={len(pre):>2}  mean ${pre.mean():6.3f}bn  "
          f"trend {b*4:+.3f}bn/yr  sd/mean {pre.std()/pre.mean():.2f}")
    res.setdefault(nm, {})["pre_trend_bn_yr"] = round(float(b * 4), 3)
    res.setdefault(nm, {})["pre_cv"] = round(float(pre.std() / pre.mean()), 2)

json.dump(res, open(os.path.join(OUT, "results_recovery_all.json"), "w"),
          indent=1, default=str)
print("\nwrote results_recovery_all.json")
