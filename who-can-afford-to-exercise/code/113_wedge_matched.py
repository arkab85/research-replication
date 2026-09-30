"""Rebuild the wedge on matched windows, and drop the gross measure from it.

Two defects the reviewer found in the first version:
  (1) for December-reporting firms the flow was calendar 2019 (12 months) against
      January-September 2020 (9 months, because the extract ends in September), which
      understates flow growth and therefore INFLATES the wedge;
  (2) JPMorgan reports repurchased and unexercised loans combined, which the text says
      is excluded from the wedge, and then includes it.

Fix: count the flow over January-September in BOTH years for every firm, use a
30 September balance wherever the filer reports one, and compute the headline wedge
only on issuers whose disclosure is net of repurchases. JPMorgan is reported beside it
as a directional observation, not inside the statistic.
"""
import pandas as pd, numpy as np, os, json, warnings
from config import OUT, PAPER as PAP
warnings.filterwarnings("ignore")

# quarterly PennyMac series gives a matched 30 September balance
pm = pd.read_csv(os.path.join(OUT, "pennymac_recovery.csv"), index_col=0, parse_dates=True)


def q3(y):
    s = pm[(pm.index >= f"{y}-08-15") & (pm.index <= f"{y}-10-15")]
    return float(s.elig.iloc[0]) if len(s) else np.nan


# A filer's balance sheet consolidates every Ginnie Mae issuer number it operates, so
# the flow must be summed over all of them. Mr. Cooper runs two.
FIRMS = {
    "PennyMac Financial": ([4094],       True,  "30 Sep", q3(2019), q3(2020)),
    "Mr.\\ Cooper Group": ([4052, 4126], True,  "31 Dec", 0.560,    6.159),
    "Caliber Home Loans": ([4213],       True,  "30 Sep", 0.195,    1.918),
    "AmeriHome":          ([3359],       True,  "30 Sep", 0.383,    2.426),
    "Flagstar Bank":      ([3345],       True,  "31 Dec", 0.070,    1.851),
    "JPMorgan Chase":     ([3975],       False, "31 Dec", 2.941,    1.413),
}
TYPE = {"PennyMac Financial": "nonbank", "Mr.\\ Cooper Group": "nonbank",
        "Caliber Home Loans": "nonbank", "AmeriHome": "nonbank",
        "Flagstar Bank": "depository", "JPMorgan Chase": "depository"}
print(f"PennyMac 30 Sep balances: 2019 ${q3(2019):.2f}bn, 2020 ${q3(2020):.2f}bn")

p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p["yr"] = (p.ym // 100).astype(int)
p["mo"] = (p.ym % 100).astype(int)
nm = pd.read_csv(os.path.join(OUT, "issuer_id_names.csv"), index_col=0)["name"].str.upper().str.strip()

rows = []
for ent, (ids, net, asof, b19, b20) in FIRMS.items():
    g = p[p.issuer_id.isin(ids)]
    print(f"  {ent.replace(chr(92)+' ',' '):<20} ids {ids} -> {len(g):,} decisions")
    r = {"entity": ent, "net": net, "asof": asof, "type": TYPE[ent],
         "rep2019": b19, "rep2020": b20}
    for y in (2019, 2020):                      # January-September in BOTH years
        h = g[(g.yr == y) & (g.mo <= 9)]
        r[f"vested{y}"] = len(h)
        r[f"rate{y}"] = h.buyout.mean()
    r["flow_mult"] = r["vested2020"] / r["vested2019"]
    r["rep_mult"] = r["rep2020"] / r["rep2019"]
    r["wedge"] = r["rep_mult"] / r["flow_mult"]
    r["d_rate"] = (r["rate2020"] - r["rate2019"]) * 100
    rows.append(r)
d = pd.DataFrame(rows)

print("\n" + "=" * 94)
print("MATCHED-WINDOW WEDGE (flow counted Jan-Sep in both years)")
print("=" * 94)
for _, r in d.iterrows():
    tag = "" if r.net else "   <- gross of repurchases, excluded from the statistic"
    print(f"  {r.entity.replace(chr(92)+' ',' '):<20} {r['asof']}  balance {r.rep_mult:5.1f}x  "
          f"flow {r.flow_mult:4.2f}x  wedge {r.wedge:5.2f}  "
          f"exercise {r.d_rate:+6.1f}pp{tag}")

net = d[d.net]
c = np.corrcoef(np.log(net.wedge), -net.d_rate)[0, 1]
call = np.corrcoef(np.log(d.wedge), -d.d_rate)[0, 1]
print(f"\n  net-of-repurchase issuers only (n={len(net)}): corr(log wedge, fall) = {c:+.3f}")
print(f"  including JPMorgan            (n={len(d)}): corr(log wedge, fall) = {call:+.3f}")
print(f"  all net issuers have wedge > 1: {bool((net.wedge > 1).all())}")
b = np.polyfit(-net.d_rate, np.log(net.wedge), 1)
print(f"  a 10pp larger fall -> wedge {np.exp(b[0]*10):.2f}x larger")

# what the old, mismatched windows gave, for the record
old = pd.read_csv(os.path.join(OUT, "four_firm.csv"))
print(f"\n  for comparison, the mismatched-window wedges were "
      f"{old.wedge.min():.1f}-{old.wedge.max():.1f}; matched they are "
      f"{net.wedge.min():.1f}-{net.wedge.max():.1f}")

# stock/flow conversion: is it stable? (the reviewer says no, and is right)
print("\n  loan-level cumulative UPB / reported stock, by firm-year:")
for _, r in d.iterrows():
    g = p[p.issuer_id.isin(FIRMS[r.entity][0])]
    for y in (2019, 2020):
        h = g[(g.yr == y) & (g.mo <= 9)]
        u = h.loc[h.buyout == 0, "upb"].sum() / 1e9
        print(f"    {r.entity.replace(chr(92)+' ',' '):<20} {y}  "
              f"cumulative {u:6.2f}  stock {r[f'rep{y}']:6.2f}  ratio "
              f"{u / r[f'rep{y}']:5.2f}")

d.to_csv(os.path.join(OUT, "wedge_matched.csv"), index=False)


def fm(x, k=2):
    return f"{x:.{k}f}".replace("-", "$-$")


mac = {"WmN": f"{len(net)}", "WmCorr": f"{c:+.2f}".replace("-", "$-$"),
       "WmCorrAll": f"{call:+.2f}".replace("-", "$-$"),
       "WmMin": f"{net.wedge.min():.1f}", "WmMax": f"{net.wedge.max():.1f}",
       "WmFlowMin": f"{net.flow_mult.min():.1f}", "WmFlowMax": f"{net.flow_mult.max():.1f}",
       "WmRepMin": f"{net.rep_mult.min():.1f}", "WmRepMax": f"{net.rep_mult.max():.0f}",
       "WmSlope": f"{np.exp(b[0]*10):.1f}",
       "WmPmQthree": f"{q3(2019):.2f}", "WmPmQthreeTwenty": f"{q3(2020):.2f}",
       "WmJpmWedge": f"{d[~d.net].wedge.iloc[0]:.1f}"}
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
json.dump(mac, open(os.path.join(OUT, "results_wedge_matched.json"), "w"), indent=1)
print(f"\nwrote wedge_matched.csv and {len(mac)} macros")
