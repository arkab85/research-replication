"""July 2020: two policy changes, opposite signs, and only one of them reaches nonbanks.

Two rules land at the same date and pull in opposite directions.

  APM 20-07 (Ginnie Mae, 29 June 2020, biting on buyouts from 1 July) cut the VALUE of
  exercise for every issuer: a loan bought out of COVID forbearance lost eligibility for
  existing pool types and could only be re-securitised into a non-TBA pool after six
  timely payments and 210 days.

  Lender Letter 2020-02 (Fannie Mae, 22 April 2020, effective with the quarter ending
  30 June) cut the CAPACITY cost of NOT exercising, for non-depositories only: a
  COVID-forbearance loan that was current when forbearance began counts at 30% rather
  than 100% toward the seriously delinquent balance that triggers the 200bp incremental
  liquidity charge. Depositories are not subject to the FHFA seller/servicer requirement.

So depositories receive the value cut alone; nonbanks receive the value cut plus the
capacity relief. Differencing across the July boundary nets out the common value shock
and leaves the capacity channel. That is a comparative static no value-based account can
reproduce: a shock that lowers the payoff to exercise for everyone cannot raise exercise
for the constrained type.

The specification writes the post period as a level plus an increment, so the increment
and its standard error come out of the regression directly rather than as a difference
of two coefficients.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
warnings.filterwarnings("ignore")
from config import OUT, PAPER as PAP

CTRL = ["coupon", "fico", "cltv", "age"]


def line(m):
    print("\n" + "=" * 92); print(m); print("=" * 92, flush=True)


p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank"])].copy()
p["nb"] = (p.itype_ext == "nonbank").astype(float)
res = {}

line("1. THE JULY 2020 BOUNDARY, BY TYPE")
for t in ["depository", "nonbank"]:
    x = p[p.itype_ext == t]
    a = x[x.ym.between(202003, 202006)].buyout.mean() * 100
    b = x[x.ym.between(202007, 202009)].buyout.mean() * 100
    print(f"  {t:<12} Mar-Jun {a:5.1f}%   Jul-Sep {b:5.1f}%   change {b-a:+5.1f}pp")
    res[f"{t}_change"] = round(float(b - a), 2)
print("\n  Depositories receive only the value cut and fall. Nonbanks receive the value")
print("  cut and the capacity relief, and rise.")

line("2. THE INCREMENT AT THE BOUNDARY, ESTIMATED DIRECTLY")
w = p[p.ym.between(201901, 202009)].copy()
w["nb_post"] = w.nb * (w.ym >= 202003)
w["nb_incr"] = w.nb * (w.ym >= 202007)
w["pm"] = w.pool_id.astype(str) + "_" + w.ym.astype(str)

cell = w.groupby(["pool_id", "ym"]).itype_ext.nunique()
mixed = set(cell[cell > 1].index)
wm = w[[(a, b) in mixed for a, b in zip(w.pool_id, w.ym)]]

for lab, dat, fe in [("issuer + month", w, "issuer_id + ym"),
                     ("issuer + pool x month", wm, "issuer_id + pm")]:
    m = pf.feols(f"buyout ~ nb_post + nb_incr + {' + '.join(CTRL)} | {fe}",
                 data=dat, vcov={"CRV1": "issuer_id"})
    print(f"\n  {lab}   N={int(m._N):,}")
    for k in ["nb_post", "nb_incr"]:
        st = "***" if m.pvalue()[k] < .01 else "**" if m.pvalue()[k] < .05 else "*" if m.pvalue()[k] < .1 else ""
        nm = "Nonbank x Post (Mar 2020)" if k == "nb_post" else "Nonbank x Jul 2020 increment"
        print(f"    {nm:<30} {m.coef()[k]:+.4f} ({m.se()[k]:.4f})  p={m.pvalue()[k]:.4f} {st}")
        res[f"{k}_{'pool' if 'pool' in lab else 'base'}"] = [
            round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4), float(m.pvalue()[k])]

line("3. PLACEBO: THE SAME JULY BOUNDARY ONE YEAR EARLIER")
q = p[p.ym.between(201810, 201912)].copy()
q["nb_post"] = q.nb * (q.ym >= 201903)
q["nb_incr"] = q.nb * (q.ym >= 201907)
mp = pf.feols(f"buyout ~ nb_post + nb_incr + {' + '.join(CTRL)} | issuer_id + ym",
              data=q, vcov={"CRV1": "issuer_id"})
k = "nb_incr"
print(f"  July 2019 increment  {mp.coef()[k]:+.4f} ({mp.se()[k]:.4f})  p={mp.pvalue()[k]:.4f}"
      f"   N={int(mp._N):,}")
print("  WARNING: this placebo is itself significant. See 129_dual_placebo.py, which")
print("  matches the window structure and finds the same magnitude one year earlier.")
print("  The July boundary cannot carry a causal reading, and the paper does not give")
print("  it one.")
res["placebo_july2019"] = [round(float(mp.coef()[k]), 4), round(float(mp.se()[k]), 4),
                           float(mp.pvalue()[k])]

line("4. WHAT THE VALUE SHOCK DID ON ITS OWN")
d = p[(p.itype_ext == "depository") & p.ym.between(201901, 202009)].copy()
d["post"] = (d.ym >= 202003).astype(int)
d["incr"] = (d.ym >= 202007).astype(int)
md = pf.feols(f"buyout ~ post + incr + {' + '.join(CTRL)} | issuer_id",
              data=d, vcov={"CRV1": "issuer_id"})
print(f"  depository-only July increment  {md.coef()['incr']:+.4f} ({md.se()['incr']:.4f})"
      f"  p={md.pvalue()['incr']:.4f}")
print("  This is the value channel alone, since the liquidity rule does not reach them.")
res["dep_value_only"] = [round(float(md.coef()["incr"]), 4), round(float(md.se()["incr"]), 4),
                         float(md.pvalue()["incr"])]


def num(x, k=3):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


mac = {
    "DualIncr": num(res["nb_incr_base"][0]), "DualIncrSE": f"{res['nb_incr_base'][1]:.3f}",
    "DualIncrP": f"{res['nb_incr_base'][2]:.3f}",
    "DualIncrPool": num(res["nb_incr_pool"][0]),
    "DualIncrPoolSE": f"{res['nb_incr_pool'][1]:.3f}",
    "DualPlac": num(res["placebo_july2019"][0]),
    "DualPlacSE": f"{res['placebo_july2019'][1]:.3f}",
    "DualPlacP": f"{res['placebo_july2019'][2]:.2f}",
    "DualDepVal": num(res["dep_value_only"][0]),
    "DualDepValSE": f"{res['dep_value_only'][1]:.3f}",
}
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k2, v in sorted(mac.items()):
        if "\\p" + k2 + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k2 + "}{" + v + "}\n")
json.dump(res, open(os.path.join(OUT, "results_dual_shock.json"), "w"), indent=1, default=str)
print(f"\nwrote {len(mac)} macros and results_dual_shock.json")
