"""APM 20-07 and the value of the buyout.

On 29 June 2020 Ginnie Mae made any loan that entered COVID forbearance and is bought
out on or after 1 July 2020 ineligible for existing pool types; such loans can only be
re-securitised into a new, non-TBA "RG" pool after six timely payments and 210 days.
That is a cut to the VALUE of exercise, dated, mid-sample, and common to both issuer
types. It is the cleanest available test of the value channel against the capacity one:

  - a pure value story predicts BOTH types cut exercise after 1 July 2020
  - a capacity story predicts the type gap is unaffected by it

Splitting the post period at July 2020 also answers the objection that the 2020 estimate
is contaminated by a policy shock the paper does not model.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT, PAPER as PAP
warnings.filterwarnings("ignore")


def line(m):
    print("\n" + "=" * 92); print(m); print("=" * 92, flush=True)

p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p = p[p.ym.between(201901, 202009)].copy()
p["shock"] = ((p.ym >= 202003) & (p.ym <= 202006)).astype(int)   # Mar-Jun 2020
p["apm"] = (p.ym >= 202007).astype(int)                          # Jul-Sep 2020

line("1. EXERCISE RATE BY TYPE AND SUB-PERIOD")
tab = (p.groupby([p.itype_ext, pd.cut(p.ym, [201812, 201912, 202002, 202006, 202009],
                                      labels=["2019", "Jan-Feb 2020",
                                              "Mar-Jun 2020", "Jul-Sep 2020"])],
                 observed=True).buyout.agg(["mean", "size"]))
tab["mean"] = (tab["mean"] * 100).round(1)
print(tab.to_string())

line("2. DID THE REPOOLING RESTRICTION CUT EXERCISE FOR BOTH TYPES?")
for t in ["depository", "nonbank"]:
    x = p[p.itype_ext == t]
    a = x[x.ym.between(202003, 202006)].buyout.mean() * 100
    b = x[x.ym.between(202007, 202009)].buyout.mean() * 100
    print(f"  {t:<12} Mar-Jun {a:5.1f}%   Jul-Sep {b:5.1f}%   change {b-a:+5.1f}pp")
print("\n  A pure value shock common to both types should move both the same way.")

line("3. THE TYPE GAP, ESTIMATED SEPARATELY FOR EACH SUB-PERIOD")
q = p[p.itype_ext.isin(["depository", "nonbank"])].copy()
q["nb"] = (q.itype_ext == "nonbank").astype(int)
q["nb_shock"] = q.nb * q.shock
q["nb_apm"] = q.nb * q.apm
m = pf.feols("buyout ~ nb_shock + nb_apm + coupon + fico + cltv + age | issuer_id + ym",
             data=q, vcov={"CRV1": "issuer_id"})
res = {}
for k in ["nb_shock", "nb_apm"]:
    st = "***" if m.pvalue()[k] < .01 else "**" if m.pvalue()[k] < .05 else "*" if m.pvalue()[k] < .1 else ""
    print(f"  {k:<10} {m.coef()[k]:+.4f} ({m.se()[k]:.4f})  p={m.pvalue()[k]:.4f} {st}")
    res[k] = [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4), float(m.pvalue()[k])]
d = m.coef()["nb_apm"] - m.coef()["nb_shock"]
print(f"\n  difference (Jul-Sep minus Mar-Jun): {d:+.4f}")
print("  -> if the repooling restriction drove the result, the gap should WIDEN after July.")

line("4. WITHIN POOL x MONTH, SAME SPLIT")
cell = q.groupby(["pool_id", "ym"]).itype_ext.nunique()
mixed = cell[cell > 1].index
qm = q.set_index(["pool_id", "ym"]).loc[q.set_index(["pool_id", "ym"]).index.isin(mixed)].reset_index()
qm["pm"] = qm.pool_id.astype(str) + "_" + qm.ym.astype(str)
m2 = pf.feols("buyout ~ nb_shock + nb_apm + coupon + fico + cltv + age | issuer_id + pm",
              data=qm, vcov={"CRV1": "issuer_id"})
for k in ["nb_shock", "nb_apm"]:
    st = "***" if m2.pvalue()[k] < .01 else "**" if m2.pvalue()[k] < .05 else "*" if m2.pvalue()[k] < .1 else ""
    print(f"  {k:<10} {m2.coef()[k]:+.4f} ({m2.se()[k]:.4f})  p={m2.pvalue()[k]:.4f} {st}")
    res[k + "_pool"] = [round(float(m2.coef()[k]), 4), round(float(m2.se()[k]), 4),
                        float(m2.pvalue()[k])]
print(f"  N = {len(qm):,}")

# macros
def f(x, k=3):
    return f"{x:.{k}f}".replace("-", "$-$")

mac = {
    "ApmShock": f(m.coef()["nb_shock"]), "ApmShockSE": f"{m.se()['nb_shock']:.3f}",
    "ApmAfter": f(m.coef()["nb_apm"]), "ApmAfterSE": f"{m.se()['nb_apm']:.3f}",
    "ApmShockPool": f(m2.coef()["nb_shock"]), "ApmShockPoolSE": f"{m2.se()['nb_shock']:.3f}",
    "ApmAfterPool": f(m2.coef()["nb_apm"]), "ApmAfterPoolSE": f"{m2.se()['nb_apm']:.3f}",
}
for t, lab in [("depository", "Dep"), ("nonbank", "Nb")]:
    x = p[p.itype_ext == t]
    mac[f"Apm{lab}Pre"] = f"{x[x.ym.between(202003,202006)].buyout.mean()*100:.1f}"
    mac[f"Apm{lab}Post"] = f"{x[x.ym.between(202007,202009)].buyout.mean()*100:.1f}"
pth = os.path.join(PAP, "numbers_pool.tex")
have = open(pth, encoding="utf-8").read()
with open(pth, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
json.dump(res, open(os.path.join(OUT, "results_apm2007.json"), "w"), indent=1, default=str)
print(f"\nwrote {len(mac)} macros and results_apm2007.json")
