"""Does the July 2020 increment survive a placebo on the same calendar boundary?

128 found a July 2020 increment of +0.072 (0.047) and a July 2019 increment of
+0.064 (0.026), but the two used windows of different lengths, which is not a fair
comparison. This re-runs the placebo with the window structure matched exactly:
fourteen months before the March break and seven months after, shifted back one year.
It also runs the boundary at every month of 2019 and 2020, so the July 2020 increment
can be read against the distribution of increments at other dates rather than against
one placebo.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
warnings.filterwarnings("ignore")
from config import OUT

CTRL = ["coupon", "fico", "cltv", "age"]
p = pd.read_parquet(os.path.join(OUT, "panel.parquet")).dropna(subset=["buyout"])
p = p[p.itype_ext.isin(["depository", "nonbank"])].copy()
p["nb"] = (p.itype_ext == "nonbank").astype(float)


def ymrange(lo, hi):
    return p[p.ym.between(lo, hi)].copy()


def fit(dat, brk, inc):
    dat = dat.copy()
    dat["nb_post"] = dat.nb * (dat.ym >= brk)
    dat["nb_incr"] = dat.nb * (dat.ym >= inc)
    m = pf.feols(f"buyout ~ nb_post + nb_incr + {' + '.join(CTRL)} | issuer_id + ym",
                 data=dat, vcov={"CRV1": "issuer_id"})
    return (float(m.coef()["nb_incr"]), float(m.se()["nb_incr"]),
            float(m.pvalue()["nb_incr"]), int(m._N))


print("MATCHED WINDOWS: 14 months before the March break, 7 months after")
real = fit(ymrange(201901, 202009), 202003, 202007)
plac = fit(ymrange(201801, 201909), 201903, 201907)
print(f"  July 2020 increment  {real[0]:+.4f} ({real[1]:.4f})  p={real[2]:.3f}  N={real[3]:,}")
print(f"  July 2019 increment  {plac[0]:+.4f} ({plac[1]:.4f})  p={plac[2]:.3f}  N={plac[3]:,}")
diff = real[0] - plac[0]
print(f"  difference           {diff:+.4f}")

print("\nINCREMENT AT EVERY BOUNDARY, 2020 window (break fixed at March 2020)")
rows = []
for inc in [202004, 202005, 202006, 202007, 202008, 202009]:
    c, s, pv, n = fit(ymrange(201901, 202009), 202003, inc)
    rows.append((inc, c, s, pv))
    star = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    mark = "   <- APM 20-07 / LL-2020-02" if inc == 202007 else ""
    print(f"  {inc}  {c:+.4f} ({s:.4f})  p={pv:.3f} {star}{mark}")

print("\nSAME, ONE YEAR EARLIER (break fixed at March 2019)")
for inc in [201904, 201905, 201906, 201907, 201908, 201909]:
    c, s, pv, n = fit(ymrange(201801, 201909), 201903, inc)
    star = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    mark = "   <- placebo July" if inc == 201907 else ""
    print(f"  {inc}  {c:+.4f} ({s:.4f})  p={pv:.3f} {star}{mark}")

print("\nREAD")
if real[2] < 0.05 and plac[2] > 0.10:
    print("  The July 2020 increment is significant and the placebo is not. The design holds.")
elif plac[2] < 0.10:
    print("  The placebo increment is itself significant at the same calendar boundary.")
    print("  A July effect exists in a year with no policy change, so the July 2020")
    print("  movement CANNOT be attributed to the July 2020 rules. The design fails.")
else:
    print("  The July 2020 increment is not distinguishable from zero. Nothing to attribute.")

json.dump({"real": real, "placebo": plac, "by_boundary": rows},
          open(os.path.join(OUT, "results_dual_placebo.json"), "w"), indent=1, default=str)
print("\nwrote results_dual_placebo.json")
