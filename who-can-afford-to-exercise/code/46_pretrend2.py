"""How much of the pre-trend problem survives the within-pool design?

Three things:
 1. joint test on the pre-period coefficients
 2. placebo breaks at every date in the pre-period, not just March 2019
 3. Rambachan-Roth breakdown value on the pool x month event study. Smaller pre-period
    violations mean the relative-magnitudes bound binds at a larger M, so if the within-
    pool design has genuinely flattened the pre-period the breakdown value should rise.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")
from honestdid import constructOriginalCS, createSensitivityResults_relativeMagnitudes

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 86); print(m); print("=" * 86, flush=True)

d = pd.read_parquet(os.path.join(OUT, "mixed_pools.parquet"))
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL: d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=CTRL + ["buyout"]).copy()
d["nonbank"] = (d.ext == "nonbank").astype(float)
d["post"] = (d.ym >= 202003).astype(float)
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)
k = d.groupby("pm").nonbank.transform(lambda s: s.nunique()); d = d[k > 1].copy()

line("1. JOINT TEST ON THE PRE-PERIOD COEFFICIENTS")
months = sorted(d.ym.unique())
for m_ in months:
    if m_ != 202002: d[f"d{m_}"] = d.nonbank * (d.ym == m_)
dc = [f"d{m_}" for m_ in months if m_ != 202002]
es = pf.feols(f"buyout ~ {' + '.join(dc)} + {' + '.join(CTRL)} | issuer_id + pm",
              data=d, vcov={"CRV1": "issuer_id"})
pre_c = [c for c in dc if int(c[1:]) <= 202001]
try:
    w = es.wald_test(R=np.array([[1.0 if cn == pc else 0.0
                                  for cn in es._coefnames] for pc in pre_c]))
    print("   Wald test, all pre-period coefficients = 0:", w)
except Exception as e:
    # manual Wald from the clustered vcov
    idx = [list(es._coefnames).index(c) for c in pre_c]
    b = np.asarray(es.coef())[idx]
    V = np.asarray(es._vcov)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.pinv(V) @ b)
    from scipy.stats import chi2
    pv = 1 - chi2.cdf(stat, len(idx))
    print(f"   chi2({len(idx)}) = {stat:.2f}   p = {pv:.4f}")
    res["joint_pre"] = {"chi2": round(stat, 2), "df": len(idx), "p": round(float(pv), 4)}

line("2. PLACEBO BREAKS AT EVERY PRE-PERIOD DATE")
P = {}
for brk in [201904, 201905, 201906, 201907, 201908, 201909, 201910, 201911, 201912, 202001]:
    s = d[d.ym < 202003].copy()
    s["pb"] = (s.ym >= brk).astype(float)
    s["nb_pb"] = s.nonbank * s.pb
    kk = s.groupby("pm").nonbank.transform(lambda z: z.nunique()); s = s[kk > 1]
    if s.pb.nunique() < 2: continue
    m = pf.feols(f"buyout ~ nb_pb + {' + '.join(CTRL)} | issuer_id + pm",
                 data=s, vcov={"CRV1": "issuer_id"})
    c, se, p = m.coef()["nb_pb"], m.se()["nb_pb"], m.pvalue()["nb_pb"]
    st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    print(f"   break {brk}: {c:+.4f} ({se:.4f}) p={p:.3f} {st:<3} N={int(m._N):>8,}")
    P[str(brk)] = [round(float(c), 4), round(float(se), 4), float(p)]
res["placebos"] = P

line("3. THE SAME EXERCISE ON THE SHORT, FLAT WINDOW (Jul 2019 - Sep 2020)")
s = d[d.ym >= 201907].copy()
kk = s.groupby("pm").nonbank.transform(lambda z: z.nunique()); s = s[kk > 1]
s["nb_post"] = s.nonbank * s.post
m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + pm",
             data=s, vcov={"CRV1": "issuer_id"})
print(f"   Nonbank x Post = {m.coef()['nb_post']:+.4f} ({m.se()['nb_post']:.4f}) "
      f"p={m.pvalue()['nb_post']:.4f}  N={int(m._N):,}")
res["short_window"] = [round(float(m.coef()["nb_post"]), 4),
                       round(float(m.se()["nb_post"]), 4),
                       float(m.pvalue()["nb_post"])]
# placebo inside the flat window
sp = s[s.ym < 202003].copy()
sp["pb"] = (sp.ym >= 201911).astype(float); sp["nb_pb"] = sp.nonbank * sp.pb
kk = sp.groupby("pm").nonbank.transform(lambda z: z.nunique()); sp = sp[kk > 1]
mp = pf.feols(f"buyout ~ nb_pb + {' + '.join(CTRL)} | issuer_id + pm",
              data=sp, vcov={"CRV1": "issuer_id"})
print(f"   placebo inside that window (break Nov 2019) = {mp.coef()['nb_pb']:+.4f} "
      f"({mp.se()['nb_pb']:.4f}) p={mp.pvalue()['nb_pb']:.3f}")
res["short_placebo"] = [round(float(mp.coef()["nb_pb"]), 4),
                        round(float(mp.se()["nb_pb"]), 4), float(mp.pvalue()["nb_pb"])]

line("4. RAMBACHAN-ROTH BREAKDOWN VALUE, WITHIN POOL x MONTH")
idx = [list(es._coefnames).index(c) for c in dc]
beta = np.asarray(es.coef())[idx]
V = np.asarray(es._vcov)[np.ix_(idx, idx)]
np.save(os.path.join(OUT, "event_vcov_poolFE.npy"), V)
ms = [int(c[1:]) for c in dc]
nP = sum(1 for x in ms if x <= 202001); nQ = len(ms) - nP
l = np.zeros(nQ); l[[i for i, x in enumerate(ms) if x >= 202003].index(
    [i for i, x in enumerate(ms) if x == 202006][0] - nP)] = 1.0
def val(x):
    v = x
    for _ in range(4):
        if hasattr(v, "iloc"): v = v.iloc[0]
        elif isinstance(v, (list, tuple, np.ndarray)): v = np.asarray(v).ravel()[0]
        else: break
    return float(v)
cs0 = constructOriginalCS(beta, V, nP, nQ, l_vec=l)
print(f"   point estimate June 2020: {float(l @ beta[nP:]):+.3f}")
print(f"   parallel trends: [{val(cs0['lb']):+.3f}, {val(cs0['ub']):+.3f}]")
H = {"pt": [round(val(cs0["lb"]), 3), round(val(cs0["ub"]), 3)]}
for M in [0.5, 1.0, 1.5, 2.0, 3.0]:
    try:
        r = createSensitivityResults_relativeMagnitudes(beta, V, nP, nQ, Mbarvec=[M],
                                                        l_vec=l, gridPoints=200)
        lb, ub = val(r["lb"]), val(r["ub"])
        inc = lb <= 0 <= ub
        print(f"   Mbar={M:<4} [{lb:+.3f}, {ub:+.3f}]{'  <- includes 0' if inc else ''}")
        H[f"M{M}"] = [round(lb, 3), round(ub, 3), bool(inc)]
    except Exception as e:
        print(f"   Mbar={M:<4} failed {type(e).__name__}")
res["honest_poolFE"] = H

json.dump(res, open(os.path.join(OUT, "results_pretrend2.json"), "w"), indent=1, default=str)
print("\nwrote results_pretrend2.json")
