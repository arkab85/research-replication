"""Reproduce the paper's tables and run the two tests it was missing."""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

res = {}

p    = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
full = pd.read_parquet(os.path.join(OUT, "full.parquet"))

CTRL = ["coupon", "fico", "cltv", "age"]
p = p.dropna(subset=CTRL + ["buyout"]).copy()
p["ext"]  = p.itype_ext
p["base"] = p.itype_base

def tab(msg):
    print("\n" + "=" * 78); print(msg); print("=" * 78)

# ---------------------------------------------------------------- Table 1
tab("TABLE 1  Buyout rate by issuer type and year (full history)")
f = full.dropna(subset=["itype"]).copy()
t1 = (f[f.itype.isin(["depository", "nonbank", "techfirst"])]
      .pivot_table(index="year", columns="itype", values="buyout",
                   aggfunc=["mean", "size"]))
print((t1["mean"] * 100).round(1)); print(t1["size"])
res["T1"] = {"mean": (t1["mean"] * 100).round(1).to_dict(), "n": t1["size"].to_dict()}

# ---------------------------------------------------------------- Table 2
tab("TABLE 2  Buyout rate by month")
t2 = (f[f.ym.between(201901, 202009) & f.itype.isin(["depository", "nonbank"])]
      .pivot_table(index="ym", columns="itype", values="buyout", aggfunc=["mean", "size"]))
print((t2["mean"] * 100).round(1).to_string())
res["T2"] = (t2["mean"] * 100).round(1).to_dict()
(t2["mean"] * 100).round(2).to_csv(os.path.join(OUT, "t2_monthly.csv"))

# ---------------------------------------------------------------- Table 3
tab("TABLE 3  Buyout rate by note rate bin and issuer type")
BINS   = [0, 3.5, 4.0, 4.5, 5.0, 5.5, 99]
LABELS = ["<3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0", "5.0-5.5", "5.5+"]
p["cbin"] = pd.cut(p.coupon, BINS, labels=LABELS, right=False)
for nm, sel in [("2019", p.ym.between(201901, 201912)),
                ("Mar-Sep 2020", p.ym.between(202003, 202009))]:
    s = p[sel & p.ext.isin(["depository", "nonbank"])]
    t = s.pivot_table(index="cbin", columns="ext", values="buyout",
                      aggfunc=["mean", "size"], observed=True)
    out = pd.DataFrame({"dep": (t["mean"]["depository"] * 100).round(1),
                        "n_dep": t["size"]["depository"],
                        "nb":  (t["mean"]["nonbank"] * 100).round(1),
                        "n_nb": t["size"]["nonbank"]})
    out["gap"] = (out.dep - out.nb).round(1)
    print(f"\n-- {nm} --"); print(out.to_string())
    res[f"T3_{nm}"] = out.to_dict()
    out.to_csv(os.path.join(OUT, f"t3_{nm.replace(' ','_').replace('-','_')}.csv"))

# ---------------------------------------------------------------- Table 4
tab("TABLE 4  Difference-in-differences")
def did(d, label, extra=None, trend=False):
    d = d.copy()
    d["nb_post"] = d.nonbank * d.post
    rhs = ["nb_post"] + CTRL
    if trend:
        d["t"] = (pd.to_datetime(d.ym.astype(str), format="%Y%m")
                  - pd.Timestamp("2019-01-01")).dt.days / 30.44
        d["nb_t"] = d.nonbank * d["t"]
        rhs.append("nb_t")
    fml = f"buyout ~ {' + '.join(rhs)} | issuer_id + state_month"
    m = pf.feols(fml, data=d, vcov={"CRV1": "issuer_id"})
    c, se = m.coef()["nb_post"], m.se()["nb_post"]
    print(f"{label:<46} {c:>8.3f} ({se:.3f})  N={len(d):>9,}  G={d.issuer_id.nunique()}")
    return {"coef": round(float(c), 4), "se": round(float(se), 4),
            "n": int(len(d)), "g": int(d.issuer_id.nunique()),
            "p": float(m.pvalue()["nb_post"])}

E  = p[p.ext.isin(["depository", "nonbank", "techfirst", "hfa"])]
E3 = p[p.ext.isin(["depository", "nonbank", "techfirst"])]
T4 = {}
T4["main"]      = did(E,  "Extended classification, full controls")
T4["no_hfa"]    = did(E3, "  excluding housing finance agencies")
T4["no_spike"]  = did(E3[~E3.ym.isin([202006, 202007])], "  excluding June-July 2020 volume spike")
T4["narrow"]    = did(E3[E3.ym.isin([201911, 201912, 202001, 202002, 202003, 202004])],
                      "  Nov 2019-Jan 2020 vs. Feb-Apr 2020")
big = E3.issuer_id.value_counts().index[0]
T4["no_largest"] = did(E3[E3.issuer_id != big], "  excluding the largest issuer")
T4["nb_dep"]     = did(E3[E3.ext.isin(["depository", "nonbank"])], "  nonbank vs. depository only")
T4["trend"]      = did(E, "  allowing a nonbank-specific linear trend", trend=True)
res["T4"] = T4

# placebo: Jan 2018 - Sep 2019, shock dated March 2019
fp = full.dropna(subset=["itype"]).copy()
fp = fp[fp.ym.between(201801, 201909) & fp.itype.isin(["depository", "nonbank", "techfirst", "hfa"])]
fp = fp.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                        "ltv_current": "cltv", "loan_age": "age"}).dropna(subset=CTRL)
fp["nonbank"] = (fp.itype == "nonbank").astype(float)
fp["post"]    = (fp.ym >= 201903).astype(float)
fp["state_month"] = fp.state.astype(str) + "_" + fp.ym.astype(str)
T4["placebo_2019"] = did(fp, "Placebo: shock dated March 2019")
res["T4"] = T4

# ---------------------------------------------------------------- Table 5
tab("TABLE 5  Event study (base classification), base month Feb 2020")
B = p[p.base.notna()].copy()
B["nonbank"] = (B.base == "nonbank").astype(float)
months = sorted(B.ym.unique())
for m_ in months:
    if m_ != 202002:
        B[f"d{m_}"] = B.nonbank * (B.ym == m_)
dcols = [f"d{m_}" for m_ in months if m_ != 202002]
es = pf.feols(f"buyout ~ {' + '.join(dcols)} | issuer_id + state_month",
              data=B, vcov={"CRV1": "issuer_id"})
ev = pd.DataFrame({"coef": es.coef()[dcols].values, "se": es.se()[dcols].values},
                  index=[int(c[1:]) for c in dcols])
print(ev.round(3).to_string())
ev.to_csv(os.path.join(OUT, "event_study.csv"))
# full covariance matrix of the event-study coefficients -> for HonestDiD
V = es._vcov
idx = [list(es._coefnames).index(c) for c in dcols]
np.save(os.path.join(OUT, "event_vcov.npy"), np.asarray(V)[np.ix_(idx, idx)])
res["T5"] = ev.round(4).to_dict()
print(f"\nevent-study N={len(B):,}  issuers={B.issuer_id.nunique()}")

# ================================================================
#                    NEW TEST 1: TRIPLE DIFFERENCE
# ================================================================
tab("NEW TABLE 9  Triple difference: does the coupon slope collapse for nonbanks?  [eq. 4]")
def tdd(d, label, sample_note=""):
    d = d.copy()
    d["nb_post"] = d.nonbank * d.post
    d["c"]       = d.coupon
    d["c_post"]  = d.coupon * d.post
    d["c_nb"]    = d.coupon * d.nonbank
    d["c_nb_post"] = d.coupon * d.nonbank * d.post
    rhs = ["nb_post", "c", "c_post", "c_nb", "c_nb_post", "fico", "cltv", "age"]
    m = pf.feols(f"buyout ~ {' + '.join(rhs)} | issuer_id + state_month",
                 data=d, vcov={"CRV1": "issuer_id"})
    co, se, pv = m.coef(), m.se(), m.pvalue()
    print(f"\n-- {label} -- {sample_note}")
    for k in ["nb_post", "c", "c_post", "c_nb", "c_nb_post"]:
        star = "***" if pv[k] < .01 else "**" if pv[k] < .05 else "*" if pv[k] < .10 else ""
        print(f"   {k:<12} {co[k]:>9.4f} ({se[k]:.4f}) {star}")
    b1, b2, b3, b4 = co["c"], co["c_post"], co["c_nb"], co["c_nb_post"]
    print(f"   depository slope pre  b1        = {b1:8.4f}")
    print(f"   depository slope post b1+b2     = {b1+b2:8.4f}")
    print(f"   nonbank    slope pre  b1+b3     = {b1+b3:8.4f}")
    print(f"   nonbank    slope post b1+b2+b3+b4 = {b1+b2+b3+b4:8.4f}")
    return {k: [round(float(co[k]), 4), round(float(se[k]), 4), float(pv[k])]
            for k in ["nb_post", "c", "c_post", "c_nb", "c_nb_post"]} | {
            "slope_dep_pre": round(float(b1), 4), "slope_dep_post": round(float(b1+b2), 4),
            "slope_nb_pre": round(float(b1+b3), 4), "slope_nb_post": round(float(b1+b2+b3+b4), 4),
            "n": int(len(d)), "g": int(d.issuer_id.nunique())}

T9 = {}
T9["main"]   = tdd(E3, "Extended classification", f"N={len(E3):,}")
T9["nb_dep"] = tdd(E3[E3.ext.isin(["depository", "nonbank"])], "Nonbank vs depository only")
T9["base"]   = tdd(B, "Base classification")
T9["fha"]    = tdd(E3[E3.agency == "F"], "FHA loans only")
res["T9"] = T9

with open(os.path.join(OUT, "results_main.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print("\nwrote results_main.json")
