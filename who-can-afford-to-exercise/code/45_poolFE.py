"""Within-pool identification.

Two loans in the same Ginnie Mae multi-issuer pool in the same month are backed by the
same programme, sit behind the same security, and are observed by the investor as one
piece of collateral. They differ in who services them. Pool-by-month fixed effects
absorb the pool, the coupon, the vintage, the geography mix and every pool-level shock,
so the coefficient on Nonbank x Post is identified from issuers operating inside the
same pool in the same month.

If the pre-trend is a composition artefact -- nonbanks and depositories holding different
books that drifted apart -- it should shrink sharply here.
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT
warnings.filterwarnings("ignore")

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
d["nb_post"] = d.nonbank * d.post
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)
d["state_month"] = d.state.astype(str) + "_" + d.ym.astype(str)
# keep cells that still contain both types after dropping missing controls
k = d.groupby("pm").nonbank.transform(lambda s: s.nunique())
d = d[k > 1].copy()
print(f"estimation sample: {len(d):,} decisions, {d.pm.nunique():,} pool-month cells, "
      f"{d.issuer_id.nunique()} issuers, {d.pool_id.nunique():,} pools")
res["n"] = int(len(d)); res["cells"] = int(d.pm.nunique())
res["issuers"] = int(d.issuer_id.nunique()); res["pools"] = int(d.pool_id.nunique())

line("1. MAIN EFFECT, PROGRESSIVELY TIGHTER FIXED EFFECTS")
SPECS = [
    ("issuer + state x month",        "issuer_id + state_month"),
    ("issuer + month + pool",         "issuer_id + ym + pool_id"),
    ("issuer + POOL x MONTH",         "issuer_id + pm"),
    ("issuer + POOL x MONTH + state", "issuer_id + pm + state"),
]
T = {}
for lab, fe in SPECS:
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | {fe}",
                 data=d, vcov={"CRV1": "issuer_id"})
    c, se, p = m.coef()["nb_post"], m.se()["nb_post"], m.pvalue()["nb_post"]
    st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    print(f"   {lab:<32} {c:>8.4f} ({se:.4f}) {st:<3} N={int(m._N):>9,}")
    T[lab] = {"coef": round(float(c), 4), "se": round(float(se), 4), "p": float(p),
              "n": int(m._N)}
res["main"] = T

line("2. EVENT STUDY WITH POOL x MONTH FIXED EFFECTS")
months = sorted(d.ym.unique())
for m_ in months:
    if m_ != 202002: d[f"d{m_}"] = d.nonbank * (d.ym == m_)
dc = [f"d{m_}" for m_ in months if m_ != 202002]
es = pf.feols(f"buyout ~ {' + '.join(dc)} + {' + '.join(CTRL)} | issuer_id + pm",
              data=d, vcov={"CRV1": "issuer_id"})
ev = pd.DataFrame({"coef": es.coef()[dc].values, "se": es.se()[dc].values},
                  index=[int(c[1:]) for c in dc])
print(ev.round(3).to_string())
ev.to_csv(os.path.join(OUT, "event_study_poolFE.csv"))
pre, post = ev[ev.index <= 202001], ev[ev.index >= 202003]
print(f"\n   largest pre-period coefficient : {pre.coef.max():+.3f} "
      f"({pre.coef.abs().max():.3f} in absolute value)")
print(f"   largest post-period coefficient: {post.coef.min():+.3f}")
print(f"   ratio |post| / |pre|           : {abs(post.coef.min())/pre.coef.abs().max():.1f}x")
res["event"] = {"max_pre": round(float(pre.coef.max()), 4),
                "max_abs_pre": round(float(pre.coef.abs().max()), 4),
                "min_post": round(float(post.coef.min()), 4)}

line("3. PLACEBO: shock dated March 2019, within pool x month")
pl = d[d.ym <= 201912].copy()
pl["post19"] = (pl.ym >= 201903).astype(float)
pl["nb_post19"] = pl.nonbank * pl.post19
kk = pl.groupby("pm").nonbank.transform(lambda s: s.nunique())
pl = pl[kk > 1]
mp = pf.feols(f"buyout ~ nb_post19 + {' + '.join(CTRL)} | issuer_id + pm",
              data=pl, vcov={"CRV1": "issuer_id"})
c, se, p = mp.coef()["nb_post19"], mp.se()["nb_post19"], mp.pvalue()["nb_post19"]
print(f"   placebo Nonbank x Post(Mar 2019) = {c:+.4f} ({se:.4f}) p={p:.3f}  N={int(mp._N):,}")
res["placebo"] = {"coef": round(float(c), 4), "se": round(float(se), 4), "p": float(p)}

line("4. THE SIZE GRADIENT, WITHIN POOL x MONTH")
try:
    iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
    q = d.merge(iss[["S", "ebo19"]], left_on="issuer_id", right_index=True, how="inner")
    q = q[q.ebo19 >= .10].copy()
    kk = q.groupby("pm").nonbank.transform(lambda s: s.nunique())
    q = q[kk > 1].copy()
    q["S_z"] = (q.S - q.S.mean()) / q.S.std()
    q["S_post"] = q.S_z * q.post; q["S_nb_post"] = q.S_z * q.nonbank * q.post
    ms = pf.feols(f"buyout ~ S_post + nb_post + S_nb_post + {' + '.join(CTRL)}"
                  f" | issuer_id + pm", data=q, vcov={"CRV1": "issuer_id"})
    print(f"   N={int(ms._N):,}  issuers={q.issuer_id.nunique()}  cells={q.pm.nunique():,}")
    for kx in ["S_post", "nb_post", "S_nb_post"]:
        pv = ms.pvalue()[kx]
        st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
        print(f"     {kx:<11} {ms.coef()[kx]:>8.4f} ({ms.se()[kx]:.4f}) p={pv:.3f} {st}")
    res["size_gradient"] = {kx: [round(float(ms.coef()[kx]), 4),
                                 round(float(ms.se()[kx]), 4), float(ms.pvalue()[kx])]
                            for kx in ["S_post", "nb_post", "S_nb_post"]}
except Exception as e:
    print("   size gradient failed:", type(e).__name__, e)

json.dump(res, open(os.path.join(OUT, "results_poolFE.json"), "w"), indent=1, default=str)
print("\nwrote results_poolFE.json")
