"""Forbearance as a known future drain.

A loan in CARES forbearance is one the issuer knows will not pay for months. The advance
obligation attached to it is therefore large and predictable, which makes buying it out
MORE attractive to an issuer that can pay for it and no more feasible for one that cannot.

  Capacity channel:  forbearance raises the incentive to exercise; unconstrained issuers
                     act on it, constrained ones cannot. The forbearance gradient in
                     exercise should widen by issuer type after the shock.
  Value channel:     forbearance is a property of the loan, not its owner. It may shift
                     the cure probability, but it does so identically for both types.

Also: does immobilisation cost more where foreclosure is slow (judicial states)?
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT, EXTRACT as DATA, ISSUER_TYPES
warnings.filterwarnings("ignore")

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 88); print(m); print("=" * 88, flush=True)

USE = ["as_of_date", "pool_id", "issuer_id", "bEBO", "upb", "interest_rate",
       "credit_score", "ltv_current", "loan_age", "state", "agency", "fb_flag",
       "num_mth_fb", "covid_flag", "Judicial", "removal_in_n_mth_code"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d = d[d.ym.between(201901, 202009)].copy()
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL: d[c] = pd.to_numeric(d[c], errors="coerce")
d["buyout"] = d.bEBO.astype(str).str.upper().eq("TRUE").astype(float)
d["fb"] = d.fb_flag.astype(str).str.upper().eq("Y").astype(float)
d["frozen"] = (d.removal_in_n_mth_code == 0).astype(float)
base = pd.read_csv(ISSUER_TYPES)
BM = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
tmap = {k: BM.get(v) for k, v in zip(base.IssuerID, base.bank_type)}
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
tmap = {**tmap, **iss.ext.to_dict()}
d["ext"] = d.issuer_id.map(tmap)
d = d[d.ext.isin(["depository", "nonbank"])].dropna(subset=CTRL + ["buyout"]).copy()
d["nb"] = (d.ext == "nonbank").astype(float)
d["post"] = (d.ym >= 202003).astype(float)
d["pm"] = d.pool_id.astype(str) + "_" + d.ym.astype(str)
k = d.groupby("pm").nb.transform(lambda s: s.nunique())
P = d[k > 1].copy()
print(f"within-pool sample: {len(P):,} decisions, {P.pm.nunique():,} cells")

line("1. RAW: EXERCISE BY FORBEARANCE STATUS, TYPE AND PERIOD")
t = (P[P.post == 1].pivot_table(index="ext", columns="fb", values="buyout") * 100)
t.columns = ["not in forbearance", "in forbearance"]
t["gradient"] = (t.iloc[:, 1] - t.iloc[:, 0]).round(1)
print("   Mar-Sep 2020:"); print(t.round(1).to_string())
t0 = (P[(P.post == 0) & (P.ym >= 201907)].pivot_table(index="ext", columns="fb",
                                                      values="buyout") * 100)
if t0.shape[1] > 1:
    t0.columns = ["not in forbearance", "in forbearance"]
    print("\n   Jul 2019-Feb 2020:"); print(t0.round(1).to_string())
else:
    print("\n   (forbearance is essentially absent before March 2020)")
print(f"\n   forbearance incidence: pre {P[P.post==0].fb.mean()*100:.1f}%  "
      f"post {P[P.post==1].fb.mean()*100:.1f}%")

line("2. THE FORBEARANCE GRADIENT, WITHIN POOL x MONTH, POST PERIOD ONLY")
q = P[P.post == 1].copy()
q["fb_nb"] = q.fb * q.nb
m = pf.feols(f"buyout ~ fb + fb_nb + {' + '.join(CTRL)} | issuer_id + pm",
             data=q, vcov={"CRV1": "issuer_id"})
for kx in ["fb", "fb_nb"]:
    pv = m.pvalue()[kx]
    st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    print(f"   {kx:<7} {m.coef()[kx]:>8.4f} ({m.se()[kx]:.4f}) p={pv:.4f} {st}")
print(f"   depository gradient        = {m.coef()['fb']:+.4f}")
print(f"   nonbank gradient           = {m.coef()['fb'] + m.coef()['fb_nb']:+.4f}")
print(f"   N={int(m._N):,}  issuers={q.issuer_id.nunique()}")
res["gradient_post"] = {kx: [round(float(m.coef()[kx]), 4), round(float(m.se()[kx]), 4),
                             float(m.pvalue()[kx])] for kx in ["fb", "fb_nb"]}

line("3. DOSE: MONTHS ALREADY IN FORBEARANCE")
q["nmfb"] = pd.to_numeric(q.num_mth_fb, errors="coerce").fillna(0)
q["nmfb_nb"] = q.nmfb * q.nb
m2 = pf.feols(f"buyout ~ nmfb + nmfb_nb + {' + '.join(CTRL)} | issuer_id + pm",
              data=q, vcov={"CRV1": "issuer_id"})
for kx in ["nmfb", "nmfb_nb"]:
    pv = m2.pvalue()[kx]
    st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    print(f"   {kx:<9} {m2.coef()[kx]:>8.4f} ({m2.se()[kx]:.4f}) p={pv:.4f} {st}")
print("   (per additional month already spent in forbearance)")
res["dose"] = {kx: [round(float(m2.coef()[kx]), 4), round(float(m2.se()[kx]), 4),
                    float(m2.pvalue()[kx])] for kx in ["nmfb", "nmfb_nb"]}

line("4. THE SAME ON THE FREEZE MARGIN")
m3 = pf.feols(f"frozen ~ fb + fb_nb + {' + '.join(CTRL)} | issuer_id + pm",
              data=q, vcov={"CRV1": "issuer_id"})
for kx in ["fb", "fb_nb"]:
    print(f"   {kx:<7} {m3.coef()[kx]:>8.4f} ({m3.se()[kx]:.4f}) p={m3.pvalue()[kx]:.4f}")
res["freeze_gradient"] = {kx: [round(float(m3.coef()[kx]), 4),
                               round(float(m3.se()[kx]), 4), float(m3.pvalue()[kx])]
                          for kx in ["fb", "fb_nb"]}

line("5. JUDICIAL STATES: does immobilisation bind harder where exit is slow?")
P["jud"] = pd.to_numeric(P.Judicial, errors="coerce").fillna(0)
P["nb_post"] = P.nb * P.post
P["nb_post_jud"] = P.nb_post * P.jud
P["jud_post"] = P.jud * P.post
F = P[P.ym >= 201907].copy()
m4 = pf.feols(f"buyout ~ nb_post + nb_post_jud + jud_post + {' + '.join(CTRL)}"
              f" | issuer_id + pm", data=F, vcov={"CRV1": "issuer_id"})
for kx in ["nb_post", "nb_post_jud", "jud_post"]:
    pv = m4.pvalue()[kx]
    st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    print(f"   {kx:<13} {m4.coef()[kx]:>8.4f} ({m4.se()[kx]:.4f}) p={pv:.3f} {st}")
print(f"   non-judicial effect {m4.coef()['nb_post']:+.4f}; judicial "
      f"{m4.coef()['nb_post'] + m4.coef()['nb_post_jud']:+.4f}")
res["judicial"] = {kx: [round(float(m4.coef()[kx]), 4), round(float(m4.se()[kx]), 4),
                        float(m4.pvalue()[kx])] for kx in ["nb_post", "nb_post_jud"]}

json.dump(res, open(os.path.join(OUT, "results_forbearance.json"), "w"),
          indent=1, default=str)
print("\nwrote results_forbearance.json")
