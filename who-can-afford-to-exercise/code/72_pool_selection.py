"""Is the within-pool sample selected?

An issuer chooses which loans to place in a multi-issuer pool and which to keep in a
custom pool. That choice is made at securitisation, years before 2020, so it cannot
respond to the shock -- but if the loans nonbanks contribute to multi-issuer pools differ
systematically, the within-pool comparison could still mislead. Five checks:

 1. how much of each type's collateral sits in multi-issuer pools
 2. observables of an issuer's own loans inside vs outside multi-issuer pools
 3. the DiD estimated separately on multi-issuer and custom pools
 4. a triple interaction: does the 2020 decline differ by pool type, within issuer?
 5. did the mix of vesting options across pool types shift in 2020?
"""
import pandas as pd, numpy as np, os, json, warnings
import pyfixest as pf
from config import OUT, EXTRACT as DATA, ISSUER_TYPES
warnings.filterwarnings("ignore")

CTRL = ["coupon", "fico", "cltv", "age"]
res = {}
def line(m): print("\n" + "=" * 88); print(m); print("=" * 88, flush=True)

USE = ["as_of_date", "pool_id", "issuer_id", "bEBO", "upb", "interest_rate",
       "credit_score", "ltv_current", "loan_age", "state", "agency", "issue_type"]
d = pd.read_csv(DATA, usecols=USE, low_memory=False)
d["ym"] = d.as_of_date.astype(int)
d = d[d.ym.between(201901, 202009)].copy()
d = d.rename(columns={"interest_rate": "coupon", "credit_score": "fico",
                      "ltv_current": "cltv", "loan_age": "age"})
for c in CTRL: d[c] = pd.to_numeric(d[c], errors="coerce")
d["buyout"] = d.bEBO.astype(str).str.upper().eq("TRUE").astype(float)
base = pd.read_csv(ISSUER_TYPES)
BM = {"traditional": "depository", "shadow": "nonbank", "fintech": "techfirst"}
tmap = {k: BM.get(v) for k, v in zip(base.IssuerID, base.bank_type)}
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
tmap = {**tmap, **iss.ext.to_dict()}
d["ext"] = d.issuer_id.map(tmap)
d = d[d.ext.isin(["depository", "nonbank"])].dropna(subset=CTRL + ["buyout"]).copy()
d["nb"] = (d.ext == "nonbank").astype(float)
d["post"] = (d.ym >= 202003).astype(float)
d["nb_post"] = d.nb * d.post
d["multi"] = (d.issue_type == "M").astype(float)
d["state_month"] = d.state.astype(str) + "_" + d.ym.astype(str)

line("1. HOW MUCH OF EACH TYPE SITS IN MULTI-ISSUER POOLS?")
t = d.groupby("ext").multi.mean() * 100
print((d.groupby(["ext", "issue_type"]).size().unstack(fill_value=0)).to_string())
print("\n   share of decisions in multi-issuer (type M) pools:")
for k, v in t.items(): print(f"     {k:<11} {v:.1f}%")
res["multi_share"] = {k: round(float(v), 1) for k, v in t.items()}

line("2. AN ISSUER'S OWN LOANS, INSIDE vs OUTSIDE MULTI-ISSUER POOLS")
both = d.groupby("issuer_id").multi.nunique()
ids = both[both > 1].index
q = d[d.issuer_id.isin(ids)]
print(f"   {len(ids)} issuers place loans in both pool types ({len(q):,} decisions)")
comp = q.groupby(["ext", "multi"])[CTRL + ["upb"]].mean().round(2)
comp.index = [f"{a}, {'multi' if b else 'custom'}" for a, b in comp.index]
print(comp.to_string())
# within-issuer difference
for c in CTRL:
    q2 = q.copy()
    m = pf.feols(f"{c} ~ multi | issuer_id", data=q2, vcov={"CRV1": "issuer_id"})
    print(f"   within-issuer difference in {c:<8}: {m.coef()['multi']:+.3f} "
          f"({m.se()['multi']:.3f})")
    res[f"diff_{c}"] = [round(float(m.coef()["multi"]), 3), round(float(m.se()["multi"]), 3)]

line("3. THE DiD, SEPARATELY BY POOL TYPE")
for nm, x in [("multi-issuer pools", d[d.multi == 1]),
              ("custom / single-issuer pools", d[d.multi == 0])]:
    m = pf.feols(f"buyout ~ nb_post + {' + '.join(CTRL)} | issuer_id + state_month",
                 data=x, vcov={"CRV1": "issuer_id"})
    c, se, p = m.coef()["nb_post"], m.se()["nb_post"], m.pvalue()["nb_post"]
    st = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
    print(f"   {nm:<32} {c:>8.4f} ({se:.4f}) {st:<3} N={int(m._N):>9,} "
          f"G={x.issuer_id.nunique()}")
    res[nm] = {"coef": round(float(c), 4), "se": round(float(se), 4), "p": float(p)}

line("4. DOES THE DECLINE DIFFER BY POOL TYPE, WITHIN ISSUER?")
d["nb_post_multi"] = d.nb_post * d.multi
d["post_multi"] = d.post * d.multi
d["nb_multi"] = d.nb * d.multi
m = pf.feols(f"buyout ~ nb_post + nb_post_multi + post_multi + nb_multi + "
             f"{' + '.join(CTRL)} | issuer_id + state_month",
             data=d, vcov={"CRV1": "issuer_id"})
for k in ["nb_post", "nb_post_multi"]:
    pv = m.pvalue()[k]
    st = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .1 else ""
    print(f"   {k:<15} {m.coef()[k]:>8.4f} ({m.se()[k]:.4f}) p={pv:.3f} {st}")
print(f"   custom-pool effect {m.coef()['nb_post']:+.4f};  multi-issuer "
      f"{m.coef()['nb_post'] + m.coef()['nb_post_multi']:+.4f}")
print("   -> if the interaction is small, the within-pool sample is not special")
res["interaction"] = {k: [round(float(m.coef()[k]), 4), round(float(m.se()[k]), 4),
                          float(m.pvalue()[k])] for k in ["nb_post", "nb_post_multi"]}

line("5. DID THE POOL-TYPE MIX SHIFT IN 2020?")
mix = d.groupby(["ext", "ym"]).multi.mean().unstack(0) * 100
print("   share of vesting options in multi-issuer pools, by month:")
print(mix.loc[[m for m in [201901, 201906, 201912, 202002, 202003, 202006, 202009]
               if m in mix.index]].round(1).to_string())
sh = d.groupby(["ext", "post"]).multi.mean().unstack() * 100
print("\n   pre vs post:")
print(sh.round(1).to_string())
res["mix_shift"] = sh.round(1).to_dict()

json.dump(res, open(os.path.join(OUT, "results_poolselect.json"), "w"),
          indent=1, default=str)
print("\nwrote results_poolselect.json")
