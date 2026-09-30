"""External validation of the paper's scale proxy against Ginnie Mae's own classification.

Both reviewer reports objected that issuer size is a proxy measured with error. Ginnie Mae
assigns every issuer to a peer group -- Mega, Large, Medium, Small, Very Small, crossed
with depository and non-depository -- and publishes the membership with Issuer IDs, free
and without a login. That is the agency's own size classification of the same firms, built
from servicing portfolio rather than from delinquency flow, so agreement between the two is
evidence the proxy measures the intended construct and not something incidental to it.

The paper's test is a contrast of gradients WITHIN each type, so the within-type rank
correlation is the one that matters.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import OUT, PAPER as PAP

pg = pd.read_csv(os.path.join(OUT, "iopp_peer_groups.csv"))
il = pd.read_csv(os.path.join(OUT, "issuer_level.csv"))
j = il.merge(pg, on="issuer_id", how="inner")

RANK = [("mega", 5), ("large", 4), ("medium", 3), ("very small", 1), ("small", 2)]


def rk(g):
    gl = str(g).lower()
    for k, v in RANK:
        if k in gl:
            return v
    return np.nan


j["gm_rank"] = j.peer_group.apply(rk)
j = j.dropna(subset=["gm_rank", "S"])
print(f"issuers matched to a Ginnie Mae peer group: {len(j)}")

out = {}
rho, p = spearmanr(j.S, j.gm_rank)
out["all"] = [round(float(rho), 3), float(p), int(len(j))]
print(f"\n  overall   rho = {rho:+.3f}  p = {p:.5f}  n = {len(j)}")
for t in ["depository", "nonbank"]:
    k = j[j.ext == t]
    if len(k) > 4:
        r, pv = spearmanr(k.S, k.gm_rank)
        out[t] = [round(float(r), 3), float(pv), int(len(k))]
        print(f"  {t:<12} rho = {r:+.3f}  p = {pv:.5f}  n = {len(k)}")

act = j[(j.ebo19 >= 0.10) & (j.n19 >= 500)]
if len(act) > 4:
    r, pv = spearmanr(act.S, act.gm_rank)
    out["active"] = [round(float(r), 3), float(pv), int(len(act))]
    print(f"  active issuers (the gradient sample) rho = {r:+.3f}  p = {pv:.5f}  n = {len(act)}")

print("\n  mean of the paper's scale measure by Ginnie Mae peer group:")
tab = j.groupby("peer_group").S.agg(["count", "mean"]).round(2).sort_values("mean")
print(tab.to_string())
mono = tab["mean"].is_monotonic_increasing
print(f"  monotone in the agency's own ordering: {mono}")

j.to_csv(os.path.join(OUT, "size_validation.csv"), index=False)
json.dump(out, open(os.path.join(OUT, "results_size_validation.json"), "w"), indent=1)


def num(x, k=2):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


mac = {"gmRho": num(out["all"][0]), "gmN": str(out["all"][2]),
       "gmRhoDep": num(out.get("depository", [np.nan])[0]),
       "gmNDep": str(out.get("depository", [0, 0, 0])[2]),
       "gmRhoNb": num(out.get("nonbank", [np.nan])[0]),
       "gmNNb": str(out.get("nonbank", [0, 0, 0])[2]),
       "gmGroups": str(j.peer_group.nunique())}
import re
pth = os.path.join(PAP, "numbers_pool.tex")
txt = open(pth, encoding="utf-8").read()
new = []
for k, v in sorted(mac.items()):
    pat = re.compile(r"(\\newcommand\{\\p" + k + r"\}\{)[^}]*(\})")
    if pat.search(txt):
        txt = pat.sub(lambda m: m.group(1) + v + m.group(2), txt)
    else:
        new.append("\\newcommand{\\p" + k + "}{" + v + "}")
if new:
    txt = txt.rstrip() + "\n" + "\n".join(new) + "\n"
open(pth, "w", encoding="utf-8").write(txt)
print(f"\nwrote {len(mac)} macros ({len(new)} new)")
