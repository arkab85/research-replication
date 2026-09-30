"""Match GMAR MSR holders to sample issuers by distinctive token, and validate the proxy.

The first attempt matched only eight issuers, almost all depositories, because a regex on
the legal name failed on exactly the large non-depositories the test is about. This matches
on a distinctive token instead and reports which holders remain unmatched, so the coverage
is visible rather than assumed.
"""
import os, json, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from config import OUT, PAPER as PAP

g = pd.read_csv(os.path.join(OUT, "gmar_msr_holders.csv"))
il = pd.read_csv(os.path.join(OUT, "issuer_level.csv"))
il["U"] = il.name.str.upper()

# distinctive token -> how it appears in the Ginnie Mae issuer name
TOK = {
    "LAKEVIEW": "LAKEVIEW", "PENNYMAC": "PENNYMAC", "WELLS FARGO": "WELLS FARGO",
    "FREEDOM": "FREEDOM", "NATIONSTAR": "NATIONSTAR", "QUICKEN": "QUICKEN",
    "US BANK": "U. S. BANK", "U.S. BANK": "U. S. BANK", "NEWREZ": "NEWREZ",
    "CALIBER": "CALIBER", "AMERIHOME": "AMERIHOME", "CARRINGTON": "CARRINGTON",
    "PLANET": "PLANET", "GUILD": "GUILD", "LOANDEPOT": "LOANDEPOT",
    "LOAN DEPOT": "LOANDEPOT", "MIDFIRST": "MIDFIRST", "TRUIST": "TRUIST",
    "PNC": "PNC BANK", "M&T": "M&T", "FIFTH THIRD": "FIFTH THIRD",
    "JPMORGAN": "JP MORGAN", "CHASE": "JP MORGAN", "FLAGSTAR": "FLAGSTAR",
    "MONEY SOURCE": "MONEY SOURCE", "HOME POINT": "HOME POINT",
    "CITIZENS": "CITIZENS", "CROSSCOUNTRY": "CROSSCOUNTRY",
    "VILLAGE CAPITAL": "VILLAGE CAPITAL", "AMERICAN FINANCIAL": "AMERICAN FINANCIAL",
    "MR. COOPER": "NATIONSTAR", "MR COOPER": "NATIONSTAR", "OCWEN": "PHH",
    "PHH": "PHH", "MATRIX": "MATRIX", "IDAHO": "IDAHO", "COLONIAL": "COLONIAL",
    "ARVEST": "ARVEST", "CMG": "CMG", "PRIMELENDING": "PRIMELENDING",
    "PROVIDENT": "PROVIDENT", "UNION HOME": "UNION HOME", "USAA": "USAA",
    "CENLAR": "CENLAR", "NAVY FEDERAL": "NAVY FEDERAL",
}

snap = g[g.file == "sep20"].copy()          # data as of July 2020
if snap.empty:
    snap = g[g.file == g.file.iloc[0]].copy()


def find(h):
    hu = h.upper()
    for tok, pat in TOK.items():
        if tok in hu:
            hit = il[il.U.str.contains(re.escape(pat), na=False)]
            if len(hit):
                return hit.sort_values("n19", ascending=False).iloc[0]
    return None


rows, miss = [], []
for _, r in snap.iterrows():
    hit = find(r.holder)
    if hit is None:
        miss.append((r["rank"], r.holder, r.upb_mn))
        continue
    rows.append({**hit.to_dict(), "holder": r.holder, "upb_mn": r.upb_mn,
                 "rank": r["rank"]})
m = pd.DataFrame(rows).drop_duplicates("issuer_id")
print(f"matched {len(m)} of {len(snap)} top-30 holders to sample issuers")
print(f"unmatched ({len(miss)}): " + ", ".join(f"#{a} {b}" for a, b, _ in miss[:12]))

m = m.dropna(subset=["S"])
m["log_upb"] = np.log(m.upb_mn)
rho, p1 = spearmanr(m.S, m.log_upb)
r, p2 = pearsonr(m.S, m.log_upb)
print(f"\n  n = {len(m)}   depositories {int((m.ext=='depository').sum())}, "
      f"nonbanks {int((m.ext=='nonbank').sum())}")
print(f"  Spearman(scale proxy, log Ginnie servicing UPB) = {rho:+.3f}  p = {p1:.5f}")
print(f"  Pearson                                         = {r:+.3f}  p = {p2:.5f}")
out = {"spearman": round(float(rho), 3), "pearson": round(float(r), 3), "n": int(len(m))}
for t in ["depository", "nonbank"]:
    k = m[m.ext == t]
    if len(k) > 4:
        rr, pp = spearmanr(k.S, k.log_upb)
        print(f"    within {t:<12} rho = {rr:+.3f}  p = {pp:.4f}  n = {len(k)}")
        out[t] = [round(float(rr), 3), float(pp), int(len(k))]

m["intensity"] = m.n19 / (m.upb_mn / 1000)
print("\n  2019 options vesting per $bn of Ginnie servicing book:")
print(m.sort_values("intensity")[["name", "ext", "upb_mn", "n19", "intensity", "ebo19"]]
      .round(2).to_string(index=False))
m.to_csv(os.path.join(OUT, "gmar_matched.csv"), index=False)
json.dump(out, open(os.path.join(OUT, "results_gmar_validation.json"), "w"), indent=1)

mac = {"gmarN": str(out["n"]),
       "gmarRho": ("$-$" if out["spearman"] < 0 else "") + f"{abs(out['spearman']):.2f}",
       "gmarPearson": ("$-$" if out["pearson"] < 0 else "") + f"{abs(out['pearson']):.2f}"}
pth = os.path.join(PAP, "numbers_pool.tex")
txt = open(pth, encoding="utf-8").read()
new = []
for k, v in sorted(mac.items()):
    pat = re.compile(r"(\\newcommand\{\\p" + k + r"\}\{)[^}]*(\})")
    if pat.search(txt):
        txt = pat.sub(lambda mm: mm.group(1) + v + mm.group(2), txt)
    else:
        new.append("\\newcommand{\\p" + k + "}{" + v + "}")
if new:
    txt = txt.rstrip() + "\n" + "\n".join(new) + "\n"
open(pth, "w", encoding="utf-8").write(txt)
print(f"\nwrote {len(mac)} macros ({len(new)} new)")
