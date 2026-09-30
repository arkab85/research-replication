"""Parse the GMAR top-30 MSR table and validate the paper's scale proxy against dollars.

The report prints each cell on its own line: rank, holder, UPB, share, cumulative share.
Having actual Ginnie Mae servicing UPB by issuer turns the size proxy from something the
paper asserts measures the book into something checkable against the book itself.
"""
import os, re, json
import numpy as np
import pandas as pd
import pymupdf
from config import OUT, PAPER as PAP

CACHE = os.path.join(OUT, "gmar")
MONEY = re.compile(r"^\$?([\d,]{3,})$")
PCT = re.compile(r"^([\d.]+)%$")


def parse(path):
    doc = pymupdf.open(path)
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if "Holders of Ginnie Mae Mortgage Servicing Rights" not in t:
            continue
        asof = re.search(r"Data as of ([A-Za-z]+ \d{4})", t)
        L = [x.strip() for x in t.split("\n") if x.strip()]
        try:
            k = L.index("Cumulative") + 1
            if L[k].lower().startswith("share"):
                k += 1
        except ValueError:
            k = 0
        rows, j = [], k
        while j + 3 < len(L) and len(rows) < 30:
            if not re.fullmatch(r"\d{1,2}", L[j]):
                j += 1
                continue
            rank = int(L[j])
            name = L[j + 1]
            m = MONEY.match(L[j + 2].replace(" ", ""))
            if not m:
                j += 1
                continue
            share = PCT.match(L[j + 3].replace(" ", ""))
            rows.append({"rank": rank, "holder": name,
                         "upb_mn": float(m.group(1).replace(",", "")),
                         "share": float(share.group(1)) if share else np.nan,
                         "asof": asof.group(1) if asof else ""})
            j += 5
        if rows:
            return rows
    return []


all_rows = []
for f in sorted(os.listdir(CACHE)):
    if not f.endswith(".pdf"):
        continue
    r = parse(os.path.join(CACHE, f))
    for x in r:
        x["file"] = f.replace("global_market_analysis_", "").replace(".pdf", "")
    print(f"  {f.split('_')[-1]:<12} {len(r):>2} holders" +
          (f"   as of {r[0]['asof']}" if r else ""))
    all_rows += r

g = pd.DataFrame(all_rows)
g.to_csv(os.path.join(OUT, "gmar_msr_holders.csv"), index=False)
print(f"\n{len(g)} rows, {g.file.nunique()} reports, {g.holder.nunique()} distinct holders")

# ---------------------------------------------------------------- match to the sample
il = pd.read_csv(os.path.join(OUT, "issuer_level.csv"))
il["key"] = il.name.str.upper()
MAP = {
    "LAKEVIEW": "LAKEVIEW", "PENNYMAC": "PENNYMAC", "WELLS FARGO": "WELLS FARGO",
    "FREEDOM": "FREEDOM", "NATIONSTAR": "NATIONSTAR", "QUICKEN": "QUICKEN",
    "US BANK": "U. S. BANK", "NEWREZ": "NEWREZ", "CALIBER": "CALIBER",
    "AMERIHOME": "AMERIHOME", "CARRINGTON": "CARRINGTON", "MATRIX": "MATRIX",
    "PLANET": "PLANET", "GUILD": "GUILD", "LOANDEPOT": "LOANDEPOT",
    "MIDFIRST": "MIDFIRST", "TRUIST": "TRUIST", "PNC": "PNC", "M&T": "M&T",
    "FIFTH THIRD": "FIFTH THIRD", "JPMORGAN": "JP MORGAN", "CHASE": "JP MORGAN",
    "FLAGSTAR": "FLAGSTAR", "MONEY SOURCE": "MONEY SOURCE", "VILLAGE": "VILLAGE",
    "HOME POINT": "HOME POINT", "CITIZENS": "CITIZENS", "CROSSCOUNTRY": "CROSSCOUNTRY",
}
latest = g[g.file == "sep20"] if (g.file == "sep20").any() else g[g.file == g.file.iloc[0]]


def to_key(h):
    hu = h.upper()
    for k, v in MAP.items():
        if k in hu:
            return v
    return None


latest = latest.copy()
latest["key"] = latest.holder.apply(to_key)
m = latest.dropna(subset=["key"]).merge(
    il.assign(k2=il.key.str.extract(r"^([A-Z&\. ]+?)(?:,| INC| LLC| BANK|$)")[0].str.strip()),
    left_on="key", right_on="k2", how="inner")
if m.empty:
    m = latest.dropna(subset=["key"]).merge(
        il, left_on="key", right_on=il.key.str[:len("PENNYMAC")], how="inner")
print(f"\nmatched {m.issuer_id.nunique() if len(m) else 0} GMAR holders to sample issuers")

if len(m) > 5:
    m = m.drop_duplicates("issuer_id")
    from scipy.stats import spearmanr, pearsonr
    m["log_upb"] = np.log(m.upb_mn)
    rho, p1 = spearmanr(m.S, m.log_upb)
    r, p2 = pearsonr(m.S, m.log_upb)
    print(f"  Spearman(paper scale S, log Ginnie servicing UPB) = {rho:+.3f}  p={p1:.5f}")
    print(f"  Pearson  (paper scale S, log Ginnie servicing UPB) = {r:+.3f}  p={p2:.5f}")
    print(f"  n = {len(m)}")
    print("\n  delinquency intensity: 2019 options vesting per $bn of servicing book")
    m["intensity"] = m.n19 / (m.upb_mn / 1000)
    print(m.sort_values("intensity")[["name", "ext", "upb_mn", "n19", "intensity",
                                      "ebo19", "d_ebo"]].round(3).to_string(index=False))
    m.to_csv(os.path.join(OUT, "gmar_matched.csv"), index=False)
    json.dump({"spearman": round(float(rho), 3), "pearson": round(float(r), 3),
               "n": int(len(m))},
              open(os.path.join(OUT, "results_gmar_validation.json"), "w"), indent=1)
