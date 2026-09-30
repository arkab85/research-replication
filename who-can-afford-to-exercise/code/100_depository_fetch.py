"""Fetch the depository issuers' FY2019-FY2021 10-Ks and find where each one states
its Ginnie Mae repurchase-option balance.

Banks disclose the same ASC 860-50 item as the nonbanks but with their own wording,
and the older filings are not in the submissions API's `recent` block, so page the
shards. This script only DISCOVERS the phrasing; the numbers are read afterwards.
"""
import urllib.request, json, os, re, time
import pandas as pd
from config import OUT, CACHE, SEC_UA

os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": SEC_UA}

BANKS = [("Wells Fargo", 72971), ("U.S. Bancorp", 36104), ("JPMorgan Chase", 19617),
         ("PNC Financial", 713676), ("Bank of America", 70858), ("Citigroup", 831001),
         ("M&T Bank", 36270), ("Fifth Third", 35527), ("Truist", 92230),
         ("Flagstar Bancorp", 1033012), ("BOK Financial", 875357)]


def get(u, name, tries=3):
    p = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", name))
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return open(p, "rb").read()
    for i in range(tries):
        try:
            b = urllib.request.urlopen(
                urllib.request.Request(u, headers=H), timeout=240).read()
            open(p, "wb").write(b)
            time.sleep(0.3)
            return b
        except Exception as e:
            if i == tries - 1:
                print(f"    fetch failed ({type(e).__name__})")
                return None
            time.sleep(2)


def all_filings(cik):
    """The recent block plus every older shard."""
    b = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"sub_{cik}.json")
    if not b:
        return pd.DataFrame()
    j = json.loads(b)
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"].get("files", []):
        bb = get("https://data.sec.gov/submissions/" + f["name"], f"sub_{cik}_{f['name']}")
        if bb:
            frames.append(pd.DataFrame(json.loads(bb)))
    d = pd.concat(frames, ignore_index=True)
    return d[["accessionNumber", "filingDate", "form", "primaryDocument"]]


RX = re.compile(r"(?:GNMA|Ginnie\s*Mae)", re.I)
REP = re.compile(r"repurchas", re.I)

rows = []
for nm, cik in BANKS:
    f = all_filings(cik)
    if f.empty:
        print(f"\n{nm}: no filings"); continue
    k = f[(f.form == "10-K") & f.filingDate.between("2020-01-01", "2022-06-30")]
    k = k.sort_values("filingDate")
    print(f"\n{nm} (CIK {cik}): {len(k)} annual reports in window")
    for _, r in k.iterrows():
        u = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
             f"{r.accessionNumber.replace('-','')}/{r.primaryDocument}")
        b = get(u, f"{cik}_{r.accessionNumber}_{r.primaryDocument}")
        if not b:
            continue
        t = re.sub(r"(?s)<(script|style).*?</\1>", " ", b.decode("utf8", "ignore"))
        t = re.sub(r"<[^>]+>", "|", t)
        t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#32;|&#xA0;", " ", t)
        t = re.sub(r"(\|\s*)+", "|", t)
        found = []
        for m in RX.finditer(t):
            seg = t[max(0, m.start() - 260):m.start() + 340]
            if not REP.search(seg):
                continue
            if not re.search(r"\|\s*\$?\s*\d[\d,]{2,}", seg):
                continue
            found.append(seg)
        print(f"  {r.form} {r.filingDate} ({len(b)/1e6:.0f}MB): {len(found)} candidate line(s)")
        for s in found[:3]:
            print("     ... " + re.sub(r"\s+", " ", s)[:300])
        rows.append({"bank": nm, "cik": cik, "filed": r.filingDate,
                     "doc": r.primaryDocument, "candidates": len(found)})

pd.DataFrame(rows).to_csv(os.path.join(OUT, "depository_scan.csv"), index=False)
print("\nwrote depository_scan.csv")
