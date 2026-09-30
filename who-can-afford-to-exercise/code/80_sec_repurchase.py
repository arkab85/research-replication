"""Find the buyout in public filings.

Under ASC 860-50 a Ginnie Mae issuer recognises loans it has the unilateral right to
repurchase as an asset, with an offsetting liability, from the moment the option vests.
So the stock of vested-but-unexercised options sits on the balance sheet of every issuer
that files with the SEC, quarterly, and keeps being reported long after this paper's
loan-level extract stops.

If that is there, it supplies all three of the things the paper is missing: a series that
runs past September 2020, a within-issuer panel instead of a 24-issuer cross-section, and
liquidity measured directly alongside the exercise decision.
"""
import urllib.request, urllib.error, json, os, time, re
import pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA, "Accept": "application/json"}

ISSUERS = {
    "PennyMac Financial":      "0001745916",
    "Mr. Cooper Group":        "0000933136",
    "Ocwen Financial":         "0000873860",
    "Rithm / New Residential": "0001556593",
    "loanDepot":               "0001831631",
    "UWM Holdings":            "0001783398",
    "Rocket Companies":        "0001805284",
    "Guild Holdings":          "0001821160",
    "Home Point Capital":      "0001830197",
    "PennyMac Mortgage Trust": "0001464423",
    "Flagstar / NYCB":         "0000910073",
    "Carrington?":             None,
}

def facts(cik):
    u = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        return json.loads(urllib.request.urlopen(
            urllib.request.Request(u, headers=H), timeout=90).read())
    except Exception as e:
        return {"__err__": str(e)[:60]}

PAT = re.compile(r"(?i)repurchas|eligible.*repurchas|ginnie|gnma")
print("=" * 90)
print("SEARCHING EVERY TAXONOMY FOR REPURCHASE-OPTION TAGS")
print("=" * 90, flush=True)
found = {}
for name, cik in ISSUERS.items():
    if not cik: continue
    fx = facts(cik); time.sleep(0.3)
    if "__err__" in fx:
        print(f"\n{name:<26} ERROR {fx['__err__']}"); continue
    hits = []
    for tax, tags in fx.get("facts", {}).items():
        for tag, node in tags.items():
            if PAT.search(tag):
                n = sum(len(v) for v in node.get("units", {}).values())
                lab = node.get("label") or ""
                hits.append((tax, tag, n, lab[:70]))
    hits.sort(key=lambda x: -x[2])
    print(f"\n{name:<26} (CIK {cik})  entity={fx.get('entityName','')[:40]}")
    if not hits:
        print("    no repurchase-related tags")
    for tax, tag, n, lab in hits[:8]:
        print(f"    {tax:<10} {tag:<52} n={n:<5} {lab}")
    found[name] = {"cik": cik, "tags": [(t, g, n) for t, g, n, _ in hits[:12]]}

json.dump(found, open(os.path.join(OUT, "sec_repurchase_tags.json"), "w"), indent=1)
print("\nwrote sec_repurchase_tags.json")
