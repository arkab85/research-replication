"""Hand-collect FY2019 balance-sheet liquidity for the nonbank Ginnie Mae issuers whose
parents filed with the SEC, from XBRL company facts.

The paper's size gradient uses issuer scale as a proxy for the dollar quantity that had to
be funded. This script builds the thing the proxy stands in for: liquidity held against the
servicing book, measured before the shock.
"""
import urllib.request, urllib.error, json, os, time, re
import pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA, "Accept": "application/json"}

# Ginnie Mae issuer_id -> the SEC registrant whose FY2019 statements consolidate it.
# Mapping is by corporate parent at the 2019 fiscal year end.
ISSUERS = {
    4094: ("PennyMac Loan Services, LLC",     "PennyMac Financial Services, Inc.", "0001745916"),
    4052: ("Nationstar Mortgage, LLC",        "Mr. Cooper Group Inc.",             "0000933136"),
    2397: ("PHH Mortgage Corporation",        "Ocwen Financial Corporation",       "0000873860"),
    4092: ("PHH Mortgage Corporation",        "Ocwen Financial Corporation",       "0000873860"),
    4206: ("NewRez LLC",                      "New Residential Investment Corp.",  "0001556593"),
    4255: ("Home Point Financial Corp.",      "Home Point Capital Inc.",           "0001830197"),
    2546: ("Home Point Financial Corp.",      "Home Point Capital Inc.",           "0001830197"),
    1555: ("Guild Mortgage Company",          "Guild Holdings Company",            "0001821160"),
    4213: ("Caliber Home Loans, Inc.",        "Caliber Home Loans, Inc.",          "0001828161"),
    4180: ("loanDepot.com, LLC",              "loanDepot, Inc.",                   "0001831631"),
    3725: ("United Wholesale Mortgage",       "UWM Holdings Corporation",          "0001783398"),
    4042: ("Quicken Loans, LLC",              "Rocket Companies, Inc.",            "0001805284"),
    4085: ("Ditech Financial LLC",            "Ditech Holding Corporation",        "0001040719"),
}

CONCEPTS = {
    "cash":      ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndRestrictedCash",
                  "CashAndDueFromBanks"],
    "assets":    ["Assets"],
    "equity":    ["StockholdersEquity",
                  "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                  "MembersEquity"],
    "msr":       ["ServicingAssetAtFairValueAmount", "ServicingAsset",
                  "MortgageServicingRightsMSRImpairmentReversalValuationAllowance"],
    "debt":      ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt",
                  "SecuredDebt", "NotesPayable"],
    "restricted": ["RestrictedCashAndCashEquivalents", "RestrictedCash",
                   "RestrictedCashAndCashEquivalentsAtCarryingValue"],
}

def facts(cik):
    u = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60)
        return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"__err__": e.code}
    except Exception as e:
        return {"__err__": type(e).__name__}

def pick(fx, tags, want_end="2019-12-31"):
    """Value of the first matching tag as of FY2019 year end."""
    gaap = fx.get("facts", {}).get("us-gaap", {})
    for t in tags:
        if t not in gaap: continue
        for unit, rows in gaap[t].get("units", {}).items():
            if unit != "USD": continue
            cand = [r for r in rows if r.get("end") == want_end]
            if not cand:
                cand = [r for r in rows if str(r.get("end", "")).startswith("2019-1")]
            if cand:
                cand.sort(key=lambda r: (r.get("fy") or 0, r.get("filed") or ""))
                return float(cand[-1]["val"]), t
    return None, None

rows = []
seen = {}
for iid, (issuer, parent, cik) in ISSUERS.items():
    if cik not in seen:
        seen[cik] = facts(cik); time.sleep(0.25)
    fx = seen[cik]
    if "__err__" in fx:
        print(f"  {parent:<38} CIK {cik}  ERROR {fx['__err__']}")
        rows.append(dict(issuer_id=iid, issuer=issuer, parent=parent, cik=cik))
        continue
    rec = dict(issuer_id=iid, issuer=issuer, parent=parent, cik=cik,
               entity=fx.get("entityName"))
    for k, tags in CONCEPTS.items():
        v, t = pick(fx, tags)
        rec[k] = v; rec[k + "_tag"] = t
    rows.append(rec)
    print(f"  {parent:<38} assets={rec.get('assets')} cash={rec.get('cash')} "
          f"msr={rec.get('msr')}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "sec_liquidity_raw.csv"), index=False)
print("\nwrote sec_liquidity_raw.csv;", df.assets.notna().sum(), "of", len(df),
      "issuer rows have FY2019 total assets")
