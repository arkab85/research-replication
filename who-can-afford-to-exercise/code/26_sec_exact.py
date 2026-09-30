"""Exact-concept XBRL pull for FY2019. No heuristics: a named us-gaap tag or nothing.
Where a tag is missing, print what the filer actually reported so the choice is deliberate."""
import urllib.request, json, os, time
import numpy as np, pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA, "Accept": "application/json"}

ISSUERS = {
    4094: ("PennyMac Loan Services",  "PennyMac Financial Services", "0001745916"),
    4052: ("Nationstar Mortgage",     "Mr. Cooper Group",            "0000933136"),
    2397: ("PHH Mortgage",            "Ocwen Financial",             "0000873860"),
    4206: ("NewRez",                  "New Residential Investment",  "0001556593"),
    1555: ("Guild Mortgage",          "Guild Holdings",              "0001821160"),
    4042: ("Quicken Loans",           "Rocket Companies",            "0001805284"),
    4180: ("loanDepot.com",           "loanDepot",                   "0001831631"),
    4255: ("Home Point Financial",    "Home Point Capital",          "0001830197"),
}
EXACT = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
               "MembersEquity", "PartnersCapital"],
    "cash":   ["CashAndCashEquivalentsAtCarryingValue",
               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "msr":    ["ServicingAssetAtFairValueAmount", "ServicingAsset"],
}
END = "2019-12-31"

def cf(cik):
    u = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try: return json.loads(urllib.request.urlopen(
        urllib.request.Request(u, headers=H), timeout=90).read())
    except Exception as e: return {"__err__": str(e)[:60]}

def val(fx, tag, end=END):
    n = fx.get("facts", {}).get("us-gaap", {}).get(tag)
    if not n: return None
    rows = [r for r in n.get("units", {}).get("USD", []) if r.get("end") == end]
    if not rows: return None
    rows.sort(key=lambda r: (r.get("fy") or 0, r.get("filed") or ""))
    return float(rows[-1]["val"])

rows, cache = [], {}
for iid, (nm, parent, cik) in ISSUERS.items():
    if cik not in cache: cache[cik] = cf(cik); time.sleep(0.3)
    fx = cache[cik]
    r = dict(issuer_id=iid, issuer=nm, parent=parent, cik=cik)
    if "__err__" in fx: rows.append(r); print(parent, "ERR"); continue
    for k, tags in EXACT.items():
        for t in tags:
            v = val(fx, t)
            if v is not None:
                r[k], r[k + "_tag"] = v, t; break
    miss = [k for k in EXACT if k not in r]
    print(f"\n{parent:<30} " + "  ".join(
        f"{k}={r.get(k, 0)/1e9:.2f}B" for k in EXACT if k in r) +
        (f"   MISSING: {miss}" if miss else ""))
    if miss:
        gaap = fx.get("facts", {}).get("us-gaap", {})
        for k in miss:
            key = k if k != "msr" else "servicing"
            cand = [t for t in gaap if key[:5] in t.lower()
                    and any(x.get("end") == END for x in
                            gaap[t].get("units", {}).get("USD", []))]
            print(f"    tags available at {END} matching '{key[:5]}': {cand[:8]}")
    rows.append(r)

sec = pd.DataFrame(rows)
sec.to_csv(os.path.join(OUT, "sec_liquidity.csv"), index=False)

iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
m = sec.set_index("issuer_id").join(
    iss[["name", "ext", "n19", "ebo19", "ebo20", "d_ebo", "S"]])
m = m[m.S.notna()].copy()
m["cash_assets"] = m.cash / m.assets
m["cash_msr"] = m.cash / m.msr
m["equity_assets"] = m.equity / m.assets
m["log_msr"] = np.log(m.msr)
m["log_assets"] = np.log(m.assets)
m = m.sort_values("S", ascending=False)
m.to_csv(os.path.join(OUT, "sec_merged.csv"))

print("\n" + "=" * 100)
print("FY2019 BALANCE SHEET vs. THE SCALE PROXY")
print("=" * 100)
sh = m[["parent", "ext", "n19", "S", "assets", "cash", "msr", "equity",
        "cash_assets", "cash_msr", "equity_assets", "ebo19", "d_ebo"]].copy()
for c in ["assets", "cash", "msr", "equity"]: sh[c] = (sh[c] / 1e9).round(2)
for c in ["cash_assets", "cash_msr", "equity_assets", "ebo19", "d_ebo"]:
    sh[c] = sh[c].round(3)
print(sh.to_string())

print("\n--- validation: is log option volume measuring the servicing book? ---")
d = m[["S", "log_msr", "log_assets", "cash_assets", "equity_assets", "cash_msr"]].dropna(
    subset=["S"])
for b in ["log_msr", "log_assets", "cash_assets", "equity_assets", "cash_msr"]:
    dd = d[["S", b]].dropna()
    if len(dd) >= 4:
        print(f"   corr(S, {b:<14}) = {dd.S.corr(dd[b]):+.3f}  "
              f"rank {dd.S.corr(dd[b], method='spearman'):+.3f}  n={len(dd)}")
json.dump({"n": int(len(m))}, open(os.path.join(OUT, "results_sec.json"), "w"))
