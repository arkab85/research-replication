"""Widen the public balance-sheet coverage of the nonbank issuers in the gradient test.

Several of the large nonbank issuers were private in 2019 but filed an S-1 or a
registration statement in 2020-21 that carries an audited FY2019 balance sheet.
Those filings are in EDGAR and in the XBRL company facts, so the capacity measure
does not have to stop at the firms that were already reporting.
"""
import urllib.request, urllib.parse, json, os, re, time
import pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA}

NAMES = ["Caliber Home Loans", "AmeriHome Mortgage", "Home Point Capital",
         "loanDepot", "Guild Holdings", "UWM Holdings", "Finance of America",
         "Ocwen Financial", "Mr. Cooper Group", "PennyMac Financial Services",
         "Rocket Companies", "New Residential Investment", "Carrington",
         "Bayview", "Planet Financial Group", "Freedom Mortgage"]


def get(u, tries=3):
    for i in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(u, headers=H), timeout=60).read()
        except Exception as e:
            if i == tries - 1:
                return None
            time.sleep(1.5)


def find_cik(name):
    u = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company="
         + urllib.parse.quote(name) + "&type=&dateb=&owner=include&count=20&output=atom")
    b = get(u)
    if not b:
        return []
    s = b.decode("utf8", "ignore")
    out = []
    for m in re.finditer(r"<company-info>(.*?)</company-info>", s, re.S):
        blk = m.group(1)
        c = re.search(r"<cik>(\d+)</cik>", blk)
        n = re.search(r"<conformed-name>(.*?)</conformed-name>", blk)
        if c and n:
            out.append((c.group(1).zfill(10), n.group(1)))
    if not out:  # single-company pages put it in the header
        c = re.search(r"CIK=(\d{10})", s)
        n = re.search(r"<conformed-name>(.*?)</conformed-name>", s)
        if c:
            out.append((c.group(1), n.group(1) if n else name))
    return out


TAGS = {"assets": ["Assets"],
        "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndDueFromBanks",
                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
        "msr": ["ServicingAssetAtFairValueAmount", "ServicingAsset"]}


def facts(cik):
    b = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if not b:
        return None
    try:
        return json.loads(b)
    except Exception:
        return None


def pick(j, tags, end_lo="2019-12-01", end_hi="2020-01-31"):
    for t in tags:
        recs = j.get("facts", {}).get("us-gaap", {}).get(t, {}).get("units", {}).get("USD", [])
        cand = [r for r in recs if r.get("end", "") >= end_lo and r.get("end", "") <= end_hi
                and "start" not in r and r.get("val") is not None]
        if cand:
            return float(max(cand, key=lambda r: abs(r["val"]))["val"]) / 1e9, t
    return None, None


rows = []
for nm in NAMES:
    hits = find_cik(nm)
    if not hits:
        print(f"{nm:<32} no EDGAR match"); continue
    cik, conf = hits[0]
    j = facts(cik)
    if not j:
        print(f"{nm:<32} CIK {cik} ({conf[:34]}) no XBRL facts"); continue
    a, ta = pick(j, TAGS["assets"])
    c, tc = pick(j, TAGS["cash"])
    m, tm = pick(j, TAGS["msr"])
    print(f"{nm:<32} CIK {cik}  assets {a}  cash {c}  msr {m}")
    rows.append({"name": nm, "edgar_name": conf, "cik": cik,
                 "assets": a, "cash": c, "msr": m,
                 "tag_assets": ta, "tag_cash": tc, "tag_msr": tm})
    time.sleep(0.2)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "balance_sheets_wide.csv"), index=False)
print("\n" + d.to_string())
print("\nwrote balance_sheets_wide.csv")
