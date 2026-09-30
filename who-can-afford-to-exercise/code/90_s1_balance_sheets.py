"""Caliber and AmeriHome were private in 2019 and their S-1s are not XBRL-tagged.
Pull the FY2019 balance sheet out of the filing text instead.
"""
import urllib.request, urllib.parse, json, os, re, time
import pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA}


def get(u, tries=3, sleep=1.5):
    for i in range(tries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(u, headers=H), timeout=90).read()
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(sleep)


def search(name):
    u = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company="
         + urllib.parse.quote(name) + "&type=S-1&dateb=&owner=include&count=40&output=atom")
    b = get(u)
    if not b:
        return []
    s = b.decode("utf8", "ignore")
    return list({(m.group(1).zfill(10), m.group(2))
                 for m in re.finditer(
                     r"<cik>(\d+)</cik>.*?<conformed-name>(.*?)</conformed-name>", s, re.S)}) \
        or [(m.group(1), "") for m in re.finditer(r"CIK=(\d{10})", s)][:1]


def filings(cik):
    b = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not b:
        return pd.DataFrame()
    j = json.loads(b)
    r = j["filings"]["recent"]
    d = pd.DataFrame({k: r[k] for k in
                      ["accessionNumber", "filingDate", "form", "primaryDocument"]})
    d["cik"] = int(cik)
    d["name"] = j.get("name", "")
    return d


def doc_url(cik, acc, doc):
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{acc.replace('-', '')}/{doc}")


NUM = r"\$?\s*\(?([\d,]{3,})\)?"


def scrape(txt, labels):
    """Return the first two numbers following a balance-sheet label."""
    t = re.sub(r"<[^>]+>", " ", txt)
    t = re.sub(r"&nbsp;?|&#160;", " ", t)
    t = re.sub(r"\s+", " ", t)
    out = {}
    for key, pats in labels.items():
        for p in pats:
            m = re.search(p + r"[^\d\(\)]{0,80}" + NUM + r"[^\d\(\)]{0,40}" + NUM,
                          t, re.I)
            if m:
                out[key] = (m.group(1), m.group(2))
                break
    return out


LABELS = {
    "assets": [r"Total\s+assets"],
    "cash": [r"Cash\s+and\s+cash\s+equivalents"],
    "msr": [r"Mortgage\s+servicing\s+rights", r"Servicing\s+rights"],
    "equity": [r"Total\s+(?:stockholders|shareholders|members)[’'\u2019]?\s*(?:equity|deficit)"],
}

TARGETS = ["Caliber Home Loans", "Aris Mortgage", "AmeriHome"]

found = []
for nm in TARGETS:
    for cik, conf in search(nm):
        f = filings(cik)
        if f.empty:
            continue
        s1 = f[f.form.str.startswith("S-1")]
        if s1.empty:
            continue
        print(f"\n{nm} -> CIK {cik} {f.name.iloc[0]}")
        print(s1.head(4).to_string(index=False))
        found.append((cik, f.name.iloc[0], s1.iloc[0]))
        break

rows = []
for cik, nm, r in found:
    u = doc_url(cik, r.accessionNumber, r.primaryDocument)
    print(f"\nfetching {nm}: {u}")
    b = get(u)
    if not b:
        print("  fetch failed"); continue
    txt = b.decode("utf8", "ignore")
    print(f"  {len(txt):,} chars")
    got = scrape(txt, LABELS)
    for k, v in got.items():
        print(f"    {k:<8} {v}")
    rows.append({"cik": cik, "name": nm, "url": u,
                 **{k: v[0] for k, v in got.items()},
                 **{k + "_prior": v[1] for k, v in got.items()}})

pd.DataFrame(rows).to_csv(os.path.join(OUT, "s1_balance_sheets.csv"), index=False)
print("\nwrote s1_balance_sheets.csv")
