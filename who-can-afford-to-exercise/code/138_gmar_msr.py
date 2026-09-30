"""Ginnie Mae servicing UPB by issuer, from the free monthly market report.

The size-gradient test measures issuer scale by the count of buyout options vesting in
2019. Both reviewer reports objected that this is a proxy for the quantity to be funded
rather than a measure of it. Ginnie Mae's monthly Global Markets Analysis Report carries a
named table -- "the 30 largest owners of mortgage servicing rights by UPB for Ginnie Mae
MBS" -- with dollar balances, published free and without a login.

That is the servicing book itself, in dollars, from the agency, for the largest issuers.
It gives a direct check on the proxy and, for the issuers it covers, a delinquency
INTENSITY measure: options vesting per dollar of book, which is much closer to the
theoretical object than a raw count.
"""
import urllib.request, os, re, ssl, io
import pandas as pd
import pymupdf
from config import OUT

UA = {"User-Agent": "Academic research (replication)"}
ctx = ssl.create_default_context()
CACHE = os.path.join(OUT, "gmar")
os.makedirs(CACHE, exist_ok=True)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
WANT = [(2019, "jun"), (2019, "dec"), (2020, "mar"), (2020, "jun"),
        (2020, "sep"), (2020, "dec"), (2021, "jun")]
BASES = ["https://www.ginniemae.gov/data_and_reports/reporting/Documents/",
         "https://www.ginniemae.gov/s3/sites/default/files/listing_pdfs/"]


def fetch(y, m):
    name = f"global_market_analysis_{m}{str(y)[2:]}.pdf"
    p = os.path.join(CACHE, name)
    if os.path.exists(p) and os.path.getsize(p) > 50_000:
        return open(p, "rb").read()
    for b in BASES:
        try:
            with urllib.request.urlopen(urllib.request.Request(b + name, headers=UA),
                                        timeout=120, context=ctx) as r:
                d = r.read()
            if d[:4] == b"%PDF":
                open(p, "wb").write(d)
                return d
        except Exception:
            continue
    return None


NUM = r"[\d,]+(?:\.\d+)?"


def parse(pdf, tag):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    rows = []
    for i in range(doc.page_count):
        t = doc[i].get_text()
        if "Mortgage Servicing Rights" not in t and "MSR" not in t:
            continue
        if "largest owners" not in t and "Holders of Ginnie Mae" not in t:
            continue
        lines = [x.strip() for x in t.split("\n") if x.strip()]
        for j, ln in enumerate(lines):
            # "1  FREEDOM MORTGAGE CORPORATION  240,123  12.45%  12.45%"
            m = re.match(rf"^(\d{{1,2}})\s+(.{{4,60}}?)\s+({NUM})\s+({NUM})\s*%", ln)
            if m and int(m.group(1)) <= 30:
                rows.append({"rank": int(m.group(1)), "holder": m.group(2).strip(),
                             "upb_mn": float(m.group(3).replace(",", "")),
                             "share": float(m.group(4)), "file": tag})
        if rows:
            break
    return rows


all_rows = []
for y, m in WANT:
    d = fetch(y, m)
    if d is None:
        print(f"  {m}{str(y)[2:]}: not retrievable")
        continue
    r = parse(d, f"{y}{MONTHS.index(m)+1:02d}")
    print(f"  {m}{str(y)[2:]}: {len(d)/1e6:.1f}MB, {len(r)} MSR-holder rows")
    all_rows += r

if not all_rows:
    print("\nno MSR tables parsed; dumping a candidate page for inspection")
    d = fetch(2021, "jun") or fetch(2019, "jun")
    if d:
        doc = pymupdf.open(stream=d, filetype="pdf")
        for i in range(doc.page_count):
            t = doc[i].get_text()
            if "Servicing Rights" in t:
                print(f"--- page {i+1} ---")
                print(t[:1500])
                break
    raise SystemExit

g = pd.DataFrame(all_rows)
g.to_csv(os.path.join(OUT, "gmar_msr_holders.csv"), index=False)
print(f"\n{len(g)} rows across {g.file.nunique()} report months")
print(g[g.file == g.file.min()].head(12).to_string(index=False))
