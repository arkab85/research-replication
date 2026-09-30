"""Two public routes to a post-2020 buyout series.

(A) EDGAR full-text search: does 'loans eligible for repurchase' appear in filings, and
    whose? That line item is the ASC 860-50 recognition of vested Ginnie Mae buyout
    options, so its level is the stock of unexercised options.
(B) SEC Financial Statement Data Sets: quarterly ZIPs carrying EVERY tag including
    company extensions, which the companyfacts API omits.
(C) Ginnie Mae's own published aggregate reports.
"""
import urllib.request, urllib.error, json, io, zipfile, os, time
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA}
def get(u, timeout=90, hdr=None):
    return urllib.request.urlopen(
        urllib.request.Request(u, headers=hdr or H), timeout=timeout).read()

print("=" * 88); print("A. EDGAR FULL-TEXT SEARCH"); print("=" * 88, flush=True)
for q in ['%22loans%20eligible%20for%20repurchase%22',
          '%22loans%20eligible%20for%20repurchase%22%20%22Ginnie%20Mae%22']:
    u = f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=10-Q,10-K"
    try:
        r = json.loads(get(u, 60))
        hits = r.get("hits", {})
        print(f"\n  query {q[:55]}...  total={hits.get('total',{}).get('value')}")
        for h in hits.get("hits", [])[:10]:
            s = h.get("_source", {})
            print(f"    {s.get('file_date','')}  {str(s.get('display_names',[''])[0])[:52]:<52}"
                  f" {s.get('form','')}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on efts; trying the UI endpoint")
        try:
            u2 = f"https://www.sec.gov/cgi-bin/srqsb?text={q}"
            print("   (skipping legacy endpoint)")
        except Exception: pass
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:80])

print("\n" + "=" * 88); print("B. SEC FINANCIAL STATEMENT DATA SETS"); print("=" * 88, flush=True)
u = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2020q3.zip"
try:
    b = get(u, 240)
    print(f"  downloaded 2020q3.zip: {len(b)/1e6:.1f} MB")
    z = zipfile.ZipFile(io.BytesIO(b))
    print("  members:", z.namelist())
    tag = z.read("tag.txt").decode("utf-8", "replace").split("\n")
    hdr = tag[0].split("\t")
    ti = hdr.index("tag"); di = hdr.index("doc") if "doc" in hdr else None
    hits = [l for l in tag[1:] if "repurchas" in l.lower() and "eligible" in l.lower()]
    print(f"  tag.txt rows {len(tag):,};  'eligible...repurchase' tags: {len(hits)}")
    for l in hits[:15]:
        p = l.split("\t")
        print("    ", p[ti][:60], "|", (p[di][:80] if di else ""))
except Exception as e:
    print("  ERR", type(e).__name__, str(e)[:160])

print("\n" + "=" * 88); print("C. GINNIE MAE PUBLIC AGGREGATE REPORTS"); print("=" * 88, flush=True)
for u in ["https://www.ginniemae.gov/data_and_reports/reporting/Documents/global_market_analysis_mar22.pdf",
          "https://www.ginniemae.gov/data_and_reports/reporting/Documents/global_market_analysis_jun24.pdf"]:
    try:
        b = get(u, 120)
        ok = b[:4] == b"%PDF"
        print(f"  {'PDF ok' if ok else 'not a pdf'}  {len(b)/1e6:.1f} MB  {u.split('/')[-1]}")
        if ok:
            p = os.path.join(OUT,
                             u.split("/")[-1])
            open(p, "wb").write(b)
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:100], u.split("/")[-1])
