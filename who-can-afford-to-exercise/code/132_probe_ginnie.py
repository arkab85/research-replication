"""Is Ginnie Mae's bulk disclosure actually gated, or only the loan-level files?

The design in prospect needs an issuer-month panel of servicing UPB and delinquent UPB.
The monthly single-family POOL files are said to carry issuer, delinquency category,
loan count and UPB, which would be enough. Earlier work established that the LOAN-level
monthly files sit behind a login. This checks, file by file, what the server actually
returns without credentials. Low volume: one listing and a handful of HEAD requests.
"""
import urllib.request, urllib.error, re, ssl

UA = {"User-Agent": "Academic research feasibility check (contact via institution)"}
ctx = ssl.create_default_context()


def get(url, method="GET", limit=400_000):
    req = urllib.request.Request(url, headers=UA, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            body = r.read(limit) if method == "GET" else b""
            return r.status, dict(r.headers), body, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(2000), url
    except Exception as e:
        return None, {}, str(e).encode()[:200], url


print("=" * 88)
print("1. THE DISCLOSURE LISTING")
print("=" * 88)
st, h, b, final = get("https://bulk.ginniemae.gov/Disclosure/")
print(f"  status {st}   final url {final}")
print(f"  content-type {h.get('Content-Type')}   length {len(b):,}")
txt = b.decode("utf8", "ignore")
if "login" in final.lower() or "login" in txt.lower()[:3000]:
    print("  -> the listing itself mentions or redirects to a login")
files = sorted(set(re.findall(r"dlfile=([A-Za-z0-9_/\.\-]+\.zip)", txt)))
print(f"  download links exposed: {len(files)}")
for f in files[:12]:
    print(f"     {f}")

print("\n" + "=" * 88)
print("2. WHAT THE SERVER RETURNS FOR SPECIFIC FILES")
print("=" * 88)
cands = [
    "data_bulk/monthlySFPS_202009.zip",       # pool-level, the month we need
    "data_bulk/nimonSFPS_202009.zip",         # non-issuer monthly pool supplement
    "data_bulk/llmon_202009.zip",             # loan level, known to be gated
    "data_bulk/issuers_202009.zip",
]
if files:
    cands = list(dict.fromkeys([files[0]] + cands))
for c in cands:
    url = "https://bulk.ginniemae.gov/protectedfiledownload.aspx?dlfile=" + c
    st, h, b, final = get(url, method="GET", limit=4096)
    ct = h.get("Content-Type", "")
    cl = h.get("Content-Length", "?")
    is_zip = b[:2] == b"PK"
    gated = ("text/html" in ct) or ("login" in final.lower())
    print(f"\n  {c}")
    print(f"    status {st}  type {ct}  length {cl}")
    print(f"    body starts with PK (a real zip): {is_zip}")
    print(f"    final url: {final[:110]}")
    print(f"    verdict: {'OPEN' if is_zip else 'GATED or missing'}")

print("\n" + "=" * 88)
print("3. THE LOGIN PAGE'S STATED REQUIREMENTS")
print("=" * 88)
for u in ["https://www.ginniemae.gov/disclosure/download-login",
          "https://bulk.ginniemae.gov/login.aspx"]:
    st, h, b, final = get(u, limit=200_000)
    t = re.sub(r"<[^>]+>", " ", b.decode("utf8", "ignore"))
    t = re.sub(r"\s+", " ", t)
    print(f"\n  {u}  -> status {st}, final {final[:90]}")
    for kw in ["free", "register", "no cost", "approval", "corporate", "organization",
               "email", "account"]:
        m = re.search(r"[^.]{0,90}\b" + kw + r"\b[^.]{0,90}\.", t, re.I)
        if m:
            print(f"    [{kw}] {m.group(0).strip()[:170]}")
