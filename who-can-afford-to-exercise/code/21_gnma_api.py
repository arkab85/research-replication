import urllib.request, urllib.error, json
from config import SEC_UA
H = {"User-Agent": SEC_UA, "Accept": "application/json"}
B = "https://www.ginniemae.gov/bulk-content/api"
D = "https://www.ginniemae.gov/disclosure-api/api"

def probe(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40)
        b = r.read()
        return r.status, r.headers.get("Content-Type", ""), b
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()[:200]
    except Exception as e:
        return None, type(e).__name__, str(e)[:160].encode()

CANDS = [
    f"{B}/directories", f"{B}/files", f"{B}/filelist",
    f"{B}/directories/disclosurehistory", f"{B}/content/directories",
    f"{B}/bulk/directories", f"{B}/download/directories",
    f"{D}/directories", f"{D}/files",
    "https://www.ginniemae.gov/api/v1/bulk/files",
]
for u in CANDS:
    s, ct, b = probe(u)
    head = b[:260].decode("utf-8", "replace") if isinstance(b, bytes) else str(b)
    print(f"{str(s):>5} {ct[:30]:<30} {u}")
    if s == 200 and "json" in ct:
        print("      ", head.replace("\n", " ")[:250])
