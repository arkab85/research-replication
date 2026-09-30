import urllib.request, re, json, sys
from config import SEC_UA
H = {"User-Agent": SEC_UA}

def get(u, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=timeout).read()

js = get("https://bulk.ginniemae.gov/main-YNJ7TS24.js").decode("utf-8", "replace")
print("main js bytes:", len(js))
for k in ["llmon", ".zip", "ownload", "apiUrl", "baseUrl", "/api", "assets/", "environment"]:
    idx = [m.start() for m in re.finditer(re.escape(k), js)][:5]
    print("===", k, len(idx))
    for i in idx:
        print("   ...", js[max(0, i - 100):i + 120].replace("\n", " "))

# also scan the lazily-loaded chunks referenced from index.html
idx_html = get("https://bulk.ginniemae.gov/").decode("utf-8", "replace")
chunks = sorted(set(re.findall(r'(chunk-[A-Z0-9]+\.js)', idx_html + js)))
print("\nchunks:", chunks[:20])
for c in chunks[:12]:
    try:
        t = get("https://bulk.ginniemae.gov/" + c).decode("utf-8", "replace")
    except Exception as e:
        print(" ", c, "ERR", e); continue
    hits = re.findall(r'["\'`](/[A-Za-z0-9_\-/\.]{4,80})["\'`]', t)
    api = sorted({h for h in hits if any(s in h.lower()
                  for s in ["api", "download", "file", "disclosure", "bulk"])})
    if api:
        print(" ", c, len(t), api[:25])
