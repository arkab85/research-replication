import urllib.request, urllib.error, urllib.parse, json
from config import SEC_UA
H = {"User-Agent": SEC_UA, "Accept": "application/json"}
A = "https://www.ginniemae.gov/disclosure-api/api/files?directory="

def ls(d):
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(A + urllib.parse.quote(d), headers=H), timeout=60)
        return json.loads(r.read())
    except Exception:
        return None

seen, queue, found = set(), [""], []
while queue:
    d = queue.pop(0)
    if d in seen or len(seen) > 400: continue
    seen.add(d)
    g = ls(d)
    if not g: continue
    for grp in g:
        nm = grp.get("name")
        files = grp.get("files", [])
        zips = [f for f in files if str(f.get("fileExtension", "")).lower() == ".zip"]
        llm = [f for f in files if "llmon" in str(f.get("fileName", "")).lower()]
        if llm or (zips and len(zips) > 3):
            print(f"\n### dir={d!r} group={nm!r} files={len(files)} zips={len(zips)} llmon={len(llm)}")
            for f in (llm or zips)[:8]:
                print("   ", f.get("fileName"), f.get("fileExtension"),
                      f.get("fileSize"), f.get("fileModified"))
            found.append((d, nm, len(files)))
        # queue plausible child directories
        child = f"{d}/{nm}" if d else nm
        if child not in seen and len(seen) < 400:
            queue.append(child)

print("\nvisited", len(seen), "paths")
print("candidates:", found[:20])
