"""Print the balance-sheet neighbourhoods so the FY2019 column is identified by eye,
not by position. Nothing goes into the paper from a regex guess.
"""
import urllib.request, re, os, sys, time
from config import OUT, SEC_UA
H = {"User-Agent": SEC_UA}

DOCS = {
    "Caliber": "https://www.sec.gov/Archives/edgar/data/1821440/000119312521013656/d103175ds1a.htm",
    "AmeriHome": "https://www.sec.gov/Archives/edgar/data/1820807/000104746921000002/a2242685zs-1a.htm",
}

for nm, u in DOCS.items():
    p = os.path.join(OUT, f"s1_{nm}.htm")
    if not os.path.exists(p):
        b = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=120).read()
        open(p, "wb").write(b)
        time.sleep(0.5)
    t = open(p, encoding="utf8", errors="ignore").read()
    t = re.sub(r"<[^>]+>", " | ", t)
    t = re.sub(r"&nbsp;?|&#160;|&#8217;|&amp;", " ", t)
    t = re.sub(r"[ \t]*\|[ \t|]*", " | ", t)
    t = re.sub(r"\s+", " ", t)
    print("\n" + "=" * 96)
    print(nm)
    print("=" * 96)
    for m in re.finditer(r"Total assets", t):
        s = max(0, m.start() - 900)
        seg = t[s:m.start() + 180]
        if "December 31" not in seg:
            continue
        seg = re.sub(r"(\| ){2,}", "| ", seg)
        print("\n---")
        print(seg[-1000:])
