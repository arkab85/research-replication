"""Print the exact balance-sheet neighbourhoods for the remaining issuers so every
number that enters the paper is read, not inferred."""
import re, os, glob, sys
from config import CACHE as C


WANT = {
    "MrCooper":  ("933136_*",  r"[Ll]oans [Ss]ubject to [Rr]epurchase"),
    "Ocwen":     ("873860_*",  r"[Rr]epurchase[d]? [Gg]innie [Mm]ae [Ll]oans|[Ll]oans held for investment"),
    "Flagstar":  ("1033012_*", r"[Rr]epurchased or (?:are )?eligible to be repurchased|LGG"),
    "BOK":       ("875357_*",  r"eligible for repurchase"),
}
pick = sys.argv[1] if len(sys.argv) > 1 else None

for who, (pat, rx) in WANT.items():
    if pick and who != pick:
        continue
    for f in sorted(glob.glob(os.path.join(C, pat))):
        if f.endswith(".json"):
            continue
        b = open(f, "rb").read().decode("utf8", "ignore")
        t = re.sub(r"(?s)<(script|style).*?</\1>", " ", b)
        t = re.sub(r"<[^>]+>", "|", t)
        t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#xA0;", " ", t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"(\|\s*)+", "|", t)
        print("\n" + "#" * 94)
        print(f"{who}   {os.path.basename(f)[:70]}")
        print("#" * 94)
        n = 0
        for m in re.finditer(rx, t):
            seg = t[m.start():m.start() + 300]
            if not re.search(r"\|\s*\$?\s*[\d,]{3,}", seg):
                continue
            pre = t[max(0, m.start() - 700):m.start()]
            print("\n--- [context] " + pre[-380:])
            print("--- [line]    " + seg)
            n += 1
            if n >= 5:
                break
        if n == 0:
            print("  (no numeric hit)")
