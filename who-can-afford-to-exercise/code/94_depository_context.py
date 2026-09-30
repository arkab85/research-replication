import re, glob, os
from config import CACHE as C
JOBS = [("BOK", "875357_*", r"loans eligible for repurchase|GNMA mortgage pools that"),
        ("Flagstar", "1033012_*",
         r"repurchased or (?:are )?eligible to be repurchased|loans with government guarantees|"
         r"eligible for repurchase|repurchase loans sold to GNMA")]
NUMLINE = re.compile(r"\|\s*\$?\s*[\d][\d,]{2,}")
for who, pat, rx in JOBS:
    for f in sorted(glob.glob(os.path.join(C, pat))):
        if f.endswith(".json"):
            continue
        b = open(f, "rb").read().decode("utf8", "ignore")
        t = re.sub(r"(?s)<(script|style).*?</\1>", " ", b)
        t = re.sub(r"<[^>]+>", "|", t)
        t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#32;|&#xA0;", " ", t)
        t = re.sub(r"(\|\s*)+", "|", t)
        print("\n###", who, os.path.basename(f)[:60])
        n = 0
        for m in re.finditer(rx, t, re.I):
            seg = t[m.start():m.start() + 240]
            if not NUMLINE.search(seg):
                continue
            print("   > " + t[max(0, m.start() - 230):m.start()][-230:])
            print("   L " + seg)
            n += 1
            if n >= 3:
                break
        if n == 0:
            print("   (no numeric hit)")
