"""Search cached 10-Ks for GNMA buyout-option disclosures. Prints short windows only."""
import re, glob, os, sys
from config import CACHE as C


def flat(f):
    b = open(f, "rb").read().decode("utf8", "ignore")
    t = re.sub(r"(?s)<(script|style).*?</\1>", " ", b)
    t = re.sub(r"<[^>]+>", "|", t)
    t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#32;|&#xA0;", " ", t)
    t = re.sub(r"&#8217;|&#146;|&#x2019;", "'", t)
    t = re.sub(r"&#8212;|&#151;|&#x2014;", "-", t)
    t = re.sub(r"&amp;", "&", t)
    t = re.sub(r"(\|\s*)+", "|", t)
    return t

def run(pat, rx, pre=260, post=420, lim=12, needdigit=False):
    for f in sorted(glob.glob(os.path.join(C, pat))):
        bn = os.path.basename(f)
        if bn.endswith(".json") or bn.startswith("sub_"):
            continue
        t = flat(f)
        print("\n" + "=" * 100)
        print(bn)
        print("=" * 100)
        n = 0
        seen = set()
        for m in re.finditer(rx, t, re.I):
            seg = re.sub(r"\s+", " ", t[m.start():m.start() + post])
            if needdigit and not re.search(r"\d", seg):
                continue
            key = re.sub(r"[^a-z0-9]", "", seg.lower())[:70]
            if key in seen:
                continue
            seen.add(key)
            p = re.sub(r"\s+", " ", t[max(0, m.start() - pre):m.start()])
            print("\n --- PRE: " + p[-pre:])
            print(" +++ HIT: " + seg)
            n += 1
            if n >= lim:
                break
        if n == 0:
            print("  (no hits)")

if __name__ == "__main__":
    pat = sys.argv[1]
    rx = sys.argv[2]
    pre = int(sys.argv[3]) if len(sys.argv) > 3 else 260
    post = int(sys.argv[4]) if len(sys.argv) > 4 else 420
    lim = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    nd = len(sys.argv) > 6 and sys.argv[6] == "d"
    run(pat, rx, pre, post, lim, nd)
