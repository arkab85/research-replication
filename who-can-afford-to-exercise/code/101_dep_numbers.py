"""Read the depository balances out of the filings, with enough context to be sure
what each number is."""
import re, glob, os, sys
from config import CACHE as C


JOBS = {
    "JPM": ("19617_*", r"Loans repurchased or option to repurchase", 3),
    "FITB": ("35527_*", r"option to repurchase|deemed to have regained effective control|"
                        r"previously sold to GNMA", 8),
    "BOK": ("875357_*", r"GNMA repurchase liability|repurchase of certain delinquent residential|"
                        r"Government guaranteed loans eligible for repurchase|"
                        r"residential mortgage loans guaranteed", 8),
    "MTB": ("36270_*", r"repurchase[sd]? of government[- ]guaranteed|Ginnie Mae pools", 4),
}
pick = sys.argv[1] if len(sys.argv) > 1 else None

for who, (pat, rx, lim) in JOBS.items():
    if pick and who != pick:
        continue
    for f in sorted(glob.glob(os.path.join(C, pat))):
        if f.endswith(".json") or "sub_" in os.path.basename(f):
            continue
        b = open(f, "rb").read().decode("utf8", "ignore")
        t = re.sub(r"(?s)<(script|style).*?</\1>", " ", b)
        t = re.sub(r"<[^>]+>", "|", t)
        t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#32;|&#xA0;", " ", t)
        t = re.sub(r"&#8217;|&#146;", "'", t)
        t = re.sub(r"(\|\s*)+", "|", t)
        print("\n" + "#" * 92)
        print(f"{who}  {os.path.basename(f)[:66]}")
        print("#" * 92)
        n = 0
        seen = set()
        for m in re.finditer(rx, t, re.I):
            seg = re.sub(r"\s+", " ", t[m.start():m.start() + 320])
            if not re.search(r"\|\s*\$?\s*\d[\d,]{1,}", seg):
                continue
            key = seg[:90]
            if key in seen:
                continue
            seen.add(key)
            pre = re.sub(r"\s+", " ", t[max(0, m.start() - 330):m.start()])
            print("\n  > " + pre[-330:])
            print("  L " + seg)
            n += 1
            if n >= lim:
                break
        if n == 0:
            print("  (none)")
