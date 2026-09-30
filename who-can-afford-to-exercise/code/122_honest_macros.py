"""The two confidence-set macros the within-pool sensitivity discussion quotes."""
import json, os
from config import OUT, PAPER as PAP

r = json.load(open(os.path.join(OUT, "results_honest_avg.json")))
f = r["full"]


def num(x, k=2):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


mac = {
    "HonFullPtSet": f"[{num(f['pt'][0])}, {num(f['pt'][1])}]",
    "HonFullQuarter": f"[{num(f['M0.25'][0])}, {num(f['M0.25'][1])}]",
}
p = os.path.join(PAP, "numbers_pool.tex")
have = open(p, encoding="utf-8").read()
with open(p, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in mac.items():
        if "\\p" + k + "}" in have:
            print(f"  (already defined: {k})")
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
        print(f"  \\p{k} = {v}")
print(f"\nbreakdown, full {f['breakdown']}   July {r['flat']['breakdown']}")
print(f"largest |pre| full {f['max_abs_pre']}   July {r['flat']['max_abs_pre']}")
