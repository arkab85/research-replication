import os, json, re
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "tex", "juefull")
m = json.load(open(os.path.join(H, "out", "geo_index.json")))
W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
out = {}
for k, v in m.items():
    out["".join(W[c] if c.isdigit() else c for c in k)] = v
open(os.path.join(T, "numbers_geo2.tex"), "w", encoding="utf-8").write("\n".join("\\newcommand{\\%s}{%s}" % (k, v) for k, v in out.items()) + "\n")
print("wrote", len(out), "macros"); print(open(os.path.join(T, "numbers_geo2.tex"), encoding="utf-8").read()[:220])
