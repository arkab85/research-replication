"""Legend below the map; drop the leading plus from numbers that read as prose."""
import os, re, json, subprocess
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "tex", "juefull")
p = os.path.join(H, "geo_main.py"); s = open(p, encoding="utf-8-sig").read()
a = 'ax.legend(frameon=False, fontsize=7.5, loc="lower left", handletextpad=.2)'
assert a in s; s = s.replace(a, 'ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=1, handletextpad=.2)')
open(p, "w", encoding="utf-8").write(s)
subprocess.run(["python", p], cwd=H, capture_output=True, text=True)
# effHi/effLo read as prose ("25 points more likely"); keep the sign only where it is a coefficient
j = json.load(open(os.path.join(H, "out", "geo_index.json")))
j["effHiAbs"] = j["effHi"].lstrip("+"); j["effLoSigned"] = j["effLo"]
W = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
out = {"".join(W[c] if c.isdigit() else c for c in k): v for k, v in j.items()}
open(os.path.join(T, "numbers_geo2.tex"), "w", encoding="utf-8").write("\n".join("\\newcommand{\\%s}{%s}" % (k, v) for k, v in out.items()) + "\n")
json.dump(j, open(os.path.join(H, "out", "geo_index.json"), "w"))
q = os.path.join(T, "main.tex"); m = open(q, encoding="utf-8").read()
for a_, b_ in [("the most advantaged quartile of neighborhoods \\effHi{} points more likely to be performing", "the most advantaged quartile of neighborhoods \\effHiAbs{} points more likely to be performing"),
               ("refusing relief had left borrowers in the most advantaged quartile of neighborhoods \\effHi{} points more likely", "refusing relief had left borrowers in the most advantaged quartile of neighborhoods \\effHiAbs{} points more likely")]:
    if a_ in m: m = m.replace(a_, b_)
m = m.replace("and left those in the least advantaged quartile no better off (\\effLo{}).", "and left those in the least advantaged quartile no better off (\\effLo{} points, s.e.\\ \\effLoSE{}).")
open(q, "w", encoding="utf-8").write(m)
r = subprocess.run([os.path.join(H, "tex", "tectonic.exe"), "main.tex"], cwd=T, capture_output=True, text=True)
print("compile:", "ok" if r.returncode == 0 else r.stderr[-600:])
