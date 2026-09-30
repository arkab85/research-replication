import json, os
from config import OUT, PAPER as PAP
r = json.load(open(os.path.join(OUT, "results_shadow.json")))
nb, dp = r["cut_nonbank"], r["cut_depository"]
m = {
    "CutNbPre":   f"{nb['pctile_pre']:.0f}",  "CutNbPost":  f"{nb['pctile_post']:.0f}",
    "CutNbCPre":  f"{nb['cstar_pre']:.2f}",   "CutNbCPost": f"{nb['cstar_post']:.2f}",
    "CutNbBp":    f"{nb['bp']:.0f}",
    "CutDepPre":  f"{dp['pctile_pre']:.0f}",  "CutDepPost": f"{dp['pctile_post']:.0f}",
    "CutDepCPre": f"{dp['cstar_pre']:.2f}",   "CutDepCPost":f"{dp['cstar_post']:.2f}",
    "ForgonePts": f"{r['forgone_pts_D4']:.1f}",
}
p = os.path.join(PAP, "numbers_pool.tex")
existing = open(p, encoding="utf-8").read()
with open(p, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(m.items()):
        if "\\p" + k + "}" in existing:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
print("added:", m)
