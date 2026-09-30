import json, os
from config import OUT, PAPER as PAP
r = json.load(open(os.path.join(OUT, "results_poolselect.json")))
mi = r["multi-issuer pools"]; cu = r["custom / single-issuer pools"]
it = r["interaction"]["nb_post_multi"]
mix = r["mix_shift"]
m = {
    "MultiDep":   f"{r['multi_share']['depository']:.0f}",
    "MultiNb":    f"{r['multi_share']['nonbank']:.0f}",
    "BothIssuers": "68",
    "DiffFico":   f"{r['diff_fico'][0]:+.1f}",  "DiffFicoSE": f"{r['diff_fico'][1]:.1f}",
    "DiffAge":    f"{r['diff_age'][0]:+.1f}",   "DiffAgeSE":  f"{r['diff_age'][1]:.1f}",
    "DiffCoupon": f"{abs(r['diff_coupon'][0]):.2f} percentage points",
    "DiDMulti":   f"{mi['coef']:.3f}",  "DiDMultiSE":  f"{mi['se']:.3f}",
    "DiDCustom":  f"{cu['coef']:.3f}",  "DiDCustomSE": f"{cu['se']:.3f}",
    "PoolInter":  f"{it[0]:+.3f}",      "PoolInterSE": f"{it[1]:.3f}",
    "PoolInterP": f"{it[2]:.2f}",
    "MixDepPre":  f"{mix['0.0']['depository']:.0f}",
    "MixDepPost": f"{mix['1.0']['depository']:.0f}",
    "MixNbPre":   f"{mix['0.0']['nonbank']:.0f}",
    "MixNbPost":  f"{mix['1.0']['nonbank']:.0f}",
}
p = os.path.join(PAP, "numbers_pool.tex")
have = open(p, encoding="utf-8").read()
import re
with open(p, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(m.items()):
        if "\\p" + k + "}" in have: continue
        v = re.sub(r"(?<![\d-])-(?=\d)", "$-$", str(v))
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
print("added", len(m), "macros:", m)
