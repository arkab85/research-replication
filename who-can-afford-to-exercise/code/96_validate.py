"""Does the loan-level immobilisation measure line up with the audited balances?

Four firms, two years, eight cells. The loan-level measure is a flow (unexercised
vested UPB over the year); the filing reports a point-in-time stock, so the levels
are not the same object and the ratio between them is what should be stable.
"""
import pandas as pd, numpy as np, os, re
from config import OUT, PAPER as PAP
d = pd.read_csv(os.path.join(OUT, "four_firm.csv"))

x = np.r_[d.unex2019.values, d.unex2020.values]
y = np.r_[d.rep2019.values, d.rep2020.values]
r_all = np.corrcoef(x, y)[0, 1]
r_log = np.corrcoef(np.log(x), np.log(y))[0, 1]
r_20 = np.corrcoef(d.unex2020, d.rep2020)[0, 1]
ratio = np.r_[d.unex2019 / d.rep2019, d.unex2020 / d.rep2020]
print(f"  n = {len(x)} firm-years")
print(f"  correlation, loan-level unexercised UPB vs reported balance : {r_all:+.3f}")
print(f"  in logs                                                     : {r_log:+.3f}")
print(f"  across the four firms in 2020 alone                         : {r_20:+.3f}")
print(f"  flow/stock ratio: mean {ratio.mean():.2f}, range "
      f"{ratio.min():.2f}-{ratio.max():.2f}")

mac = {"ValN": f"{len(x)}", "ValCorr": f"{r_all:+.3f}".replace("-", "$-$"),
       "ValCorrLog": f"{r_log:+.2f}".replace("-", "$-$"),
       "ValCorrTwenty": f"{r_20:+.3f}".replace("-", "$-$"),
       "ValRatioLo": f"{ratio.min():.1f}", "ValRatioHi": f"{ratio.max():.1f}"}
p = os.path.join(PAP, "numbers_pool.tex")
have = open(p, encoding="utf-8").read()
with open(p, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        if "\\p" + k + "}" in have:
            continue
        fh.write("\\newcommand{\\p" + k + "}{" + v + "}\n")
print(f"\nwrote {len(mac)} macros")
