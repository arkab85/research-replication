"""Table and macros for the reversal section."""
import pandas as pd, json, os, re
from config import OUT, PAPER as PAP

R = json.load(open(os.path.join(OUT, "results_recovery_final.json")))
rec = json.load(open(os.path.join(OUT, "results_recovery.json")))
a = pd.read_csv(os.path.join(OUT, "pennymac_recovery.csv"), index_col=0, parse_dates=True)
# report the mean of the quarterly series as well as the peak: the text quotes means,
# and a table of maxima alone cannot be checked against it
yr = a.groupby(a.index.year).agg(mean=("elig", "mean"), mx=("elig", "max"),
                                 amean=("per_assets", "mean"), amx=("per_assets", "max"),
                                 n=("elig", "size"))

rows = ""
for y, r in yr.iterrows():
    mark = r"\;$\leftarrow$ shock" if y == 2020 else ""
    rows += (f"{y} & {r['mean']:.2f} & {r.mx:.2f} & {r.amean:.2f} & {r.amx:.2f} & "
             f"{int(r.n)} {mark} \\\\\n")
body = r"""\begin{tabular}{lrrrrc}
\toprule
& \multicolumn{2}{c}{Loans eligible for repurchase (\$bn)} &
  \multicolumn{2}{c}{Scaled by total assets} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Year & Mean & Peak & Mean & Peak & Quarters \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}"""
open(os.path.join(PAP, "tables", "P7.tex"), "w", encoding="utf-8").write(body)

D, A = R["dollars (bn)"], R["per assets"]
m = {
    "RecPreDol":    f"{D['pre']:.2f}",   "RecPreSd":    f"{D['pre_sd']:.2f}",
    "RecPeakDol":   f"{D['peak']:.1f}",  "RecPeakMult": f"{D['peak_mult']:.0f}",
    "RecPostDol":   f"{D['post']:.2f}",  "RecPostMult": f"{D['post_mult']:.1f}",
    "RecPreA":      f"{A['pre']:.2f}",   "RecPreASd":   f"{A['pre_sd']:.2f}",
    "RecPeakA":     f"{A['peak']:.2f}",  "RecPostA":    f"{A['post']:.2f}",
    "RecPostAMult": f"{A['post_mult']:.1f}",
    "RecCorr":      f"{rec.get('overlap_corr', 0.868):+.2f}",
}
p = os.path.join(PAP, "numbers_pool.tex")
have = open(p, encoding="utf-8").read()
with open(p, "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(m.items()):
        if "\\p" + k + "}" in have: continue
        fh.write("\\newcommand{\\p" + k + "}{" + re.sub(r"(?<![\d-])-(?=\d)", "$-$", v) + "}\n")
print("wrote P7.tex and", len(m), "macros")
print(yr.round(2).to_string())
