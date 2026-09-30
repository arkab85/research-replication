"""Rebuild P8, P9 and the APM table from the matched-window data, so no table
disagrees with the text."""
import pandas as pd, numpy as np, os, json
from config import OUT, PAPER as PAP
T = os.path.join(PAP, "tables")
w = pd.read_csv(os.path.join(OUT, "wedge_matched.csv"))


def rows(sub, head):
    L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{Reported balance (\$bn)} & "
         r"\multicolumn{2}{c}{Options vesting, Jan--Sep} & "
         r"\multicolumn{2}{c}{Exercise rate}\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Issuer & As of & 2019 & 2020 & 2019 & 2020 & 2019 & 2020\\", r"\midrule"]
    for _, r in sub.iterrows():
        L.append(f"{r.entity} & {r['asof']} & {r.rep2019:.2f} & {r.rep2020:.2f} & "
                 f"{int(r.vested2019):,} & {int(r.vested2020):,} & "
                 f"{r.rate2019*100:.1f}\\% & {r.rate2020*100:.1f}\\% \\\\")
    L += [r"\midrule",
          r"\multicolumn{2}{l}{\textit{Growth 2019--2020}} & \multicolumn{2}{c}{balance}"
          r" & \multicolumn{2}{c}{vesting flow} & \multicolumn{2}{c}{wedge}\\"]
    for _, r in sub.iterrows():
        L.append(f"\\quad {r.entity} & & \\multicolumn{{2}}{{c}}{{{r.rep_mult:.1f}$\\times$}} & "
                 f"\\multicolumn{{2}}{{c}}{{{r.flow_mult:.1f}$\\times$}} & "
                 f"\\multicolumn{{2}}{{c}}{{{r.wedge:.1f}}}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


nb = w[w.type == "nonbank"]
dep = w[w.type == "depository"]
open(os.path.join(T, "P8.tex"), "w", encoding="utf-8").write(rows(nb, "nonbank"))
open(os.path.join(T, "P9.tex"), "w", encoding="utf-8").write(rows(dep, "depository"))
print("P8 (nonbanks):")
print(nb[["entity", "rep_mult", "flow_mult", "wedge", "d_rate"]].round(2).to_string(index=False))
print("\nP9 (depositories):")
print(dep[["entity", "net", "rep_mult", "flow_mult", "wedge", "d_rate"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- APM 20-07 table
r = json.load(open(os.path.join(OUT, "results_apm2007.json")))


def cell(k):
    c, s, p = r[k]
    st = "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .1 else ""
    num = f"{abs(c):.3f}"
    return ("$-$" + num + st if c < 0 else num + st), f"({s:.3f})"


L = [r"\begin{tabular}{lcc}", r"\toprule",
     r"& Issuer, month & Issuer, pool $\times$ month\\",
     r"& \multicolumn{2}{c}{fixed effects}\\",
     r"\cmidrule(lr){2-3}", r"\midrule"]
a1, s1 = cell("nb_shock"); a2, s2 = cell("nb_shock_pool")
b1, t1 = cell("nb_apm"); b2, t2 = cell("nb_apm_pool")
L += [f"Nonbank $\\times$ March--June 2020 & {a1} & {a2} \\\\",
      f" & {s1} & {s2} \\\\", r"\addlinespace",
      f"Nonbank $\\times$ July--September 2020 & {b1} & {b2} \\\\",
      f" & {t1} & {t2} \\\\", r"\midrule",
      r"Loan controls & Yes & Yes\\", r"\bottomrule", r"\end{tabular}"]
open(os.path.join(T, "P10.tex"), "w", encoding="utf-8").write("\n".join(L))
print("\nwrote P8.tex, P9.tex, P10.tex")
