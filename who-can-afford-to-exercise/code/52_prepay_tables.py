"""Tables and macros for the prepayment-incidence section."""
import pandas as pd, numpy as np, os, json, re
from config import OUT, PAPER as PAP

TAB = os.path.join(PAP, "tables")
R = json.load(open(os.path.join(OUT, "results_prepay.json")))
g = pd.read_parquet(os.path.join(OUT, "pool_month.parquet"))
mac = {}
_M = re.compile(r"(?<=[\s(\[&{,])-(?=\d)")
def W(n, b): open(os.path.join(TAB, n), "w", encoding="utf-8").write(_M.sub("$-$", b))
def star(p): return "$^{***}$" if p < .01 else "$^{**}$" if p < .05 else "$^{*}$" if p < .10 else ""

# ---- Panel A: quartile table
g["q"] = pd.qcut(g.nb_share, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
                 duplicates="drop")
qt = (g.pivot_table(index="q", columns="post", values="buy_rate",
                    observed=True) * 100).round(1)
qn = g.groupby("q", observed=True).nb_share.mean().round(2)
rows = ""
for i in qt.index:
    rows += (f"{i} & {qn[i]:.2f} & {qt.loc[i,0.0]:.1f}\\% & {qt.loc[i,1.0]:.1f}\\% & "
             f"{qt.loc[i,1.0]-qt.loc[i,0.0]:+.1f} \\\\\n")
# ---- Panel B: regressions
regs = ""
for lab, key in [(r"Pool, month", "pool + month"),
                 (r"Pool, coupon-bin $\times$ month", "pool + coupon-bin x month"),
                 (r"Pool, month; multi-issuer pools only", "pool + month, mixed pools")]:
    v = R[key]
    regs += f"{lab} & {v['coef']:.3f}{star(v['p'])} & ({v['se']:.3f}) \\\\\n"
W("P6.tex", r"""\begin{tabular}{lrrrr}
\toprule
\multicolumn{5}{l}{\textit{Panel A: buyout-driven prepayment of delinquent principal}}\\
\addlinespace
Quartile of nonbank share & Mean share & 2019--Feb 2020 & Mar--Sep 2020 & Change (pp)\\
\midrule
""" + rows + r"""\addlinespace
\multicolumn{5}{l}{\textit{Panel B: coefficient on (nonbank share $\times$ Post)}}\\
\addlinespace
Fixed effects & Coef. & (s.e.) & & \\
\midrule
""" + regs + r"""\bottomrule
\end{tabular}""")

mac["PrepayCoef"] = f"{R['pool + coupon-bin x month']['coef']:.3f}"
mac["PrepayCoefSE"] = f"{R['pool + coupon-bin x month']['se']:.3f}"
mac["PrepayPP"] = f"{abs(R['implied'])*100:.0f}"
mac["PrepayDep"] = f"{R['rate_depository']:.1f}"
mac["PrepayNb"] = f"{R['rate_nonbank']:.1f}"
mac["PrepayGap"] = f"{R['gap_pp']:.1f}"
mac["PrepayValFive"] = f"{R['value_5pt']:.2f}"
mac["PoolMonths"] = f"{len(g):,}"
mac["PoolsPrepay"] = f"{g.pool_id.nunique():,}"
mac["MixedPoolShare"] = f"{(g.n_iss>1).mean()*100:.0f}"
mac["NbShareMean"] = f"{g.nb_share.mean():.2f}"
d = R["dispersion"]["corr"]
vals = [v for v in d.values() if v == v]
mac["CorrMin"] = f"{min(vals):.2f}"; mac["CorrMax"] = f"{max(vals):.2f}"
sd = R["dispersion"]["sd_nb"]
mac["SdNbShare"] = f"{np.mean([v for v in sd.values() if v==v]):.2f}"
q1 = qt.iloc[0]; q4 = qt.iloc[-1]
mac["QOneLow"] = f"{q1[0.0]:.1f}"; mac["QOneHigh"] = f"{q1[1.0]:.1f}"
mac["QFourLow"] = f"{q4[0.0]:.1f}"; mac["QFourHigh"] = f"{q4[1.0]:.1f}"

with open(os.path.join(PAP, "numbers_pool.tex"), "a", encoding="utf-8") as fh:
    fh.write("\n")
    for k, v in sorted(mac.items()):
        s = re.sub(r"(?<![\d-])-(?=\d)", "$-$", str(v))
        fh.write("\\newcommand{\\p" + k + "}{" + s + "}\n")
print(f"appended {len(mac)} macros; wrote P6.tex")
for k in sorted(mac): print(f"   \\p{k} = {mac[k]}")
